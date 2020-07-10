""" Base code for RL training. Collects rollouts and updates policy networks. """

import os
from time import time
from collections import defaultdict, OrderedDict
import gzip
import pickle
import h5py
import torch
import wandb
import numpy as np
from tqdm import tqdm, trange
from rl.policies import MlpActor, MlpCritic
from rl.ppo_agent import PPOAgent
from rl.sac_agent import SACAgent
from rl.rollouts import RolloutRunner
from utils.logger import logger
from utils.pytorch import get_ckpt_path
from utils.mpi import mpi_sum
from environments import make

class Trainer():
    """
    Trainer class for SAC and PPO in PyTorch.
    """

    def __init__(self, config):
        """
        Initializes class with the configuration.
        """
        self._config = config
        self._is_chef = config.is_chef

        # create a new environment
        self._env = make("PandaGrasp", config)
        ob_space = self._env.observation_space # e.g. OrderedDict([('object-state', [10]), ('robot-state', [36])])
        ac_space = self._env.action_space # e.g. ActionSpace(shape=OrderedDict([('default', 8)]),minimum=-1.0, maximum=1.0)
        print('***', ac_space)

        # get actor and critic networks
        actor, critic = MlpActor, MlpCritic

        # build up networks for PPO agent
        if self._config.algo == 'sac':
            self._agent = SACAgent(config, ob_space, ac_space, actor, critic)
        else:
            self._agent = PPOAgent(config, ob_space, ac_space, actor, critic)

        # build rollout runner
        self._runner = RolloutRunner(config, self._env, self._agent)

        # setup log
        if self._is_chef and self._config.is_train:
            exclude = ['device']
            if not self._config.wandb:
                os.environ['WANDB_MODE'] = 'dryrun'

        # Weights and Biases (wandb) is used for logging, set the account details below or dry run above
            # user or team name
            entity = 'panda'
            # project name
            project = 'robo'

            wandb.init(
                resume=config.run_name,
                project=project,
                config={k: v for k, v in config.__dict__.items() if k not in exclude},
                dir=config.log_dir,
                entity=entity,
                notes=config.notes
            )

    def _save_ckpt(self, ckpt_num, update_iter):
        """
        Save checkpoint to log directory.

        Args:
            ckpt_num: number appended to checkpoint name. The number of environment step is used in this code.
            update_iter: number of policy update. It will be used for resuming training.
        """
        ckpt_path = os.path.join(self._config.log_dir, 'ckpt_%08d.pt' % ckpt_num)
        state_dict = {'step': ckpt_num, 'update_iter': update_iter}
        state_dict['agent'] = self._agent.state_dict()
        torch.save(state_dict, ckpt_path)
        logger.warn('Save checkpoint: %s', ckpt_path)

        replay_path = os.path.join(self._config.log_dir, 'replay_%08d.pkl' % ckpt_num)
        with gzip.open(replay_path, 'wb') as f:
            replay_buffers = {'replay': self._agent.replay_buffer()}
            pickle.dump(replay_buffers, f)

    def _load_ckpt(self, ckpt_num=None):
        """
        Loads checkpoint with index number @ckpt_num. If @ckpt_num is None,
        it loads and returns the checkpoint with the largest index number.
        """
        ckpt_path, ckpt_num = get_ckpt_path(self._config.log_dir, ckpt_num)

        if ckpt_path is not None:
            logger.warn('Load checkpoint %s', ckpt_path)
            ckpt = torch.load(ckpt_path) # ckpt is a dict with keys (step, update_iter, agent)
            self._agent.load_state_dict(ckpt['agent'])

            if self._config.is_train:
                replay_path = os.path.join(self._config.log_dir, 'replay_%08d.pkl' % ckpt_num)
                logger.warn('Load replay_buffer %s', replay_path)
                with gzip.open(replay_path, 'rb') as f:
                    replay_buffers = pickle.load(f)
                    self._agent.load_replay_buffer(replay_buffers['replay'])

            return ckpt['step'], ckpt['update_iter']
        else:
            logger.warn('Randomly initialize models')
            return 0, 0

    def _log_ep(self, step, ep_info):
        """
        Logs episode information to wandb.
        Args:
            step: the number of environment steps.
            ep_info: episode information to log, such as reward, episode time.
        """
        for k, v in ep_info.items():
            wandb.log({'train_ep/%s' % k: np.mean(v)}, step=step)
            wandb.log({'train_ep_max/%s' % k: np.max(v)}, step=step)

    def _log_train(self, step, train_info):
        """
        Logs training information to wandb.
        Args:
            step: the number of environment steps.
            train_info: training information to log, such as loss, gradient.
        """
        for k, v in train_info.items():
            if np.isscalar(v) or (hasattr(v, 'shape') and np.prod(v.shape) == 1):
                wandb.log({'train_rl/%s' % k: v}, step=step)
            else:
                wandb.log({'train_rl/%s' % k: [wandb.Image(v)]}, step=step)

    def _log_test(self, step, ep_info):
        """
        Logs episode information during testing to wandb.
        Args:
            step: the number of environment steps.
            ep_info: episode information to log, such as reward, episode time.
        """
        if self._config.is_train:
            for k, v in ep_info.items():
                wandb.log({'test_ep/%s' % k: np.mean(v)}, step=step)

    def train(self):
        """ Trains an agent. """
        config = self._config
        num_batches = config.num_batches

        # load checkpoint
        step, update_iter = self._load_ckpt()

        # sync the networks across the cpus
        self._agent.sync_networks()

        logger.info("Start training at step=%d", step)
        if self._is_chef:
            pbar = tqdm(initial=step, total=config.max_global_step, desc=config.run_name)
            ep_info = defaultdict(list)

        # decide how many episodes or how long rollout to collect
        if self._config.algo == 'sac':
            run_ep_max = 1
            run_step_max = 600
        else:
            run_step_max = self._config.rollout_length
            run_ep_max = 3

        # dummy run for preventing weird error in a cold run
        self._runner.run_episode()

        st_time = time()
        st_step = step
        log_step = 0
        while step < config.max_global_step:
            # collect rollouts
            run_ep = 0
            run_step = 0
            while run_step < run_step_max and run_ep < run_ep_max:
                # return one episode and info{len:, rew:, qpos:[], success:} about that episode
                rollout, info = self._runner.run_episode()
                run_step += info['len']
                run_ep += 1
                log_step += info['len']

                for k, v in info.items():
                    if isinstance(v, list):
                        ep_info[k].extend(v)
                    else:
                        ep_info[k].append(v)

                self._log_ep(log_step, ep_info)
                ep_info = defaultdict(list)
                logger.info('rollout: %s', {k: v for k, v in info.items() if not 'qpos' in k})
                self._agent.store_episode(rollout)
                self._update_normalizer(rollout)

            step_per_batch = mpi_sum(run_step)

            # train an agent
            logger.info('Update networks %d', update_iter)
            train_info = self._agent.train()
            logger.info('Update networks done')

            step += step_per_batch
            update_iter += 1

            # log training and episode information or evaluate
            if self._is_chef:
                pbar.update(step_per_batch)
                if update_iter % config.log_interval == 0:
                    train_info.update({
                        'sec': (time() - st_time) / config.log_interval,
                        'steps_per_sec': (step - st_step) / (time() - st_time),
                        'update_iter': update_iter
                    })
                    st_time = time()
                    st_step = step
                    self._log_train(step, train_info)

                if update_iter % config.evaluate_interval == 1:
                    logger.info('Evaluate at %d', update_iter)
                    rollout, info = self._evaluate(step=step)##
                    self._log_test(step, info)

                if update_iter % config.ckpt_interval == 0:
                    self._save_ckpt(step, update_iter)

        logger.info('Reached %s steps. worker %d stopped.', step, config.rank)

    def _update_normalizer(self, rollout):
        """ Updates normalizer with @rollout. """
        if self._config.ob_norm:
            self._agent.update_normalizer(rollout['ob'])

    def _evaluate(self, step=None, idx=None):
        """
        Runs one rollout if in eval mode (@idx is not None).
        Runs num_record_samples rollouts if in train mode (@idx is None).

        Args:
            step: the number of environment steps.
        """
        for i in range(self._config.num_record_samples):
            rollout, info = self._runner.run_episode(is_train=False)

            if idx is not None:
                break
        logger.info('rollout: %s', {k: v for k, v in info.items() if not 'qpos' in k})
        return rollout, info

    def evaluate(self):
        """ Evaluates an agent stored in chekpoint with @self._config.ckpt_num. """

        step, update_iter = self._load_ckpt(ckpt_num=self._config.ckpt_num)
        logger.info('Run %d evaluations at step=%d, update_iter=%d', self._config.num_eval, step, update_iter)

        for i in trange(self._config.num_eval):
            logger.warn("Evalute run %d", i+1)
            rollout, info = self._evaluate(step=step, idx=i)
