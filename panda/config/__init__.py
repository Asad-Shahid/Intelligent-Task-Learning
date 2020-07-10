import argparse

from utils import str2bool, str2list


def argparser():
    parser = argparse.ArgumentParser('Grasping Environment', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Grasping
    import config.grasping as grasp
    grasp.add_argument(parser)

    # training algorithm
    parser.add_argument('--algo', type=str, default='ppo',
                        choices=['sac', 'ppo'])

    # vanilla rl
    parser.add_argument('--rl_hid_size', type=int, default= 64)
    parser.add_argument('--rl_activation', type=str, default='relu',
                        choices=['relu', 'elu', 'tanh'])
    parser.add_argument('--tanh_policy', type=str2bool, default=True)

    # observation normalization
    parser.add_argument('--ob_norm', type=str2bool, default=True)
    parser.add_argument('--clip_obs', type=float, default=200, help='the clip range of observation')
    parser.add_argument('--clip_range', type=float, default=5, help='the clip range after normalization of observation')

    # off-policy rl
    parser.add_argument('--buffer_size', type=int, default=int(1e5), help='the size of the buffer')
    parser.add_argument('--discount_factor', type=float, default=0.99, help='the discount factor')
    parser.add_argument('--lr_actor', type=float, default=3e-4, help='the learning rate of the actor')
    parser.add_argument('--lr_critic', type=float, default=3e-4, help='the learning rate of the critic')
    parser.add_argument('--polyak', type=float, default=0.995, help='the average coefficient')

    # training
    parser.add_argument('--is_train', type=str2bool, default=False)
    parser.add_argument('--num_batches', type=int, default=60, help='the times to update the network per epoch')
    parser.add_argument('--batch_size', type=int, default=512, help='the sample batch size')
    parser.add_argument('--max_grad_norm', type=float, default=100)
    parser.add_argument('--max_global_step', type=int, default=int(3e6))
    parser.add_argument('--gpu', type=int, default=0)

    # ppo
    parser.add_argument('--clip_param', type=float, default=0.2)
    parser.add_argument('--value_loss_coeff', type=float, default=0.5)
    parser.add_argument('--action_loss_coeff', type=float, default=1e-2)
    parser.add_argument('--entropy_loss_coeff', type=float, default=1e-4)
    parser.add_argument('--rollout_length', type=int, default=1800)
    parser.add_argument('--gae_lambda', type=float, default=0.95)

    # sac
    parser.add_argument('--reward_scale', type=float, default=1.0, help='reward scale')

    # log
    parser.add_argument('--log_interval', type=int, default=1)
    parser.add_argument('--evaluate_interval', type=int, default=10)
    parser.add_argument('--ckpt_interval', type=int, default=200)
    parser.add_argument('--log_root_dir', type=str, default='log')
    parser.add_argument('--wandb', type=str2bool, default=True,
                        help='set it True if you want to use wandb')

    # evaluation
    parser.add_argument('--ckpt_num', type=int, default=None)
    parser.add_argument('--num_eval', type=int, default=10)
    parser.add_argument('--num_record_samples', type=int, default=1, help='number of trajectories to collect during eval')


    # misc
    parser.add_argument('--prefix', type=str, default='rl')
    parser.add_argument('--suffix', type=str, default='')
    parser.add_argument('--notes', type=str, default='')
    parser.add_argument('--seed', type=int, default=111, help='Random seed')

    args, unparsed = parser.parse_known_args()

    return args, unparsed
