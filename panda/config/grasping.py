from utils import str2bool, str2intlist



def add_argument(parser):
    """
    Adds a list of arguments to argparser for the lift environment.
    """

    # training scene

    parser.add_argument('--mode', type=int, default=1,
                        help='1: nominal cube scene, 2: collision avoidance scene')


    # mujoco simulation

    parser.add_argument('--table_full_size', type=float, default=(0.35, 0.46, 0.02),
                        help='x, y, and z dimensions of the table')
    parser.add_argument('--gripper_type', type=str, default='PandaGripper',
                        help='Gripper type of robot')
    parser.add_argument('--gripper_visualization', type=str2bool, default=True,
                        help='using gripper visualization')

    # rendering

    parser.add_argument('--render_collision_mesh', type=str2bool, default=False,
                        help='if rendering collision meshes in camera')
    parser.add_argument('--render_visual_mesh', type=str2bool, default=True,
                        help='if rendering visual meshes in camera')

    # episode settings

    parser.add_argument('--horizon', type=int, default=600,
                        help='Every episode lasts for exactly @horizon timesteps')
    parser.add_argument('--ignore_done', type=str2bool, default=True,
                        help='if never terminating the environment (ignore @horizon)')

    # controller

    parser.add_argument('--control_freq', type=int, default= 250,
                        help='control signals to receive in every simulated second, sets the amount of simulation time that passes between every action input')


def get_default_config():
    """
    Gets default configurations for the lift environment.
    """
    import argparse
    parser = argparse.ArgumentParser("Default Configuration for lift Environment")
    add_argument(parser)

    config = parser.parse_args([])
    return config
