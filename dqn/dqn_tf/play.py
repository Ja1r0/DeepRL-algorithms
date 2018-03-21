import tensorflow as tf
import tensorflow.contrib.layers as layers
import random
import agent
import utils
from atari_wrappers import *
import gym
from gym import wrappers
import os.path as osp
from collections import namedtuple





def play(env,session,timesteps_num):
    ###################
    # build q network #
    ###################
    def build_cnn(input, act_num, scope, reuse=False):
        with tf.variable_scope(scope, reuse=reuse):
            out = input
            with tf.variable_scope('convnet'):
                out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
                out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
                out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
            out = layers.flatten(out)
            with tf.variable_scope('action_value'):
                out = layers.fully_connected(out, num_outputs=512, activation_fn=tf.nn.relu)
                out = layers.fully_connected(out, num_outputs=act_num, activation_fn=None)
            return out

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return utils.get_wrapper_by_name(env, "Monitor").get_total_steps() >= timesteps_num
    ##########################
    # learning rate schedule #
    ##########################
    iterations_num=float(timesteps_num)/4.0
    lr_multiplier=1.0
    lr_schedule=utils.PiecewiseSchedule([
        (0 , 1e-4*lr_multiplier),
        (iterations_num/10 , 1e-4*lr_multiplier),
        (iterations_num/2 , 5e-5*lr_multiplier)
    ],outside_value=5e-5*lr_multiplier)
    #################
    # set optimizer #
    #################
    OptimizerSepc = namedtuple('OptimizerSpec', ['constructor', 'kwargs', 'lr_schedule'])
    optimizer=OptimizerSepc(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )
    ########################
    # exploration schedule #
    ########################
    exploration_schedule=utils.PiecewiseSchedule([
        (0 , 1.0),
        (1e6 , 0.1),
        (iterations_num/2 , 0.01)
    ],outside_value=0.01)
    #################
    # play the game #
    #################
    agent.learn_by_dqn(env=env,
                       q_net=build_cnn,
                       optimizer_spec=optimizer,
                       session=session,
                       exploration=exploration_schedule,
                       replay_buffer_size=1000000,
                       batch_size=32,
                       gamma=0.99,
                       learn_start=50000,
                       learn_freq=4,
                       history_frames_num=4,
                       target_update_freq=10000,
                       grad_norm_clipping=10,
                       stop_criterion=stopping_criterion
                       )

def set_global_seeds(i):
    tf.set_random_seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config=tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1
    )
    session=tf.Session(config=tf_config)
    print('AVAILABLE GPUS:',utils.get_available_gpus())
    return session

def get_env(task, seed):
    env_id = task.env_id
    env = gym.make(env_id)
    set_global_seeds(seed)
    env.seed(seed)
    expt_dir = '/tmp/hw3_vid_dir2/'
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True)
    env = wrap_deepmind(env)
    return env

if __name__ == '__main__':
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game.
    task = benchmark.tasks[3]

    # Run training
    seed = 0  # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)
    session = get_session()
    play(env, session, timesteps_num=task.max_timesteps)