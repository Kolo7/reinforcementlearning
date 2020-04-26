import argparse
import os


import matplotlib.pyplot as plt
import numpy as np
#import tensorflow as tf
import tensorflow.compat.v1 as tf
from dqn_agent import DQN
from RSSmodel import *

dicts = {'model_path': None,
        'save_path': './models/',
        'double_q': True,
        'log_every': 500,
        'gpu': 0,
        'seed': 31,
        'max_ep': 3000,
        'max_episode_steps': 5,
        'test_ep': 50,
        'init_epsilon': 0.75,
        'final_epsilon': 0.2,
        'buffer_size': 50000,
        'lr': 1e-4,
        'batch_size': 128,
        'gamma': 0.99,
        'target_network_update': 1000}

def main(args):
    set_random_seed(args.seed)

    #env = gym.make("CartPole-v0")
    env = RssEnv()
    agent = DQN(env, args)
    agent.construct_model(args.gpu)

    # load pre-trained models or init new a model.
    saver = tf.train.Saver(max_to_keep=1)
    if args.model_path is not None:
        saver.restore(agent.sess, args.model_path)
        ep_base = int(args.model_path.split('_')[-1])
        best_mean_rewards = float(args.model_path.split('/')[-1].split('_')[0])
    else:
        agent.sess.run(tf.global_variables_initializer())
        ep_base = 0
        best_mean_rewards = None

    rewards_history, steps_history = [], []
    train_steps = 0
    # Training
    for ep in range(args.max_ep):
        state = env.reset()
        ep_rewards = 0
        for step in range(args.max_episode_steps):
            # pick action
            action = agent.sample_action(state, policy='egreedy')
            # execution action.
            next_state, reward = env.step(action)
            train_steps += 1
            ep_rewards += reward
            # modified reward to speed up learning
            # reward = 0.1 if not done else -1
            # learn and Update net parameters
            agent.learn(state, action, reward, next_state)

            state = next_state
            """
            if done:
                break
            """
        steps_history.append(train_steps)
        if not rewards_history:
            rewards_history.append(ep_rewards)
        else:
            rewards_history.append(
                rewards_history[-1] * 0.9 + ep_rewards * 0.1)

        # decay epsilon
        if agent.epsilon > args.final_epsilon:
            agent.epsilon -= (args.init_epsilon - args.final_epsilon) / args.max_ep

        # evaluate during training
        if ep % args.log_every == args.log_every-1:
            total_reward = 0
            for i in range(args.test_ep):
                state = env.reset()
                for j in range(args.max_episode_steps):
                    action = agent.sample_action(state, policy='greedy')
                    state, reward = env.step(action)
                    total_reward += reward
                    """
                    if done:
                        break
                    """
            current_mean_rewards = total_reward / args.test_ep
            print('Episode: %d Average Reward: %.2f' %
                  (ep + 1, current_mean_rewards))
            # save model if current model outperform the old one
            if best_mean_rewards is None or (current_mean_rewards >= best_mean_rewards):
                best_mean_rewards = current_mean_rewards
                if not os.path.isdir(args.save_path):
                    os.makedirs(args.save_path)
                save_name = args.save_path + str(round(best_mean_rewards, 2)) \
                    + '_' + str(ep_base + ep + 1)
                saver.save(agent.sess, save_name)
                print('Model saved %s' % save_name)
        env._d_move()

    plt.plot(steps_history, rewards_history)
    plt.xlabel('steps')
    plt.ylabel('running avg rewards')
    plt.show()


def set_random_seed(seed):
    np.random.seed(seed)
    tf.set_random_seed(seed)
    #tf.random.set_seed(seed)



class Args():
    def __init__(self, dicts):
        self.model_path = dicts['model_path']
        self.save_path = dicts['save_path']
        self.double_q = dicts['double_q']
        self.log_every = dicts['log_every']
        self.gpu = dicts['gpu']
        self.seed = dicts['seed']
        self.max_ep = dicts['max_ep']
        self.max_episode_steps = dicts['max_episode_steps']
        self.test_ep = dicts['test_ep']
        self.init_epsilon = dicts['init_epsilon']
        self.final_epsilon = dicts['final_epsilon']
        self.buffer_size = dicts['buffer_size']
        self.lr = dicts['lr']
        self.batch_size = dicts['batch_size']
        self.gamma = dicts['gamma']
        self.target_network_update = dicts['target_network_update']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model_path', default=None,
        help='Whether to use a saved model. (*None|model path)')
    parser.add_argument(
        '--save_path', default='./models/',
        help='Path to save a model during training.')
    parser.add_argument(
        '--double_q', default=True, help='enable or disable double dqn')
    parser.add_argument(
        '--log_every', default=500, help='Log and save model every x episodes')
    parser.add_argument(
        '--gpu', default=-1,
        help='running on a specify gpu, -1 indicates using cpu')
    parser.add_argument(
        '--seed', default=31, help='random seed')

    parser.add_argument(
        '--max_ep', type=int, default=2000, help='Number of training episodes')
    parser.add_argument(
        '--test_ep', type=int, default=50, help='Number of test episodes')
    parser.add_argument(
        '--init_epsilon', type=float, default=0.75, help='initial epsilon')
    parser.add_argument(
        '--final_epsilon', type=float, default=0.2, help='final epsilon')
    parser.add_argument(
        '--buffer_size', type=int, default=50000, help='Size of memory buffer')
    parser.add_argument(
        '--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument(
        '--batch_size', type=int, default=128, help='Size of training batch')
    parser.add_argument(
        '--gamma', type=float, default=0.99, help='Discounted factor')
    parser.add_argument(
        '--target_network_update', type=int, default=1000,
        help='update frequency of target network.')
    main(parser.parse_args())



    
