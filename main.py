from gym import spaces                  # 导入spaces
from random import random    # 随机策略时用到
from gym import Env
import gym
from gridworld import *      # 可以导入各种格子世界环境


class Agent(object):
    """
    个体类，存在属性：环境，Q表（字典套字典），观测值
    具有和环境互动方法：执行策略，执行行为，学习（改变Q表，改善策略）
    """
    def __init__(self, env: Env):
        """
        初始化个体类，包括环境，Q表，所处观测值
        """
        self.env = env
        self.Q = {}
        self._initAgent()
        self.state = None   #当前个体观测，可写成obs

    # using simple decaying epsilon greedy exploration
    def _curPolicy(self, s, episode_num, use_epsilon):
        epsilon = 1.00 / (episode_num+1)
        Q_s = self.Q[s]
        str_act = "unknown"
        rand_value = random()
        action = None
        if use_epsilon and rand_value < epsilon:  
            action = self.env.action_space.sample()
        else:
            str_act = max(Q_s, key=Q_s.get)
            action = int(str_act)
        return action
    
    def performPolicy(self, s, episode_num, use_epsilon):
        """
        对状态执行策略，得到行为，不会影响到环境
        ε-greedy 策略，使用衰减ε，使训练能够收敛
        s:状态
        episode_num:当前训练轮次，用以衰减计算
        use_epsilon:是否使用ε

        """
        return self._curPolicy(s, episode_num, use_epsilon)

    def act(self, a):
        """
        执行行为，将会影响环境，得到即时奖励已经下一状态
        """
        return self.env.step(a)

    # sarsa learning
    def learning(self, gamma, alpha, max_episode_num):
        total_time, time_in_episode, num_episode = 0, 0, 0
        
        while num_episode <= max_episode_num: # 设置终止条件
            self.state = self.env.reset()    # 环境初始化
            s0 = self._get_state_name(self.state) # 获取个体对于观测的命名
            #self.env.render()                # 显示UI界面
            a0 = self.performPolicy(s0, num_episode, use_epsilon = True)

            time_in_episode = 0
            is_done = False
            while not is_done:               # 针对一个Episode内部
                # a0 = self.performPolicy(s0, num_episode)
                s1, r1, is_done, info = self.act(a0) # 执行行为
                #self.env.render()            # 更新UI界面
                s1 = self._get_state_name(s1)# 获取个体对于新状态的命名
                self._assert_state_in_Q(s1, randomized = True)
                # 获得A'
                # 在下行代码中添加参数use_epsilon = False即变成Q学习算法
                a1 = self.performPolicy(s1, num_episode, use_epsilon=True)
                old_q = self._get_Q(s0, a0)  
                q_prime = self._get_Q(s1, a1)
                td_target = r1 + gamma * q_prime  
                #alpha = alpha / num_episode
                new_q = old_q + alpha * (td_target - old_q)
                self._set_Q(s0, a0, new_q)

                if num_episode == max_episode_num: # 终端显示最后Episode的信息
                    print("t:{0:>2}: s:{1}, a:{2:2}, s1:{3}".\
                        format(time_in_episode, s0, a0, s1))

                s0, a0 = s1, a1
                time_in_episode += 1

            print("Episode {0} takes {1} steps.".format(
                num_episode, time_in_episode)) # 显示每一个Episode花费了多少步
            total_time += time_in_episode
            num_episode += 1
        return

    def _get_state_name(self, state):
        """
        将观测值转为字典键
        """
        return str(state)

    def _is_state_in_Q(self, s):
        """
        判断s的Q值是否存在
        """
        return self.Q.get(s) is not None

    def _init_state_value(self, s_name, randomized = True):
        """
        初始化某状态的Q值
        """
        if not self._is_state_in_Q(s_name):
            self.Q[s_name] = {}
            for action in range(self.env.action_space.n): # 针对其所有可能行为
                default_v = random() / 10 if randomized is True else 0.0
                self.Q[s_name][action] = default_v

    def _assert_state_in_Q(self, s, randomized=True):
        """
        确保某状态Q值存在
        """
        # cann't find the state
        if not self._is_state_in_Q(s):
            self._init_state_value(s, randomized)

    def _get_Q(self, s, a):
        self._assert_state_in_Q(s, randomized=True)
        return self.Q[s][a]

    def _set_Q(self, s, a, value):
        self._assert_state_in_Q(s, randomized=True)
        self.Q[s][a] = value

    def _initAgent(self):
        self.state = self.env.reset()
        s_name = self._get_state_name(self.state)
        self._assert_state_in_Q(s_name, randomized = False)
        
def main():
    env = SimpleGridWorld()
    agent = Agent(env)
    print("Learning...")  
    agent.learning(gamma=0.9, 
                   alpha=0.1, 
                   max_episode_num=2)

if __name__ == "__main__":
    main()

