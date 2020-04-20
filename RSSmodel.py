import math
import numpy as np
import random
class RssEnv:
    def __init__(self):
        # 路径损耗
        self.gamma = 2.5
        # 行为功率
        self.a_powers = [0, 0.02, 0.01]
        # Ps
        self.source_power = 0.1
        # Pr
        self.transmit_powers = random.sample([0.04, 0.05, 0.06], 2)
        # 高斯白噪声方差
        self.sigma2 = 0.1
        
        #self.alpha = 0.7
        #self.delta = 0.5
        # 行为集合
        self.actions = [0, 1, 2]
        
        self.s_point = (0,0)
        #self.r1_point = (0.5,0.5)
        #self.r2_point = (-0.6,-0.6)
        self.r1_point = (random.uniform(-1,1), random.uniform(-1,1))
        self.r2_point = (random.uniform(-1,1), random.uniform(-1,1))
        #self.d_point = (0.06,0.06)
        self.d_point = (random.uniform(-1,1), random.uniform(-1,1))
        self.transmit_powers = random.sample([0.04, 0.05, 0.06], 2)
        

    def reset(self):
        """
        完成对环境的初始化，返回当前的状态，一般是(0,0,0)
        """
        
        
        #s, r = self.step(random.sample(self.actions, 1)[0])
        
        return s

    def action_space_sample(self):
        """
        获得一个随机的行为
        """
        return random.sample(self.actions, 1)[0]

    def step(self, a):
        """
        对当前状态采取行动a，获得下一步状态，奖励
        """
        g3 = self._g_channel_gain(self._cal_distance(self.s_point, self.d_point))
        snr1 = self._snr1(g3)
        if a == 0:
            g1 = 0.0
            g2 = 0.0
            snr2 = 0
            I = self._i_S_D(snr1)
        elif a == 1:
            g1 = self._g_channel_gain(self._cal_distance(self.s_point, self.r1_point))
            g2 = self._g_channel_gain(self._cal_distance(self.r1_point, self.d_point))
            snr2 = self._snr2(g2, self.transmit_powers[0])
            I = self._i_S_D_R(g1, g2, snr1, self.transmit_powers[0])
        else:
            g1 = self._g_channel_gain(self._cal_distance(self.s_point, self.r2_point))
            g2 = self._g_channel_gain(self._cal_distance(self.r2_point, self.d_point))
            snr2 = self._snr2(g2, self.transmit_powers[1])
            I = self._i_S_D_R(g1, g2, snr1, self.transmit_powers[0])
        s = snr1, snr2, I
        r = self._utility(s, a)
        return s, r
    
    def get_all_action(self):
        return self.actions
    
    def _utility(self, s, a):
        """
        获得状态行为效用
        """
        snr1, snr2, I = s
        return math.log(I) - self.a_powers[a]
    
    def _cal_distance(self, pa, pb):
        return math.sqrt((pa[0] - pb[0])**2 + (pa[1] - pb[1])**2)
        
    def _g_channel_gain(self, x):
        mu = 0
        sigma2 = 0.16
        y = 1.0/math.sqrt(2*math.pi*sigma2) * math.exp(-(x-mu)**2 /(2*sigma2))
        #a = np.random.normal(mu, sigma2**0.5)
        return y
    
    def _snr1(self, g3):
        print('Ps*g3/delta**2 => {} * {} / {}**2'.format(self.source_power, g3, self.sigma2))
        return self.source_power * g3/ self.sigma2
    
    def _snr2(self, g2, pr):
        print('Pr*g2/delta**2 => {} * {} / {}**2'.format(pr, g2, self.sigma2))
        return pr * g2 / self.sigma2
    
    def _i_S_D(self, snr1):
        #print('log2(1+{})'.format(snr1))
        #print('math.log2(1+{})'.format(snr1))
        return math.log2(1 + snr1)
    
    def _i_S_D_R(self, g1, g2, snr1, pr):
        #print('0.5*log2(1+{}+{})'.format(snr1, self.source_power*pr*g1*g2/(self.sigma2*(self.source_power*g1+pr*g2)+self.sigma2**2)))
        #print('{}*{}*{}*{}/({}*({}*{}+{}*{})+{}**2)'.format( self.source_power,pr,g1,g2,self.sigma2,self.source_power,g1,pr,g2,self.sigma2))
        #print('math.log2(1+{}+{}/{})'.format(snr1, self.source_power*pr*g1,(self.sigma2*(self.source_power*g1+pr*g2)+self.sigma2**2)*g2))
        return 0.5 * math.log2(1 + snr1 + self.source_power*pr*g1/ \
                               (self.sigma2*(self.source_power*g1+pr*g2)+self.sigma2**2)*g2)
    """
    def _h_channel_conefficients(self):
        
        模拟高斯分布（0，0.1），得到变量h
        返回的是一个元组复数，a是实部，b是虚部
        
        mu = 0
        sigma2 = 0.5
        a = np.random.normal(mu, sigma2**0.5)
        b = np.random.normal(mu, sigma2**0.5)
        a = math.sqrt(self.gamma) * a
        b = math.sqrt(self.gamma) * b
        return a, b
    """
    
    """
    def _g_channel_gain(self, x):
        
        计算信道增益，量化为N个等级
        
        const_g = [1340.350486820242,7276.415305088715,21178.137065954445,47399.196112188634,100468.8998433885,218839.31862315763,1879783.760044627]
        a, b = self._h_channel_conefficients()
        #print('(a**2 + b**2)/x**(-gamma) => ({}**2 + {}**2)/{}** (-{})'.format(a, b, x, self.gamma))
        g = (a*a + b*b)/x**(-self.gamma)
        level = 0
        for each in const_g:
            if g<each:
                g = each
                break
        if g>const_g[-1]:
            g = const_g[-1]
        return g
    """
    