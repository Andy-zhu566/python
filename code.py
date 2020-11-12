import simpy
import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt


def WN(sd):
    noise = np.random.normal(0, sd, 1)
    return noise[0]

class Signal:
    def __init__(self, value, velocity):
        self.value = value
        self.velocity = velocity
        self.value_trace = np.array([value])
    
    
    def move(self):
        self.value += self.velocity
        self.value_trace =  np.append(self.value_trace,self.value)


class SignalSynchronize(gym.Env):
    """
    Obervasion
    0 - reset
    1 - S1 is larger than S2 (over 1 sd of the noise)
    2 - S1 is euqal to S2 (within 1 sd of the noise)
    3 - S1 is lower than S2 (over 1 sd of the noise)
    
    rewards
    
    0 if |S1-S2|>= 1 sd
    1 if |S1-S2|< 1 sd
    
    action space (1 X 2)
    action[0] - value adjust for S1
    action[1] - velocity adjust for S1
    """
    def __init__(self, s1, s2):
        self.s1=s1
        self.s2=s2
        self.observation_space = spaces.Discrete(4)
        self.observation = 0
        self.reset()
        self.delta_s = s1.value-s2.value
    
    def reset(self):
        self.observation = 0
        return self.observation
    
    
    def step(self, action):
        sd = 1

        self.s1.value += action[0]+ WN(sd)
        self.s1.velocity += action[1]+ WN(sd)
        
        self.s1.move()
        self.s2.move()
        
        
        self.delta_s = self.s1.value - self.s2.value + WN(sd)
        
        
        if self.delta_s>=sd:
            self.observation = 1
        
        if -sd< self.delta_s < sd:
            self.observation = 2
        
        if self.delta_s<= -sd:
            self.observation = 3
            
        
        reward = 0
        done = False
        
        
        if self.observation == 2:
            reward = 1
            done = True
            
        
        return self.observation, reward, done, self.delta_s
        
        

class Signal_proc:
    def __init__(self,env,s1,s2,time_interval):
        self.s1 = s1
        self.s2 = s2
        self.time_interval = time_interval
        self.env = env
        self.action = env.process(self.run())
        self.signal_synchronizer = SignalSynchronize(s1,s2)
        self.signal_synchronizer.reset()

    
    def run(self):
        action = np.array([0,0])
        while True:
            yield self.env.timeout(self.time_interval)
            observation, reward, done , delta_s= self.signal_synchronizer.step(action)
            
#            RL policy
            if observation == 2:
                action = np.array([0,0])
            
            if observation == 1:
                value_adjust = -delta_s
                velocity_adjust = - delta_s
                action=np.array([value_adjust,velocity_adjust])
            
            if observation == 3:
                value_adjust = -delta_s
                velocity_adjust =  -delta_s
                action=np.array([value_adjust,velocity_adjust])
            



time_interval = 5
Max_timeout = 100
Max_interation = Max_timeout / time_interval



env = simpy.Environment()

rnd = random.sample(range(-50, 50), 2)

s_1 = Signal( -10, rnd[0])
s_2 = Signal(2, rnd[1])

signal_process = Signal_proc(env,s_1,s_2,time_interval)


env.run(until = Max_timeout)



t = np.arange(0,Max_interation,1)
plt.plot(s_1.value_trace,'r-',label="S_1")
plt.plot(s_2.value_trace,'b--',label="S_2")
plt.legend(loc="upper left")
plt.ylabel('Signal value')
plt.show()

