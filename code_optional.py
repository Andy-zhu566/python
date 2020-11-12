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
        self.childlist = []
        self.parent = None
        self.signal_synchronize = None
        self.action = np.array([0,0])

    def move(self):
        self.value += self.velocity
        self.value_trace =  np.append(self.value_trace,self.value)
        
    
    def AttachChild(self,child):
        child.parent = self
        self.childlist.append(child)
        

class SignalSynchronize(gym.Env):
    
    """
    num - number of signals
    
    Obervasion (num X 1)
    0 - reset & root node
    1 - child signal is larger than parent signal (over 1 sd of the noise)
    2 - child signal is euqal to parent signal (within 1 sd of the noise)
    3 - child signal is lower than parent signal (over 1 sd of the noise)
    
    rewards
    
    1 is all signals are synchronized
    0 otherwise
    
    action space (num x 2)
    
    action[i][0] - value adjust for signal i-1
    action[i][1] - velocity adjust for signal i-1
    
    
    
    delta_list - difference between each signal and its parent 
               - 0 for root signal
    
    """
    
    def __init__(self, signal_list):
        self.signal_list = signal_list
        self.num = len(self.signal_list)
        self.observation_space = spaces.Discrete(4)
        self.observation = np.zeros(self.num)
        self.reset()
        self.delta_list = np.zeros(self.num)
        for i in range(self.num):
            if self.signal_list[i].parent != None:
                self.delta_list[i] = self.signal_list[i].value - self.signal_list[i].parent.value
            else:
                self.delta_list[i] = 0
    
    def reset(self):
        self.observation = np.zeros(self.num)
        return self.observation
    
    
    def step(self, action):
        sd = 0.1
        
        # take action
        for i in range(self.num):
            if self.signal_list[i].parent != None:
                self.signal_list[i].value += action[i][0] + WN(sd)
                self.signal_list[i].velocity += action[i][1] + WN(sd)
            self.signal_list[i].move()

        # observation
        for i in range(self.num):
            if self.signal_list[i].parent != None:
                self.delta_list[i] = self.signal_list[i].value - self.signal_list[i].parent.value + WN(sd)
            else:
                self.delta_list[i] = 0
                            
        
        
            if self.delta_list[i]>=sd:
                self.observation[i] = 1
            
            if -sd< self.delta_list[i] < sd:
                self.observation[i] = 2
            
            if self.delta_list[i]<= -sd:
                self.observation[i] = 3
            
        
        reward = 0
        done = True
            
        if np.array_equal(self.observation,2*np.ones(4)):
            reward = 1
            done = True
            
        
        return self.observation, reward, done, self.delta_list

class Signal_proc:
    def __init__(self,env,signal_list,time_interval):
        self.signal_list = signal_list
        self.num = len(signal_list)
        self.time_interval = time_interval
        self.env = env
        self.action = env.process(self.run())
        self.signal_synchronizer = SignalSynchronize(self.signal_list)


    
    def run(self):
        action  = np.zeros((self.num,2))
        while True:
            yield self.env.timeout(self.time_interval)
            observation, reward, done , delta_list= self.signal_synchronizer.step(action)
            # RL Policy
            for i in range(self.num):
                if observation[i] == 2:
                    action[i][0] = 0
                    action[i][1] = 0
                
                if observation[i] == 1:
                    action[i][0] = - delta_list[i]
                    action[i][1] = - delta_list[i]
                
                if observation[i] == 3:
                    action[i][0] = - delta_list[i]
                    action[i][1] = - delta_list[i]






time_interval = 5
Max_timeout = 40
Max_interation = Max_timeout / time_interval


env = simpy.Environment()

"""
Tree structure for the example

S1______S2_____S4
    |
    |___S3_____S5
          |
          |____S6

"""

rnd = random.sample(range(-50, 50), 12)
s1 = Signal(rnd[0],rnd[1] )
s2 = Signal(rnd[2],rnd[3] )
s3 = Signal(rnd[4],rnd[5] )
s4 = Signal(rnd[6],rnd[7] )
s5 = Signal(rnd[8],rnd[9] )
s6 = Signal(rnd[10],rnd[11] )

SignalList = [s1, s2, s3, s4, s5, s6]

s1.AttachChild(s2)
s1.AttachChild(s3)
s2.AttachChild(s4)
s3.AttachChild(s5)
s3.AttachChild(s6)

signal_process = Signal_proc(env,SignalList,time_interval)


env.run(until = Max_timeout)


t = np.arange(0,Max_interation,1)

plt.plot(t,s1.value_trace,'r-',label="S1")
plt.plot(t,s2.value_trace,'b--',label="S2")
plt.plot(t,s3.value_trace,'m-.',label="S3")
plt.plot(t,s4.value_trace,'g:',label="S4")
plt.plot(t,s5.value_trace,'y-.',label="S3")
plt.plot(t,s6.value_trace,'c--',label="S4")

plt.legend(loc="upper left")
plt.ylabel('Signal value')
plt.show()
