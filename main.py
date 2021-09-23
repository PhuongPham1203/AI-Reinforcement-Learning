import gym
import numpy as np 
import pandas as pd 
import random
# enumerate()

q_table = np.zeros( (6,6,4) )
#print(q_table)
actions = ['up', 'right', 'down', 'left']

for index, value in enumerate(actions):
    print(index,value)

print(actions.index('right'))


