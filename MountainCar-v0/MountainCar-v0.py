import gym
import numpy as np
import random
import datetime
from pathlib import Path
import pandas as pd
# todo : CartPole-v0

"""
! ghi chú về bài toán

    *Description:
        The agent (a car) is started at the bottom of a valley. For any given
        state the agent may choose to accelerate to the left, right or cease
        any acceleration.
    *Source:
        The environment appeared first in Andrew Moore's PhD Thesis (1990).
    *Observation:
        Type: Box(2)
        Num    Observation               Min            Max
        0      Car Position              -1.2           0.6
        1      Car Velocity              -0.07          0.07
    *Actions:
        Type: Discrete(3)
        Num    Action
        0      Accelerate to the Left
        1      Don't accelerate
        2      Accelerate to the Right
        Note: This does not affect the amount of velocity affected by the
        gravitational pull acting on the car.
    *Reward:
         Reward of 0 is awarded if the agent reached the flag (position = 0.5)
         on top of the mountain.
         Reward of -1 is awarded if the position of the agent is less than 0.5.
    *Starting State:
         The position of the car is assigned a uniform random value in
         [-0.6 , -0.4].
         The starting velocity of the car is always assigned to 0.
    *Episode Termination:
         The car position is more than 0.5
         Episode length is greater than 200


"""

def save_weight(dirfile,weight):
    return np.save(dirfile,weight)

def load_weight(dirfile):
    return np.load(dirfile)

def create_Qtable(state_space, action_space, bins_size = 30):
    
    bins = np.array([
        np.linspace(-1.2,0.6, num=bins_size), # Car Position : -1.2 -> 0.6
        np.linspace(-0.07,0.07, num=bins_size), # Car Velocity : -0.07 -> 0.07
        
       ])
    #print(bins.shape)

    q_table = np.random.uniform(low=-1,high=1,size=([bins_size] * state_space + [action_space]))
    # size = [30,30,3]
    
    return q_table, bins

def discrete(state, bins): # find index of state in linespace
    #? index = discrete([-1.2,-0.07],bins=bins) # how to use

    index = []
    for i in range(len(state)): index.append(np.digitize(state[i],bins[i]) - 1)
    return index

def get_next_action(q_table,state_current,bins,epsilon):
    
    index = discrete(state_current,bins)
    #print('index in table ',index,'value',q_table[index[0]][index[1]][index[2]][index[3]])
    
    if(np.random.random() <= epsilon): # chose from table
        action = np.argmax( q_table[index[0]][index[1]] )
        q_value = q_table[index[0]][index[1]][action]
        return action, q_value, index
    else: # chose random
        action = random.randint(0,1)
        q_value = q_table[index[0]][index[1]][action]
        return action, q_value, index

def run_game_to_create_weight(env,k_game,episode, q_table,bins, epsilon,discount_factor,learning_rate):
    
    max_pos_L = 2        
    max_pos_R = -2

    max_vec_L = 1
    max_vec_R = -1

    for k in range(k_game):
        #? observation = state_current
        observation = env.reset()
        sum_reward = 0
        #print("k_game = ", k, end=' -> ')
        
        

        for e in range(episode):
            #env.render() # show image
            
            # chose action
            action_should_do, q_value, index = get_next_action(q_table,observation,bins,epsilon)
            #print('action : ',action_should_do,"~~",str(action[action_should_do]))
            
            # perform action
            observation, reward, done, info = env.step(action_should_do)
            if(reward == -1):
                r = 0
                if( observation[0]>max_pos_R):
                    max_pos_R = observation[0]
                    r += 1
                if( observation[0]<max_pos_L):
                    max_pos_L = observation[0]
                    r += 2
                
                if( observation[1]>max_vec_R):
                    max_vec_R = observation[1]
                    r += 1
                if( observation[1]<max_vec_L):
                    max_vec_L = observation[1]
                    r += 2

                if(r > 0):
    	            reward = r

            elif(reward == 0):
                reward = 10
                
                
            # measure reward 
            sum_reward += reward
            
            index_new = discrete(observation,bins)
            temp = reward + (discount_factor * np.max(q_table[index_new[0]][index_new[1]])) - q_value 
            new_q_value = q_value + (learning_rate * temp)
            
            # update Q_table
            q_table[index[0]][index[1]][action_should_do] = new_q_value

            if done:
                print("Game ",k," length reward = ",sum_reward)
                break

        print(max_pos_L,max_pos_R,max_vec_L,max_vec_R)
        


    env.close()

    return q_table

def run_game(env,k_game,episode, q_table,bins):
    arr_reward = []
    for k in range(k_game):
        #? observation = state_current
        observation = env.reset()
        sum_reward = 0
        #print("k_game = ", k, end=' -> ')

        for e in range(episode):
            env.render()
            # get action
            action_should_do, q_value, index = get_next_action(q_table,observation,bins,epsilon=1)
            
            # perform action
            observation, reward, done, info = env.step(action_should_do)
            
            # measure reward
            sum_reward += reward

            if done:
                #print("Episode length = ", sum_reward)
                arr_reward.append(sum_reward)
                break

    env.close()
    return arr_reward

def calculate_weight_and_save(env,q_table,bins,epsilon,discount_factor,learning_rate, dirfile):
    # TODO : calculate weight and save
    k_game = 250
    print('\nrun n game ')
    q_table = run_game_to_create_weight(env=env,
                                        k_game=k_game,
                                        episode=200,
                                        q_table=q_table,
                                        bins=bins,
                                        epsilon=epsilon,
                                        discount_factor=discount_factor,
                                        learning_rate=learning_rate)
    print("Done !\n")
    
    save_weight(dirfile,q_table)
    print('save weight to ',dirfile)

def load_weight_and_run(env,q_table,bins, dirfile):
    q_table = load_weight(dirfile)
    print('q_table ',q_table.shape)

    # k_game>100 => if 100 game have average episode ~~ 195  
    k_game = 3
    arr_reward = run_game(env=env,
                            k_game=k_game,
                            episode=200,
                            q_table=q_table,
                            bins=bins)
    arr_reward = np.array(arr_reward)
    print('Average ',k_game,'k_game = ',np.average( arr_reward ) )
def test(env):
    env.reset()
    for _ in range(200):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        print(observation,reward)
        if(done):
            break
    env.close()

def main_v001():
    # *dir file weight 
    dirfile = 'weight\MountainCar-v0_Qtable_v001.npy'

    # *rate
    epsilon = 0.8 #0.92 #0.80
    discount_factor = 0.84 #0.92 #0.84
    learning_rate = 0.84 #0.92 #0.84

    # *create env
    env = gym.make('MountainCar-v0')

    # * action, reward, q_table(s,a)
    action = ['left', 'right']
    reward = 1

    state_space = 2
    action_space = 3
    bins_size = 30 # state can infinity, so we collap them to bins

    q_table,bins = create_Qtable(state_space,action_space,bins_size)
    print("q_table",q_table.shape,",bins",bins.shape)

    
    # TODO : Test
    #test(env)
    
    
    # TODO : calculate weight and save
    calculate_weight_and_save(env,q_table,bins,epsilon,discount_factor,learning_rate, dirfile)

    
    # TODO : load weight and run
    load_weight_and_run(env,q_table,bins, dirfile)

    

    #arr_reward_Solved = np.delete(arr_reward,np.where(arr_reward<195))
    #print('Average ',arr_reward_Solved.shape[0],'k_game_Solved = ',np.average( arr_reward_Solved ) )
    
    
if(__name__ == '__main__'):
    

    main_v001()


    