import gym
import numpy as np
import random
import datetime
from pathlib import Path
import pandas as pd
# todo : CartPole-v0

"""
! ghi chú về bài toán

    Description:
        A pole is attached by an un-actuated joint to a cart, which moves along
        a frictionless track. The pendulum starts upright, and the goal is to
        prevent it from falling over by increasing and reducing the cart's
        velocity.
    Source:
        This environment corresponds to the version of the cart-pole problem
        described by Barto, Sutton, and Anderson
    *Observation:
        Type: Box(4)
        Num     Observation               Min                     Max
        0       Cart Position             -4.8                    4.8
        1       Cart Velocity             -Inf                    Inf
        2       Pole Angle                -0.418 rad (-24 deg)    0.418 rad (24 deg)
        3       Pole Angular Velocity     -Inf                    Inf
    *Actions:
        Type: Discrete(2)
        Num   Action
        0     Push cart to the left
        1     Push cart to the right
        Note: The amount the velocity that is reduced or increased is not
        fixed; it depends on the angle the pole is pointing. This is because
        the center of gravity of the pole increases the amount of energy needed
        to move the cart underneath it
    *Reward:
        Reward is 1 for every step taken, including the termination step
    *Starting State:
        All observations are assigned a uniform random value in [-0.05..0.05]
    *Episode Termination:
        Pole Angle is more than 12 degrees.
        Cart Position is more than 2.4 (center of the cart reaches the edge of
        the display).
        Episode length is greater than 200.
        Solved Requirements:
        Considered solved when the average return is greater than or equal to
        195.0 over 100 consecutive trials.


"""

def save_weight(dirfile,weight):
    return np.save(dirfile,weight)

def load_weight(dirfile):
    return np.load(dirfile)

def create_Qtable(state_space, action_space, bins_size = 30):
    
    bins = np.array([
        np.linspace(-4.8,4.8, num=bins_size), # Cart Position : -4.8 -> 4.8
        np.linspace(-10,10, num=bins_size), # Cart Velocity : -Inf -> Inf
        np.linspace(-0.418,0.418, num=bins_size), # Pole Angle :-0.418 rad (-24 deg) -> 0.418 rad (24 deg)
        np.linspace(-10,10, num=bins_size), # Pole Angular Velocity : -Inf -> Inf
    
    ])
    #print(bins.shape)

    q_table = np.random.uniform(low=-1,high=1,size=([bins_size] * state_space + [action_space]))
    # size = [30,30,30,30,2]
    
    return q_table, bins

def discrete(state, bins): # find index of state in linespace
    #? index = discrete([4.8,10,0.418,10],bins=bins) # how to use

    index = []
    for i in range(len(state)): index.append(np.digitize(state[i],bins[i]) - 1)
    return index

def get_next_action(q_table,state_current,bins,epsilon):
    
    index = discrete(state_current,bins)
    #print('index in table ',index,'value',q_table[index[0]][index[1]][index[2]][index[3]])
    
    if(np.random.random() <= epsilon): # chose from table
        action = np.argmax( q_table[index[0]][index[1]][index[2]][index[3]] )
        q_value = q_table[index[0]][index[1]][index[2]][index[3]][action]
        return action, q_value, index
    else: # chose random
        action = random.randint(0,1)
        q_value = q_table[index[0]][index[1]][index[2]][index[3]][action]
        return action, q_value, index

def run_game_to_create_weight(env,k_game,episode, q_table,bins, epsilon,discount_factor,learning_rate):
    
    

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
            
            # measure reward 
            sum_reward += 1
            
            index_new = discrete(observation,bins)
            temp = reward + (discount_factor * np.max(q_table[index_new[0]][index_new[1]][index_new[2]][index_new[3]])) - q_value 
            new_q_value = q_value + (learning_rate * temp)
            
            # update Q_table
            q_table[index[0]][index[1]][index[2]][index[3]][action_should_do] = new_q_value

            if done:
                #print("Episode length = ",sum_reward)
                break


        


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

    
    return arr_reward



def main_v001():
    # *dir file weight 
    dirfile = 'weight\CartPole-v0_Qtable_v001.npy'

    # *rate
    epsilon = 0.9
    discount_factor = 0.9
    learning_rate = 0.9

    # *create env
    env = gym.make('CartPole-v0')

    # * action, reward, q_table(s,a)
    action = ['left', 'right']
    reward = 1

    state_space = 4
    action_space = 2
    bins_size = 30 # state can infinity, so we collap them to bins

    q_table,bins = create_Qtable(state_space,action_space,bins_size)
    print("q_table",q_table.shape,",bins",bins.shape)

    
    '''
    # TODO : calculate weight and save
    k_game = 300
    print('\nrun n game ')
    q_table = run_game_to_create_weight(env=env,
                                        k_game=1000,
                                        episode=200,
                                        q_table=q_table,
                                        bins=bins,
                                        epsilon=epsilon,
                                        discount_factor=discount_factor,
                                        learning_rate=learning_rate)
    print("Done !\n")
    
    save_weight(dirfile,q_table)
    print('save weight to ',dirfile)
    '''

    # TODO : load weight and run
    q_table = load_weight(dirfile)
    print('q_table ',q_table.shape)

    # k_game>100 => if 100 game have average episode ~~ 195  
    k_game = 10
    arr_reward = run_game(env=env,
                            k_game=k_game,
                            episode=200,
                            q_table=q_table,
                            bins=bins)
    arr_reward = np.array(arr_reward)
    print('Average ',k_game,'k_game = ',np.average( arr_reward ) )

    arr_reward_Solved = np.delete(arr_reward,np.where(arr_reward<195))
    print('Average ',arr_reward_Solved.shape[0],'k_game_Solved = ',np.average( arr_reward_Solved ) )
    
    #*Average  300 k_game =  144.57666666666665
    #*Average  78 k_game_Solved =  199.66666666666666
    #*Average  2000 k_game =  140.656
    #*Average  501 k_game_Solved =  199.71656686626747

if(__name__ == '__main__'):
    

    main_v001()


    