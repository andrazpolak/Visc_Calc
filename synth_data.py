import numpy as np
import tensorflow as tf
from tensorflow import keras


def visc_calc(inp, path_model, si):
    '''This function calculates the synthetic data points from a feed forward artifical neural network. A bisection algorithm is used to find specific viscosities in
    the high and low viscosity regime. These are chosen depending on the SiO2 content in wt% of the respective sample.''' 
    
    
    ### Loading the neural network and choosing viscosity values that are calculated using the bisection algorithm, dependent on SiO2 content. ### 
    
    model = tf.keras.models.load_model(path_model)
    if si <= 60:
        eta_goal = [0.0, 0.5, 1.0, 1.5, 2.0, 9.5, 10, 10.5, 11.0, 11.5]
    else:
        eta_goal = [2, 2.5, 3, 3.5, 4, 4.5, 9.5, 10, 10.5, 11.0, 11.5]
    
    ### Normalisation factor for input temperatures. ###
    t_max = 2023.0

    
    t_goal = []
    eta = []
    
    ### Temperature boundaries for the biscetion. ###
    ### This interval has been chosen rather large to ensure that the goal viscosity values are within thises boundaries. ###
    t_top_start = (3000.0/t_max- 0.602847523)/np.sqrt(0.031535353)
    t_bot_start = (300.0/t_max- 0.602847523)/np.sqrt(0.031535353)
    t_mid_start = (t_top_start + t_bot_start)/2
    
    ### Defining the input array without Cr. ###    
    comp  = np.c_[t_mid_start, inp[0], inp[1], inp[2], inp[3], inp[4], inp[5], inp[6], inp[7], inp[8], inp[9], inp[11], inp[12], inp[13], inp[14]]
    
    eta_mid_start = model(comp)
    
    ### Start of bisection loop. ###
    for goal in eta_goal:
        
        ### Resetting boundary values and central viscosity and calculating error.###
        t_top = t_top_start
        t_bot = t_bot_start
        t_mid = t_mid_start
        eta_mid = eta_mid_start
        
        
        err = eta_mid-goal
        
        ### Depending on position of the middle viscosity value, the temperatuer boundaries are moved to reduce the interval size. This is repeated until the ###
        ### mid value viscosity is within 10^-3 of the goal viscosity. ###
        while np.absolute(err) > 10**(-3):
                
            if eta_mid < goal:
                t_top = t_mid
            elif eta_mid > goal:
                t_bot = t_mid

            
            t_mid = (t_bot+t_top)/2

            
            
            comp[0][0] = t_mid
            eta_mid = model(comp)    
            eta_mid = np.concatenate(eta_mid)   
            

    
            err = eta_mid-goal

        ### Adding calculated values to a temperatuer and viscosity array. ###
        t_goal.append((t_mid*np.sqrt(0.031535353) + 0.602847523)*t_max)
        eta.append(eta_mid)

    return t_goal, eta
    