import numpy as np
import tensorflow as tf
from tensorflow import keras


def visc_calc(mol_frac, path_model, si):
    '''This function calculates the synthetic data point from a feed forward artifical neural network. A bisection algorithm is used to find specific viscosities in
    the high and low viscosity regime. These are chosen depending on the SiO2 content in wt% of the respective sample.''' 
    
    
    ### Loading the neural network and choosing viscosity values that are searched for using the bisection algorithm dependent on SiO2 content. ### 
    
    model = tf.keras.models.load_model(path_model)
    if si <= 60:
        eta_goal = [0.0, 0.5, 1.0, 1.5, 2.0, 9.5, 10, 10.5, 11.5]
    else:
        eta_goal = [2, 2.5, 3, 3.5, 4, 4.5, 9.5, 10, 10.5, 11.5]
    
    ### Normalisation factor for input temperatures. ###
    t_max = 2023.0

    
    
    ### Calculation of the structure modifier (SM) and K/(Na + K) parameters, which are used as input. ###
    sm = mol_frac[3] + mol_frac[4] + mol_frac[5] + mol_frac[6] + mol_frac[7] + mol_frac[8]

    if mol_frac[8] + mol_frac[7] == 0:
        nak = 0
    else:
        nak = mol_frac[8]/(mol_frac[8] + mol_frac[7])
        
    ### Defining the input array without Cr. ###    
    comp  = np.c_[1, mol_frac[0], mol_frac[1], mol_frac[2], mol_frac[3], mol_frac[4], mol_frac[5], mol_frac[6], mol_frac[7], mol_frac[8], mol_frac[9], mol_frac[11], mol_frac[12], sm, nak]
        
    
    
    t_goal = []
    eta = []
    
    ### Temperature boundaries for the biscetion. ###
    ### This interval has been chosen rather large to ensure that the goal viscosity values are within thises boundaries. ###
    t_top_start = 3000/t_max
    t_bot_start = 300.0/t_max
    
    ### Start of bisection loop. ###
    for goal in eta_goal:
        
        ### Resetting boundary values after calculating a viscosity. ###
        t_top = t_top_start
        t_bot = t_bot_start
        t_mid = (t_top + t_bot)/2
        
        ### Assigning the central temperature and calculation of viscosity value. ### 
        
        comp[0][0] = t_mid
        eta_mid = model(comp)
        
        
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
        t_goal.append(t_mid*t_max)
        eta.append(eta_mid)

    return t_goal, eta
    