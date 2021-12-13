import streamlit as st
from visco_calc import neural_net as nn
import numpy as np
import matplotlib.pyplot as plt


'''
Code created by Dominic Langhammer, at Bayerisches Geoinstitut, Bayreuth, Germany \n
Version: 13th Dec. 2021
'''

st.title('Viscosity Calculator')
readme = st.expander('About', False)
readme.write('This web application uses an artificial neural network (ANN) to calculate viscosities of silicate melts. To your left you see a tab where the appropriate oxide'
          'components can be entered in weight %. These are transformed into mol fractions and then input into the ANN. The total input composition is always normalised to 100 mol%.')
readme.write('')
readme.write('The ANN is used to calculate synthetic data points at [0.0, 0.5, 1.0, 1.5, 2.0, 9.5, 10, 10.5, 11.5] for a SiO2 content less than or equal to 60 wt% '
          'and [2, 2.5, 3, 3.5, 4, 4.5, 9.5, 10, 10.5, 11.5] for a SiO2 content larger then 60 wt%. These datapoints are then fit using the MYEGA equation by Mauro et al. (2009) '
          'with A = -2.9. The final output are the fitting parameters m and Tg.')
readme.write('')
readme.write('The application and code has been developed by Dominic Langhammer at Bayerisches Geoinstitut (BGI), Bayreuth, Germany. As of now, (6th of Dec. 2021) this has not been subject to peer review.')


path_model = './model'


left_column, right_column = st.columns(2)

with st.sidebar.form(key = 'composition'):
    si_input = st.number_input('SiO2 content in weight%', min_value = 0.0, max_value = 100.0, value = 49.3)
    ti_input = st.number_input('TiO2 content in weight%', min_value = 0.0, max_value = 100.0, value = 0.86)
    al_input = st.number_input('Al2O3 content in weight%', min_value = 0.0, max_value = 100.0, value = 16.9)
    fe_input = st.number_input('FeO content in weight%', min_value = 0.0, max_value = 100.0, value = 4.045)
    mn_input = st.number_input('MnO content in weight%', min_value = 0.0, max_value = 100.0, value = 0.16)
    mg_input = st.number_input('MgO content in weight%', min_value = 0.0, max_value = 100.0, value = 6.12)
    ca_input = st.number_input('CaO content in weight%', min_value = 0.0, max_value = 100.0, value = 12.00)
    na_input = st.number_input('Na2O content in weight%', min_value = 0.0, max_value = 100.0, value = 2.74)
    k_input = st.number_input('K2O content in weight%', min_value = 0.0, max_value = 100.0, value = 2.14)
    p_input = st.number_input('P2O5 content in weight%', min_value = 0.0, max_value = 100.0, value = 0.5)
    cr_input = st.number_input('Cr2O3 content in weight%', min_value = 0.0, max_value = 100.0, value = 0.0)
    fe2_input = st.number_input('Fe2O3 content in weight%', min_value = 0.0, max_value = 100.0, value = 4.48995)
    h_input = st.number_input('H2O content in weight%', min_value = 0.0, max_value = 100.0, value = 0.0)
    t_input = st.number_input('Temperature in K', min_value = 0.0, value = 1000.0)
    
    button = st.form_submit_button(label = 'Calculate viscosity')
 
if button:    
    comp = np.array([si_input, ti_input, al_input, fe_input, mn_input, mg_input, ca_input, na_input, k_input, p_input, cr_input, fe2_input, h_input])

    if len(comp) < 13:
        print(f'The composition file only has {len(comp)} entries. Please be sure that it only includes 13 entries of oxide components in wt%.')
        print(f'Insert them as column in the order: SiO2, TiO2, Al2O3,	FeO, MnO, MgO, CaO, Na2O, K2O, P2O5, Cr2O3, Fe2O3, H2O')
    elif len(comp) > 13:
        print(f'The composition file has {len(comp)} entries. Please be sure that it only includes 13 entries of oxide components in wt%.')
        print(f'Insert them as column in the order: SiO2, TiO2, Al2O3,	FeO, MnO, MgO, CaO, Na2O, K2O, P2O5, Cr2O3, Fe2O3, H2O')
    else:
        param, t, eta, fit, t_plot, eta_goal = nn(path_model, comp, t_input)

       

    
    fig, ax = plt.subplots()
    ax.set_ylabel('$\log \eta$ with $\eta$ in Pa s')
    ax.set_xlabel('10000/T in 1/K')
    ax.plot(10000/t_plot, fit , label = 'MYEGA')
    ax.plot(10000/t, eta, 'bo', label = 'Synthetic data')  
    ax.plot(10000/t_input, eta_goal , 'ro', label = 'Input')
    ax.legend()

    st.pyplot(fig)
    st.write('__Parameters resulting from MYEGA fit with $A$ = -2.9:__')
    st.write('$T_{g}$ = ' + str(param[0]) + ' K')
    st.write('$m$ = ' + str(param[1]))
    st.write('__Viscosity at input temperature:__')
    st.write('log $\eta$ = ' + str(eta_goal) + '  at T = ' + str(t_input) + ' K')

