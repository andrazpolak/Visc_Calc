# Visc_Calc
This project aims at creating synthetic viscosity models for natural silicate melts using a trained artifical neural network (ANN).

The ANN is used to calculate synthetic data points at [0.0, 0.5, 1.0, 1.5, 2.0, 9.5, 10, 10.5, 11.5] for a SiO2 content less than or equal to 60 wt% 
and [2, 2.5, 3, 3.5, 4, 4.5, 9.5, 10, 10.5, 11.5] for a SiO2 content larger than 60%. These datapoints are then fit using the MYEGA equation by Mauro et al. (2009)
with A = -2.9. The final output are the fitting parameters m and Tg.
