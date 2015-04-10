=======================================================
Project has been developed in Python for 32 bit windows
=======================================================

The project contains 2 Python files :

1. logistic_regression_mfcc_fft.py -- Implementation of Logistic Regression for FFT and MFCC features
2. logistic_regression_fft20.py  -- Implemented rank method to select top 20 features for the classification

Instructions to run logistic_regression_mfcc_fft.py
===================================================

It is time consuming to calculate FFT and MFCC's everytime for the given WAV files. Hence I have calculated them once and saved
them in file [fft_matrix.out, mfcc_matrix], so that the program can use those files as input and provide us with faster results.

If you want to calculate FFt and MFCC directly then please arrange the music folders like this in the program directory:

classical
country
jazz
metal
pop
rock

To run logistic_regression_mfcc_fft.py from the command line, type:

1. command to run for MFCC:

python logistic_regression_mfcc_fft.py mfcc

2. command to run for FFT:

python logistic_regression_mfcc_fft.py fft

Instructions to run logistic_regression_fft20.py
=================================================

python logistic_regression_fft20.py

============
OutPut Log
============

Please find attached output logs for all 3 Parts :

1. fft_run_log.txt
2. mfcc_run_log.txt
3. fft_20_run_log.txt

