# DSP_labs
Task solutions for Digital Signal Processing laboratory (run on STM32F404 and Analog Discovery 2). Results (generated signals/files etc.) are stored in .pdf file in respective subfolder. <br /> <br />
Problems revolved around:
+ typical DSP problems on embedded systems (ADC, DAC, modulations etc.)
+ DSP in Python 
+ OpenCV
+ ML using TensorFlow  


## Lab 1 - Intro
Basic introduction - simple LED blinking (omitted).

## Lab 2 - ADC
Read ADC input from generator with sampling rate 50 kHz and oversampling x16.

## Lab 3 - DAC
+ use ADC input to drive DAC output
+ output signal using LUT with sine values
+ tweak frequency of LUT sine with DDS algorithm
+ attach RC filter

## Lab 4 - Modulations
Generate different AM and FM waves using DAC from precious exercise and make observations using FFT.

## Lab 5 - Filters
Implement 3rd order low pass IIR filter and 2nd order band pass IIR filter.

## Lab 6 - Signal Processing in Python
Filter out the noise from heartbeat signal (both manually and using library) and represent new data.

## Lab 7 - OpenCV
+ implement face tracking script 
+ implement plate recognition script

## Lab 8 - Perceptron
Build Neural Network implementing linear regression. Plot the results for different hyperparameters (learning rate) and optimizers (Adam, Stochastic Gradient Descent).
