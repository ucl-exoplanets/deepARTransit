# DeepARTransit
DeepARTransit is a library for de-trending transit light curves.
It contains a Tensorflow implementation of a stacked Long Short-Term Memory network architecture: 
- trained to predict the next step mean and standard deviation of a gaussian likelihood.
- used for interpolating of the input time-series on an inner chunk

Although designed for predicting the flux of transit light curves during transit time, this could more generally be used for interpolating any kind of time-series with possible covariate data. 

# Usage 

## References


