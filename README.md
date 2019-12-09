# DeepARTransit
DeepARTransit (Deep Auto-Regressive Transit) is a Python/Tensorflow library for de-trending transit light curves.
It implements a stacked Long Short-Term Memory network nicknamed TLCD-LSTM (standing for *Transit Light Curve Detrending LSTM*), which: 
- is trained to predict the next step mean and standard deviation of a gaussian likelihood.
- is used for interpolating the input time-series on an inner chunk - typically on the in-transit time for transit light curves.


# Usage 

### 1- Train model and provide in-transit flux predictions
 - Pointing to an experiment folder containing a parameter file
```console
$ python main_deepartrans.py -e experiment_folder_path 
```

 - Pointing to a configuration file
```console
$ python main_deepartrans.py -c configuration_file_path
```

### 2- Plot results and transit fitting

Using the notebook located in deepartransit/notebooks/post_processing.ipynb



### 3- Licensing and citing

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="Creative Commons License" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a><br />This work is licensed under a <a rel="license" href="http://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International License</a>.
 while 
 
It can be cited using the following bibtex entry:

    @misc{morvan_deepartransit_2019,
        author       = {Morvan, Mario},
        title        = {{DeepARTransit: A library for interpolating and detrending transit light curves}},
        month        = Dec,
        year         = 2019,
        doi          = {190091225},
        version      = {1.0.0},
        publisher    = {Zenodo},
        url          = {https://zenodo.org/badge/latestdoi/190091225}
        }
