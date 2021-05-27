# Project 1 - Classification, weight sharing, auxiliary losses

###### EE-559 Deep Learning (EPFL), spring 2021

A more extended version of Project 1 is available in the sister folder `Proj1-extended`. It uses libraries that are not allowed for official submission, but implements interesting aspects like automatic figure generation, modern argparsing for running tests, and hyper-parameter grid-search.

## Running `Proj1`

As enforced in the project description, you can train and test our best performing model using

```bash
python test.py
```

The MNIST data will be downloaded to `Proj1/data`. You should run the test script from inside the `Proj1` directory.

## Structure

### test

The `test.py` script run is able to run training trials on all evaluated models. Originally, this script used the `click` library with argument parsing which allowed for much flexibility in performing experiments. Unfortunately, this library can not run on the VM and thus the script has been modified to run on hard-coded instructions only.

### train

This module implements the `train` method, used for training a given model.

### utils

This module implements training and testing set standardisation and is also able to pull a dataset using predefined functions in [`dlc_practical_prologue.py`](dlc_practical_prologue.py).

### metrics

This module implements the custom `TrainingMetrics` and `TestingMetrics` classes used to track model performance. A much more sophisticated version can be found in [the extended Proj1 directory]('../Proj1-extended/src/metrics.py). Unfortunately, the full version uses some plotting libraries that are not accepted for the official submission of this mini-project.

### models

The models package contains all the models evaluated during the course of this project.


## Authors

* COUPET, Léopold
* DHAENE, Arnaud
* PISAREWSKI, Alexander
