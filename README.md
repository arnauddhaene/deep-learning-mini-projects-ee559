# Mini Projects for Deep Learning EE-559 (EPFL), spring semester 2021

## Structure

This repository is split into the two mini-projects described in the [Mini-projects guidelines](https://fleuret.org/dlc/materials/dlc-miniprojects.pdf).

```
├── .github/workflows     <- GitHub Actions
├── Proj1                 <- Project 1 – Classification, weight sharing, auxiliary losses
|   ├── data/mnist/MNIST  <- MNIST dataset
|   ├── notebooks
|   ├── src
|   |   ├── models
|   |   |   ├── __init__.py
|   |   |   ├── convnet.py
|   |   |   └── mlp.py
|   |   ├── __init__.py
|   |   ├── dlc_practical_prologue.py
|   |   ├── metrics.py
|   |   ├── test.py
|   |   ├── train.py
|   |   └── utils.py
|   └── README.md
├── Proj2                 <- Project 2 - Mini deep-learning framework
├── README.md             <- The top-level README for developers using this project.
├── requirements.txt      <- Necessary Python libraries to run the projects
├── tox.ini               <- Configuration file for [flake8]
```

The version of Python used in this project is Python 3.8

## Running the projects

More information can be found in the projects' respective `README.md` files.

## GitHub Actions

The project follows flake8 practices, with the exception of:

* [E501 - Line too long] We use a maximum line length of 100
* [W293 - Blank line contains whitespace]

Furthermore, we don't lint Prof. Fleuret's [prologue file](Proj1/src/dlc_practical_prologue.py) as he consistently violates rule E251.

The Action is triggered on every push, which means you should make sure to run `flake8` in the root directory before committing any changes.

If you are using VSCode as an IDE, it is highly recommended to install the [cornflakes-linter](https://marketplace.visualstudio.com/items?itemName=kevinglasson.cornflakes-linter) extension.

## Authors

| SURNAME Name         | SCIPER |
| -------------------- | ------ |
| DHAENE Arnaud        | 269883 |
| COUPET Léopold       | |
| PISAREWSKI Alexander | |
