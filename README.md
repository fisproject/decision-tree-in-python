decision-tree-in-python
====

## Overview
Example of Decision Tree Classifier and Regressor in Python.

## Usage

* Classification tree

```bash
$ python classifier-example.py
 if X[2] <= 2.45
    then {value: 0, samples: 50}
    else if X[3] <= 1.75
        then if X[2] <= 4.95
            then {value: 1, samples: 48}
            else {value: 2, samples: 6}
        else if X[2] <= 4.85
            then {value: 2, samples: 3}
            else {value: 2, samples: 43}
```

* Regression tree

```bash
$ python regressor-example.py
 if X[0] <= 3.13275045531
    then if X[0] <= 0.513901088514
        then {value: 0.0523606779563, samples: 11}
        else {value: 0.713825681714, samples: 40}
    else if X[0] <= 3.85022857897
        then {value: -0.451902639773, samples: 14}
        else {value: -0.868642556986, samples: 15}
```

## Licence
[MIT](http://opensource.org/licenses/MIT)

## Author
[t2sy](https://github.com/fisproject)
