# Rosetta version 3-alpha (3a) 

Pedotransfer functions by Schaap et al., 2001 and Zhang and Schaap, 2016.

Rosetta predicts van Genuchten soil water retention curve parameters utilizing a
weighted recalibration of the Rosetta pedotransfer model with improved estimates of hydraulic parameter 
distributions and summary statistics (Rosetta3). Journal of Hydrology.

This version of Rosetta has been refactored by Roger Lew (2021) for use as a Python3 package under GNU GPL V2.

For more information regrading Rosetta see https://cals.arizona.edu/research/rosetta/

## Installation

Easiest method of installation is to clone to your dist-packages folder.


e.g. homebrew Python install on MacOS
```bash
% cd /opt/homebrew/lib/python3.9/site-packages/rosetta/
% git clone https://github.com/rogerlew/rosetta
```

## Rosetta can be used from Python
```python
> from rosetta import Rosetta
> import numpy as np
> ros = Rosetta(model_no=3, debug=False)
> data = np.array([[35.0, 50.0, 15.0, 0.2]]) # sand, silt, clay, bulk density
> ros.predict(data)
{'theta_r': array([0.14238712]), 
 'theta_s': array([0.7368426]), 
 'alpha': array([0.00368141]), 
 'npar': array([1.44084392]), 
 'ks': array([1047.97575913])}
```

## Twarakavi et al., (2009) Wilting Point and Field Capacity Estimates
```python
>>> ros.predict(data, calc_wilting_point=True, calc_field_capacity=True)
{'theta_r': array([0.14238712]), 
 'theta_s': array([0.7368426]), 
 'alpha': array([0.00368141]), 
 'npar': array([1.44084392]), 
 'ks': array([1047.97575913]), 
 'wp': array([0.2437112]), 
 'fc': array([0.34023513])}
```

## CLI interface by calling rosetta as a module
```
%  python3 -m rosetta -h
usage: __main__.py [-h] [--raw] [--calc_wilting_point] [--calc_field_capacity] [-i INPUT] [-o OUTPUT] model_n

Rosetta 3 pedotransfer function CLI.

positional arguments:
  model_n

optional arguments:
  -h, --help            show this help message and exit
  --raw                 get raw data
  --calc_wilting_point  calculate wilting point
  --calc_field_capacity
                        calculate field capacity
  -i INPUT, --input INPUT
                        input from file
  -o OUTPUT, --output OUTPUT
                        store predicted data

% python3 -m rosetta 3 -i test_input_H3w.txt -o test_output_H3w.txt
```
