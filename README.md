# 2D red blood cell aggregation using deplition force approach

This repository introduces the simulation of red blood cell (RBC) aggregation using depletion force approach. 
This simulation is mainly for __demonstration__ purposes, not to physically study the phenomenon. 
Further modification of the code as well as its verification is required in order to use it for scientific purposes. 

__Note__: In the simulation, the RBC is represented as a solid rectangle.

## Usage

See __Example.ipynb__ file.

## Example

![Example](https://github.com/Petr-Ermolinskiy/2D_RBC_aggregation_simulation/tree/main/test.mp4)

## Requirements
This code was tested only with `Python 3.10.13`.

Necessary libraries for the simulation:
```python
pip install "numpy==1.26.4"
pip install "matplotlib==3.8.3"
pip install "seaborn==0.13.2"
pip install "pandas==2.2.1"
```
Also, make sure that the version of jupyter notebook < 7.0. The latest version of jupyter notebook may cause some problems.
```python
pip install notebook==6.5.6
```

## Citation

If it usufull you can cite this repository:
```
@software{Ermolinskiy2024,
	author = {Ermolinskiy, P.},
	title = {GitHub repository},
	year = {2024},
	url = {https://github.com/Petr-Ermolinskiy/2D_RBC_aggregation_simulation}
}
```
