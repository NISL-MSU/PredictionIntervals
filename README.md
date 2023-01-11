# DualAQD: Dual Accuracy-quality-driven Prediction Intervals

## Description

We present a method to learn prediction intervals for regression-based neural networks automatically in addition to the conventional target predictions.
In particular, we train two companion neural networks: one that uses one output, the target estimate, and another that uses two outputs, the upper and lower bounds of the corresponding PI.
We designed a loss function, [DualAQD](https://arxiv.org/abs/2212.06370), for the PI-generation network that takes into account the output of the target-estimation network and has two optimization objectives: minimizing the mean prediction interval width and ensuring the PI integrity using constraints that maximize the prediction interval probability coverage implicitly.
Both objectives are balanced within the loss function using a self-adaptive coefficient.
Furthermore, we apply a Monte Carlo-based approach that evaluates the model uncertainty in the learned PIs.

## Usage

This repository contains the following scripts:

* `PIGenerator.py`: Contains the PIGenerator class that is used to perform cross-validation using different NN-based PI-generation methods.        
* `utils.py`: Additional methods used to transform the data and calculate the metrics. 
* `models/NNmodel.py`: Implements the PI-generation methods tested in this work: DualAQD, QD+, QD, MC-Dropout.
* `models/network.py`: Defines the network architecture.
* `Demo.ipynb`: Jupyter notebook demo using a synthetic dataset.

# Citation
Use this Bibtex to cite this repository

```
@misc{DualAQD,
  doi = {10.48550/ARXIV.2212.06370},  
  author = {Morales, Giorgio and Sheppard, John W.},  
  title = {Dual Accuracy-Quality-Driven Neural Network for Prediction Interval Generation},
  publisher = {arXiv},  
  year = {2022}
}
```