[![arXiv](https://img.shields.io/badge/arXiv-2212.06370-b31b1b.svg)](https://arxiv.org/abs/2212.06370)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/NISL-MSU/PredictionIntervals/blob/master/DualAQD_PredictionIntervals.ipynb)

# DualAQD: Dual Accuracy-quality-driven Prediction Intervals

## Description

We present a method to learn prediction intervals for regression-based neural networks automatically in addition to the conventional target predictions.
In particular, we train two companion neural networks: one that uses one output, the target estimate, and another that uses two outputs, the upper and lower bounds of the corresponding PI.
We designed a loss function, [DualAQD](https://arxiv.org/abs/2212.06370), for the PI-generation network that takes into account the output of the target-estimation network and has two optimization objectives: minimizing the mean prediction interval width and ensuring the PI integrity using constraints that maximize the prediction interval probability coverage implicitly.
Both objectives are balanced within the loss function using a self-adaptive coefficient.
Furthermore, we apply a Monte Carlo-based approach that evaluates the model uncertainty in the learned PIs.

<p align="center">
  <img src="https://raw.githubusercontent.com/GiorgioMorales/PredictionIntervals/master/images/introduction.jpg" alt="alt text" width="400">
</p>

## Installation

The following libraries have to be installed:
* [Git](https://git-scm.com/download/) 
* [Pytorch](https://pytorch.org/)

To install the package, run `!pip install -q git+https://github.com/NISL-MSU/PredictionIntervals` in the terminal. 
This will also install additional packages such as pymoo, sklearn, and tensorboard.

You can also try the package on [Google Colab](https://colab.research.google.com/github/NISL-MSU/PredictionIntervals/blob/master/DualAQD_PredictionIntervals.ipynb).

## Usage

### Train the models

DualAQD uses two neural networks: a target-estimation network $f$ that is trained to generate accurate estimates, and a PI-generation NN $g$ that produces the upper and lower bounds of a prediction interval.

First, create an instance of the class `PredictionIntervalsTrainer.`

**Parameters**:

*   `X`: Input data (explainable variables). 2-D numpy array, shape (#samples, #features)
*   `Y`: Target data (response variable). 1-D numpy array, shape (#samples, #features)
*   `Xval`: Validation input data. 2-D numpy array, shape (#samples, #features)
*   `Yval`: Validation target data. 1-D numpy array, shape (#samples, #features)
*   `method`: PI-generation method. Options: 'DualAQD' or '[MCDropout](https://arxiv.org/pdf/1709.01907.pdf)'
*   `normData`: If True, apply z-score normalization to the inputs and min-max normalization to the outputs

**Note**: Normalization is applied to the training set; then, the exact same scaling is applied to the validation set.

```python
from PredictionIntervals.Trainer.TrainNN import Trainer
trainer = Trainer(X=Xtrain, Y=Ytrain, Xval=Xval, Yval=Yval)
```

To train the model, we'll call the `train` method.

**Parameters**:

*   `batch_size`: Mini batch size. It is recommended a small number. *default: 16*
*   `epochs`: Number of training epochs *default: 1000*
*   `eta_`: Scale factor used to update the self-adaptive coefficient lambda (Eq. 6 of the paper). *default: 0.01*
*   `printProcess`: If True, print the training process (loss and validation metrics after each epoch). *default: False*
*   `plotCurves`: If True, plot the training and validation curves at the end of the training process

```python
trainer.train(printProcess=False, epochs=2000, batch_size=16, plotCurves=True)
```

### Evaluate the model on the test set

To do this, we call the method `evaluate`.

**Parameters**:

*   `Xeval`: Evaluation data
*   `Yeval`: Optional. Evaluation targets. *default: None*
*   `normData`: If True, apply the same normalization that was applied to the training set

**Note**: `Yeval` is *None* in the case that the target values of the evaluation data are not known.

**Returns**:
*   If `Yeval` is *None*: It returns predictions `ypred, y_u, y_l` (i.e., target predictions, PI upper bounds, and PI lower bounds).
*   If `Yeval` is not *None*: It returns performance metrics and predictions `mse, PICP, MPIW, ypred, y_u, y_l` (i.e., mean square error of target predictions, PI coverage probability, mean PI width, target predictions, PI upper bounds, and PI lower bounds).

```python
val_mse, PICP, MPIW, ypred, y_u, y_l = trainer.evaluate(Xtest, Ytest, normData=True)
print('Test Performance:')
print("Test MSE: " + str(val_mse) + " Test PICP: " + str(PICP) + " Test MPIW: " + str(MPIW))
```

## Repository Structure

This repository contains the following scripts in `src/PredictionIntervals/`:

* `TrainNN.py`: Main method, used to train a PI-generation NN using DualAQD or MCDropout-PI.
* `PIGenerator.py`: Performs cross-validation using different NN-based PI-generation methods. Used for replication of the paper results.        
* `utils.py`: Additional methods used to transform the data and calculate the metrics.
* `utils.py`: Additional methods used to transform the data and calculate the metrics. 
* `models/NNmodel.py`: Implements the PI-generation methods tested in this work: DualAQD, QD+, QD, MC-Dropout.
* `models/network.py`: Defines the network architecture.
* `Demo.ipynb`: Jupyter notebook demo using a synthetic dataset.

# Citation
Use this Bibtex to cite this repository

```
@ARTICLE{10365540,
  author={Morales, Giorgio and Sheppard, John W.},
  journal={IEEE Transactions on Neural Networks and Learning Systems}, 
  title={Dual Accuracy-Quality-Driven Neural Network for Prediction Interval Generation}, 
  year={2023},
  volume={},
  number={},
  pages={1-11},
  doi={10.1109/TNNLS.2023.3339470}
}
```
