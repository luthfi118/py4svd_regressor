# py4svd-regressor

SVD Based Linear Regression Python Library

## Getting Started

This project is simply an implementation of a Linear Regression algorithm based on SVD in python programming language

### Prerequisites

Numpy

### Installation

The easiest way to install py4svd-regressor is by using pip

```
pip install py4svd-regressor
```

### Usage
2 public methods are provided namely ```learn``` and ```predict```. The ```learn``` method used to train the model. It takes 2 arguments, the data, and its target. The ```predict``` method used to predict the given data, it takes 1 argument, it is the  data user wanted to predict. It returns the resulting prediction. The weight and intercept are stored on attributes namely ```w``` and ```b``` respectively
```
from py4svd_regressor.regression import Svd_Regressor
from sklearn.datasets import make_regression
from matplotlib import pyplot

x, y = make_regression(n_samples=200, n_features=1, noise=5)
model = Svd_Regressor()
model.train(x, y)
pyplot.scatter(x, y)
pyplot.plot(x, model.w*x + model.b, color='red')
pyplot.show()
y_predict = model.predict(x)
```
<a href="https://ibb.co/1bmf57T"><img src="https://i.ibb.co/QXKFRpM/svd-regression.png" alt="svd-regression" border="0"></a>
