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
