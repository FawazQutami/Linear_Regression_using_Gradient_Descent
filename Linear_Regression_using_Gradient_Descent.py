# File: Linear_Regression_using_Gradient_Descent.py

import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# scikit learn
from sklearn import datasets
from sklearn.model_selection import train_test_split

import warnings
warnings.filterwarnings("ignore")


def mse(y_tst, y_predicted):
    """ Mean squared error regression loss """
    # MSE = 1/N ∑i=1:n (yi−(xi.w + b))^2
    _MSE = np.mean(np.power((y_tst - y_predicted), 2))
    return _MSE


def rmse(y_tst, y_predicted):
    """ Root Mean Squared Error """
    _mse = mse(y_tst, y_predicted)
    return np.sqrt(_mse)


def r2_score(y_tst, y_predicted):
    """ (coefficient of determination) regression score function """
    # Sum of square of residuals
    RSS = np.sum((y_predicted - y_tst) ** 2)
    #  Total sum of squares
    TSS = np.sum((y_tst - np.mean(y_tst)) ** 2)
    # R2 score
    r2 = 1 - (RSS / TSS)

    return r2


class LinearRegressionGd:
    """ Linear Regression Using Gradient Descent """

    def __init__(self, alpha=0.01, n_iter=100):
        """   Constructor  """
        # # Initialize the parameters
        self.alpha = alpha
        self.n_iter = n_iter

        # Initialize the attributes
        self.weights = None
        self.bias = None

        self.info = []

    def __repr__(self):
        """
        Class Representation:  represent a class's objects as a string
        :return: string
        """
        df = pd.DataFrame.from_dict(self.info)
        df.set_index('Iteration', inplace=True)
        return f'\n ---------- \n Training Model Coefficients - verify the minimum cost: \n ----------\n {df}'

    def fit(self, x_trn, y_trn):
        """
        Fit Method
        :param x_trn: [array-like]
            it is n x m shaped matrix where n is number of samples
                                            and m is number of features
        :param y_trn: [array-like]
            it is n shaped matrix where n is number of samples
        :return: self:object
        """
        # Initialize the parameters
        n_samples, n_features = x_trn.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        # initialize a dict and a list to store the coefficients and cost
        self.info = []
        temp_dict = {}

        # Start the iteration
        for iteration in range(self.n_iter):
            # Calculate Y hat = weight.X + bias
            y_predicted = self.predict(x_trn)
            # Calculate the difference between predicted y and actual y
            residuals = y_predicted - y_trn

            # Cost Function --> MSE = 1 / N * ∑i=1:n (yi − (xi.w + b))^2
            cost = np.mean(np.power((y_trn - y_predicted), 2))

            """ Gradient Descent -- BackProp """
            self.gradient_descent(n_samples, x_trn, residuals)

            # Record the model info
            # Append cost, weight and bias information in a list
            temp_dict['Iteration'] = iteration
            for i in range(len(self.weights)):
                temp_dict['W' + str(i)] = self.weights[i]
            temp_dict['Bias'] = self.bias
            temp_dict['Cost'] = cost
            self.info.append(temp_dict.copy())

    def gradient_descent(self, n_samples, x_trn, residuals):
        """
        Gradient Descent -- BackProp
        :param n_samples: {int}
        :param x_trn: {array_like}
        :param residuals: {array_like}
        :return: None
        """
        """ 
        Gradient Descent:
        It is an iterative method to reach the deepest descent,
        which is toward the negative direction of the gradient.
        We do that, iteratively, to reach the global minimum.

        In each iteration, we have to update both weights
        and bias:
            weight = weight - self.alpha * change in weight
            bias = bias - self.alpha * change in bias
        """
        # Calculate the partial derivatives of the cost function:
        weight_derivative = (1 / n_samples) * np.dot(x_trn.T, residuals)
        bias_derivative = (1 / n_samples) * np.sum(residuals)

        """ 
        Update the parameters - We subtract because the
        # derivatives point in direction of steepest ascent
        # and we need the opposite direction.
        # The size of our update is controlled by the learning rate (alpha).
        """
        self.weights -= self.alpha * weight_derivative
        self.bias -= self.alpha * bias_derivative

    def predict(self, x_tst):
        """
        Prediction
        :param x_tst:{array_like}
        :return: {array_like} (0's and 1's)
        """
        # Calculate Y hat :: (y_hat = w.X + b)
        y_predicted = np.dot(x_tst, self.weights) + self.bias

        return y_predicted


def draw_plot(xs, p_line, x_trn, x_tst,
              y_trn, y_tst):
    """
    Plot the regression
    :param xs: {array_like}
    :param p_line: {array_like}
    :param x_trn: {array_like}
    :param x_tst: {array_like}
    :param y_trn: {array_like}
    :param y_tst: {array_like}
    :return: None
    """
    cmap = plt.get_cmap('viridis')
    fig = plt.figure(figsize=(8, 6))
    plt.scatter(x_trn,
                y_trn,
                color=cmap(0.8),
                s=20)
    plt.scatter(x_tst,
                y_tst,
                color=cmap(0.2),
                s=20)
    plt.plot(xs, p_line,
             color='r',
             linewidth=1,
             label="Predicted Values - Best Fit Line")
    plt.legend()
    plt.show()


if __name__ == '__main__':
    try:
        # Create regression data
        X, y = datasets.make_regression(n_samples=1000,
                                        n_features=2,
                                        noise=20,
                                        random_state=5)
        # Split the data to training and test sets
        X_train, X_test, y_train, y_test = train_test_split(X,
                                                            y,
                                                            test_size=0.2,
                                                            random_state=5)

        start = time.time()

        # Create a Linear Regression object
        reg = LinearRegressionGd(alpha=0.5, n_iter=1000)
        # Fit the training model using gradient descent
        reg.fit(X_train, y_train)
        # Print the object information - to verify the minimum
        print(reg)
        # Predict labels using the test data and the training model parameters
        y_predictions = reg.predict(X_test)

        # Measure the accuracy between the actual data and the predicted ones
        _mse = mse(y_test, y_predictions)
        print("\n ----------\n MSE: {:.2f}".format(_mse))
        _rmse = rmse(y_test, y_predictions)
        print("\n ----------\n RMSE: {:.2f}".format(_rmse))
        _r2_score = r2_score(y_test, y_predictions)
        print("\n ----------\n R^2 score: %.2f%%" % (_r2_score * 100))

        end = time.time()  # ----------------------------------------------
        print('\n ----------\n Execution Time: {%f}' \
              % ((end - start) / 1000) + ' seconds.')

        # Draw the regression
        predicted_line = reg.predict(X)
        draw_plot(X, predicted_line, X_train, X_test, y_train, y_test)

    except:
        pass
