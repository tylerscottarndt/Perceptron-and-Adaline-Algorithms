import numpy as np
import pandas as pd


class AdalineGD(object):
    """ADAptive LInear NEuron classifier.

    Parameters
    ------------
    eta : float
      Learning rate (between 0.0 and 1.0)
    n_iter : int
      Passes over the training dataset.
    random_state : int
      Random number generator seed for random weight
      initialization.


    Attributes
    -----------
    w_ : 1d-array
      Weights after fitting.
    cost_ : list
      Sum-of-squares cost function value in each epoch.

    """
    def __init__(self, eta=0.01, n_iter=50, random_state=1):
        self.eta = eta
        self.n_iter = n_iter
        self.random_state = random_state

    def fit(self, X, y):
        """ Fit training data.

        Parameters
        ----------
        X : {array-like}, shape = [n_samples, n_features]
          Training vectors, where n_samples is the number of samples and
          n_features is the number of features.
        y : array-like, shape = [n_samples]
          Target values.

        Returns
        -------
        self : object

        """
        rgen = np.random.RandomState(self.random_state)
        self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
        self.cost_ = []

        for i in range(self.n_iter):
            net_input = self.net_input(X)
            # Please note that the "activation" method has no effect
            # in the code since it is simply an identity function. We
            # could write `output = self.net_input(X)` directly instead.
            # The purpose of the activation is more conceptual, i.e.,
            # in the case of logistic regression (as we will see later),
            # we could change it to
            # a sigmoid function to implement a logistic regression classifier.
            output = self.activation(net_input)
            errors = (y - output)
            self.w_[1:] += self.eta * X.T.dot(errors)
            self.w_[0] += self.eta * errors.sum()
            cost = (errors**2).sum() / 2.0
            self.cost_.append(cost)
        return self

    def net_input(self, X):
        """Calculate net input"""
        return np.dot(X, self.w_[1:]) + self.w_[0]

    def activation(self, X):
        """Compute linear activation"""
        return X

    def predict(self, X):
        """Return class label after unit step"""
        return np.where(self.activation(self.net_input(X)) >= 0.0, 1, 0)


def format_data(file_name):
    td = pd.read_csv(file_name)

    # Pclass, Sex, Age, Sibsp, Parch, Fare, Embarked
    X = td.drop(columns=['Survived', 'PassengerId', 'Name', 'Ticket', 'Cabin'])
    y = td.iloc[0:, 1]

    # change strings to integer values
    X = X.replace(to_replace='male', value=1)
    X = X.replace(to_replace='female', value=2)
    X = X.replace(to_replace='C', value=1)
    X = X.replace(to_replace='Q', value=2)
    X = X.replace(to_replace='S', value=3)


    # find most common or average values for NaN replacement
    common_class = X['Pclass'].value_counts().idxmax()
    common_sex = X['Sex'].value_counts().idxmax()
    mean_age = X['Age'].mean()
    mean_sibsp = X['SibSp'].mean()
    mean_parch = X['Parch'].mean()
    mean_fare = X['Fare'].mean()
    common_embarked = X['Embarked'].value_counts().idxmax()

    values = {'Pclass': common_class, 'Sex': common_sex, 'Age': mean_age, 'SibSp': mean_sibsp, 'Parch': mean_parch,
              'Fare': mean_fare, 'Embarked': common_embarked}
    X = X.fillna(value=values)

    return X, y


def train_the_algorithm(file_name):
    data_tuple = format_data(file_name)
    X = data_tuple[0]
    y = data_tuple[1]

    X = np.copy(X)
    X_std = np.copy(X)

    X_std[:, 0] = (X[:, 0] - X[:, 0].mean()) / X[:, 0].std()
    X_std[:, 1] = (X[:, 1] - X[:, 1].mean()) / X[:, 1].std()
    X_std[:, 2] = (X[:, 2] - X[:, 2].mean()) / X[:, 2].std()
    X_std[:, 3] = (X[:, 3] - X[:, 3].mean()) / X[:, 3].std()
    X_std[:, 4] = (X[:, 4] - X[:, 4].mean()) / X[:, 4].std()
    X_std[:, 5] = (X[:, 5] - X[:, 5].mean()) / X[:, 5].std()
    X_std[:, 6] = (X[:, 6] - X[:, 6].mean()) / X[:, 6].std()

    # adaline = AdalineGD(n_iter=100, eta=0.0000001).fit(X_std, y)
    adaline = AdalineGD(n_iter=50, eta=0.0001).fit(X_std, y)
    print(adaline.w_)
    return adaline


def test_algorithm(train_file_name, test_file_name):
    algorithm = train_the_algorithm(train_file_name)
    data_tuple = format_data(test_file_name)
    X = data_tuple[0]
    y = data_tuple[1]

    prediction = algorithm.predict(X)

    # test prediction
    i = 0
    mistakes = 0
    while i < 50:
        if prediction[i] != y[i]:
            mistakes += 1
        i += 1

    accuracy = (len(X)-mistakes) / len(X) * 100

    print("The algorithm made " + str(mistakes) + " mistakes out of " + str(len(X)) + " predictions for a total accuracy of " + str(accuracy) + "%")


test_algorithm('train.csv', 'test.csv')
