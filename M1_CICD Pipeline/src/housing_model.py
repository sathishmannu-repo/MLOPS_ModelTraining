from sklearn.linear_model import LinearRegression

class HousingModel:
    """
    Encapsulates a Linear Regression model
    """

    def __init__(self):
        """
        Initializes the Linear Regression model.
        """
        self.model = LinearRegression()

    def train(self, X_train, y_train):
        """
        Trains the Linear Regression model.
        :param X_train: Training feature data.
        :param y_train: Training target data.
        """
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        """
        Predicts the target values for the given test data.
        :param X_test: Test feature data.
        :return: Predicted target values.
        """
        return self.model.predict(X_test)
