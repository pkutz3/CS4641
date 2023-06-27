import numpy as np
from typing import Tuple, List


class Regression(object):
    def __init__(self):
        pass

    def rmse(self, pred: np.ndarray, label: np.ndarray) -> float:  # [5pts]
        """
        Calculate the root mean square error.

        Args:
            pred: (N, 1) numpy array, the predicted labels
            label: (N, 1) numpy array, the ground truth labels
        Return:
            A float value
        """
        #rmse formula
        return np.sqrt((np.sum((label-pred)**2))/pred.shape[0])
        #raise NotImplementedError

    def construct_polynomial_feats(
        self, x: np.ndarray, degree: int
    ) -> np.ndarray:  # [5pts]
        """
        Given a feature matrix x, create a new feature matrix
        which is all the possible combinations of polynomials of the features
        up to the provided degree

        Args:
            x: N x D numpy array, where N is number of instances and D is the
               dimensionality of each instance.
            degree: the max polynomial degree
        Return:
            feat:
                For 1-D array, numpy array of shape Nx(degree+1), remember to include
                the bias term. feat is in the format of:
                [[1.0, x1, x1^2, x1^3, ....,],
                 [1.0, x2, x2^2, x2^3, ....,],
                 ......
                ]
        Hints:
            - For D-dimensional array: numpy array of shape N x (degree+1) x D, remember to include
            the bias term.
            - Example:
            For inputs x: (N = 3 x D = 2) and degree: 3,
            feat should be:

            [[[ 1.0        1.0]
                [ x_{1,1}    x_{1,2}]
                [ x_{1,1}^2  x_{1,2}^2]
                [ x_{1,1}^3  x_{1,2}^3]]

                [[ 1.0        1.0]
                [ x_{2,1}    x_{2,2}]
                [ x_{2,1}^2  x_{2,2}^2]
                [ x_{2,1}^3  x_{2,2}^3]]

                [[ 1.0        1.0]
                [ x_{3,1}    x_{3,2}]
                [ x_{3,1}^2  x_{3,2}^2]
                [ x_{3,1}^3  x_{3,2}^3]]]

        """
        if x.ndim == 2:
            N = x.shape[0]
            D = x.shape[1]
            feat = np.array([np.full((degree+1,D),1.0)for i in range(N)])
            for i in range(N):
               for j in range(degree+1):
                    if j == 0:
                        continue
                    for d in range(D):
                        feat[i,j,d] = x[i,d]**j
        elif x.ndim == 1:
            N = x.shape[0]
            feat = np.full((N,degree+1),1.0)
            for i in range(degree+1):
                if i == 0:
                    continue
                feat[:,i] = x**i
        return feat
        #raise NotImplementedError

    def predict(self, xtest: np.ndarray, weight: np.ndarray) -> np.ndarray:  # [5pts]
        """
        Using regression weights, predict the values for each data point in the xtest array

        Args:
            xtest: (N,D) numpy array, where N is the number
                   of instances and D is the dimensionality
                   of each instance
            weight: (D,1) numpy array, the weights of linear regression model
        Return:
            prediction: (N,1) numpy array, the predicted labels
        """
        return np.matmul(xtest,weight)
        #raise NotImplementedError

    # =================
    # LINEAR REGRESSION
    # =================

    def linear_fit_closed(
        self, xtrain: np.ndarray, ytrain: np.ndarray
    ) -> np.ndarray:  # [5pts]
        """
        Fit a linear regression model using the closed form solution

        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
        Hints:
            - For pseudo inverse, you can use numpy linear algebra function (np.linalg.pinv)
        """
        return np.matmul(np.linalg.pinv(xtrain),ytrain)
        #raise NotImplementedError

    def linear_fit_GD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        epochs: int = 5,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a linear regression model using gradient descent

        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        """
        N = xtrain.shape[0]
        D = xtrain.shape[1]

        weight = np.zeros((D,1))
        loss_per_epoch = np.zeros(epochs)

        for epoch in range(epochs):
            predicted = np.matmul(xtrain,weight)
            weight += (learning_rate*np.matmul(xtrain.T,ytrain-predicted))/N
            predicted = np.matmul(xtrain,weight)
            loss_per_epoch[epoch] = self.rmse(predicted,ytrain)

        return weight,loss_per_epoch
        #raise NotImplementedError

    def linear_fit_SGD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a linear regression model using stochastic gradient descent

        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step

        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.


        Note: Keep in mind that the number of epochs is the number of
        complete passes through the training dataset. SGD updates the
        weight for one datapoint at a time, but for each epoch, you'll
        need to go through all of the points.
        """
        N = xtrain.shape[0]
        D = xtrain.shape[1]

        weight = np.zeros((D,1))
        loss_per_step = []

        for i in range(epochs):
            for j in range(N):
                predicted = np.matmul(xtrain[j],weight)
                temp = xtrain[j]*(ytrain[j]-predicted)
                weight += learning_rate * temp[:,np.newaxis]
                predicted = np.matmul(xtrain,weight)
                loss_per_step.append(self.rmse(predicted,ytrain))

        return weight, loss_per_step
        #raise NotImplementedError

    # =================
    # RIDGE REGRESSION
    # =================

    def ridge_fit_closed(
        self, xtrain: np.ndarray, ytrain: np.ndarray, c_lambda: float
    ) -> np.ndarray:  # [5pts]
        """
        Fit a ridge regression model using the closed form solution

        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value
        Return:
            weight: (D,1) numpy array, the weights of ridge regression model
        Hints:
            - For pseudo inverse, you can use numpy linear algebra function (np.linalg.pinv)
            - You should adjust your I matrix to handle the bias term differently than the rest of the terms
        """
        D = xtrain.shape[1]
        I = np.eye(D)
        I[0,0]=0
        xTx = np.matmul(xtrain.T,xtrain)
        xTxcI = np.linalg.pinv(xTx+(c_lambda*I))
        xTy = np.matmul(xtrain.T,ytrain)
        weight = np.matmul(xTxcI,xTy)
        return weight
        #raise NotImplementedError

    def ridge_fit_GD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        c_lambda: float,
        epochs: int = 500,
        learning_rate: float = 1e-7,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a ridge regression model using gradient descent.

        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
            c_lambda: float value
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_epoch: (epochs,) list of floats, rmse of each epoch
        """
        N = xtrain.shape[0]
        D = xtrain.shape[1]

        weight = np.zeros((D,1))
        loss_per_epoch = np.zeros(epochs)

        for epoch in range(epochs):
            predicted = np.matmul(xtrain,weight)
            lgd = (learning_rate*np.matmul(xtrain.T,ytrain-predicted))/N
            weight += 2*c_lambda*learning_rate*weight + lgd
            predicted = np.matmul(xtrain,weight)
            loss_per_epoch[epoch] = self.rmse(predicted,ytrain)

        return weight,loss_per_epoch
        #raise NotImplementedError

    def ridge_fit_SGD(
        self,
        xtrain: np.ndarray,
        ytrain: np.ndarray,
        c_lambda: float,
        epochs: int = 100,
        learning_rate: float = 0.001,
    ) -> Tuple[np.ndarray, List[float]]:  # [5pts]
        """
        Fit a ridge regression model using stochastic gradient descent.

        Args:
            xtrain: (N,D) numpy array, where N is number
                    of instances and D is the dimensionality of each
                    instance
            ytrain: (N,1) numpy array, the true labels
        Return:
            weight: (D,1) numpy array, the weights of linear regression model
            loss_per_step: (N*epochs,) list of floats, rmse calculated after each update step

        NOTE: For autograder purposes, iterate through the dataset SEQUENTIALLY, NOT stochastically.

        Note: Keep in mind that the number of epochs is the number of
        complete passes through the training dataset. SGD updates the
        weight for one datapoint at a time, but for each epoch, you'll
        need to go through all of the points.
        """
        raise NotImplementedError

    def ridge_cross_validation(
        self, X: np.ndarray, y: np.ndarray, kfold: int = 10, c_lambda: float = 100
    ) -> float:  # [5 pts]
        """
        For each of the kfolds of the provided X, y data, fit a ridge regression model
        and then evaluate the RMSE. Return the mean RMSE across all kfolds

        Args:
            X : (N,D) numpy array, where N is the number of instances and D is the dimensionality of each instance
            y : (N,1) numpy array, true labels
            kfold: Number of folds you should take while implementing cross validation.
            c_lambda: Value of regularization constant
        Returns:
            meanErrors: float, average rmse error
        Hints:
            - np.concatenate might be helpful.
            - Use ridge_fit_closed for this function.
            - Look at 3.5 to see how this function is being used.
            - If kfold=10:
                split X and y into 10 equal-size folds
                use 90 percent for training and 10 percent for test
        """
        N = X.shape[0]
        D = X.shape[1]
        rmse = []

        for k in range(kfold):
            start = int((k/kfold)*N)
            end = int(((k+1)/kfold)*N)
            xtest = X[start:end]
            ytest = y[start:end]
            xtrain = np.array([x for i,x in enumerate(X) if i not in range(start,end)])
            ytrain = np.array([yt for i,yt in enumerate(y) if i not in range(start,end)])

            weights = self.ridge_fit_closed(xtrain,ytrain,c_lambda)
            predicted = self.predict(xtest,weights)
            result = self.rmse(predicted,ytest)
            rmse.append(result)
        return np.mean(rmse)
        #raise NotImplementedError

    def hyperparameter_search(
        self, X: np.ndarray, y: np.ndarray, lambda_list: List[float], kfold: int
    ) -> Tuple[float, float, List[float]]:
        """
        PROVIDED TO STUDENTS
        Search over the given list of possible lambda values lambda_list
        for the one that gives the minimum average error from cross-validation

        Args:
            X : (N,D) numpy array, where N is the number of instances and D is the dimensionality of each instance
            y : (N,1) numpy array, true labels
            lambda_list: list of regularization constants to search from
            kfold: Number of folds you should take while implementing cross validation.
        Returns:
            best_lambda: (float) the best value for the regularization const giving the least RMSE error
            best_error: (float) the RMSE error achieved using the best_lambda
            error_list: list[float] list of errors for each lambda value given in lambda_list
        """

        best_error = None
        best_lambda = None
        error_list = []

        for lm in lambda_list:
            err = self.ridge_cross_validation(X, y, kfold, lm)
            error_list.append(err)
            if best_error is None or err < best_error:
                best_error = err
                best_lambda = lm

        return best_lambda, best_error, error_list
