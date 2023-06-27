import numpy as np
from matplotlib import pyplot as plt


class PCA(object):
    def __init__(self):
        self.U = None
        self.S = None
        self.V = None

    def fit(self, X: np.ndarray) -> None:  # 5 points
        """
        Decompose dataset into principal components by finding the singular value decomposition of the centered dataset X
        You may use the numpy.linalg.svd function
        Don't return anything. You can directly set self.U, self.S and self.V declared in __init__ with
        corresponding values from PCA. See the docstrings below for the expected shapes of U, S, and V

        Args:
            X: (N,D) numpy array corresponding to a dataset

        Return:
            None

        Set:
            self.U: (N, min(N,D)) numpy array
            self.S: (min(N,D), ) numpy array
            self.V: (min(N,D), D) numpy array
        """
        N = X.shape[0]
        D = X.shape[1]
        #center dataset
        X_centered = X - np.mean(X,axis=0)
        U, S, V = np.linalg.svd(X_centered,full_matrices=False)
        #get correct USV based on above shapes
        self.U = U[:,:min(N,D)]
        self.S = S[:min(N,D)]
        self.V = V[:min(N,D),:]
        #raise NotImplementedError

    def transform(self, data: np.ndarray, K: int = 2) -> np.ndarray:  # 2 pts
        """
        Transform data to reduce the number of features such that final data (X_new) has K features (columns)
        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            K: int value for number of columns to be kept

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data
        """
        return np.matmul(self.U[:,:K],np.diag(self.S[:K]))
        #raise NotImplementedError

    def transform_rv(
        self, data: np.ndarray, retained_variance: float = 0.99
    ) -> np.ndarray:  # 3 pts
        """
        Transform data to reduce the number of features such that the retained variance given by retained_variance is kept
        in X_new with K features
        Utilize self.U, self.S and self.V that were set in fit() method.

        Args:
            data: (N,D) numpy array corresponding to a dataset
            retained_variance: float value for amount of variance to be retained

        Return:
            X_new: (N,K) numpy array corresponding to data obtained by applying PCA on data, where K is the number of columns
                   to be kept to ensure retained variance value is retained_variance
        """
        #finds retained variance where index + 1 = k
        ret_var = np.cumsum(self.S**2)/np.sum(self.S**2)
        for i in range(len(ret_var)):
            #find first k greater than retained variance
            if ret_var[i] >= retained_variance:
                return self.transform(data,i+1)

        #raise NotImplementedError

    def get_V(self) -> np.ndarray:
        """ Getter function for value of V """

        return self.V

    def visualize(self, X: np.ndarray, y: np.ndarray, fig=None) -> None:  # 5 pts
        """
        Use your PCA implementation to reduce the dataset to only 2 features. You'll need to run PCA on the dataset and then transform it so that the new dataset only has 2 features.
        Create a scatter plot of the reduced data set and differentiate points that have different true labels using color.
        Hint: To create the scatter plot, it might be easier to loop through the labels (Plot all points in class '0', and then class '1')
        Hint: To reproduce the scatter plot in the expected outputs, use the colors 'blue', 'magenta', and 'red' for classes '0', '1', '2' respectively.
        
        Args:
            xtrain: (N,D) numpy array, where N is number of instances and D is the dimensionality of each instance
            ytrain: (N,) numpy array, the true labels
            
        Return: None
        """
        self.fit(X)
        Xnew = self.transform(X,2)
        for i, label in enumerate(y):
            if label == 0:
                plt.scatter(Xnew[i,0],Xnew[i,1],c='blue')
            elif label == 1:
                plt.scatter(Xnew[i,0],Xnew[i,1],c='magenta')
            elif label == 2:
                plt.scatter(Xnew[i,0],Xnew[i,1],c='red')

        #raise NotImplementedError

        ##################### END YOUR CODE ABOVE, DO NOT CHANGE BELOW #######################
        plt.legend()
        plt.show()
