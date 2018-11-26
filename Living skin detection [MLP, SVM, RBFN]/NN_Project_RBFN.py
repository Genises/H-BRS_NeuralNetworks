class RBFN(object):

    def __init__(self, hidden_dimension, sigma=1.0):
        """ Radial basis function network (RBFN)
        # Arguments
            hidden_dimension: Integer indicating number of
                radial basis functions
            sigma: Float indicating the precision of the Gaussian.
        """
        self.hidden_dimension = hidden_dimension
        self.sigma = sigma
        self.centers = None
        self.weights = None

    def _kernel_function(self, center, data_point):
        """ Calculates the similarity/kernel function between
        the selected/constructed centers and the samples.
        # Arguments:
            center: numpy array of shape(, feature_dimension)
            data_points: numpy array of shape (, feature_dimension)
        # Returns:
            kernel_value: Float entry for the interpolation matrix.
        """
        kernel_value = np.exp(-(np.linalg.norm(data_point-center) ** 2 / self.sigma ** 2))
        return kernel_value

    def _calculate_interpolation_matrix(self, X):
        """ Calculates interpolation matrix using a kernel_function
        # Arguments
            X: numpy array of features
                with shape (num_samples, feature_dimension)
        # Returns
            G: Numpy array of the interpolation matrix with
                shape (num_samples, hidden_dimensions)
        """
        interpolation_matrix = np.zeros((X.shape[0], self.hidden_dimension), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                interpolation_matrix[xi,ci] = self._kernel_function(c, x)
        return interpolation_matrix

    def _select_centers(self, X):
        """ Selects/creates centers from features.
        # Arguments:
            X: numpy array containing features of
                shape (num_samples, feature_dimension)
        # Returns:
            centers: numpy array containing feature centers
                of shape (hidden_dimension, feature_dimension)
        """
        random_rows = np.random.randint(X.shape[0], size=self.hidden_dimension)
        centers = X[random_rows,]
        return centers

    def fit(self, X, Y):
        """ Fits weights using linear regression
        # Arguments
            X: numpy array containing features of
                shape (num_samples, feature_dimension)
            Y: numpy array containing the targets
                of shape (num_samples, feature_dimension)
        """
        # select centers randomly from X
        self.centers = self._select_centers(X)
        #print("centers", self.centers)
        
        # calculate interpolation matrix 
        interpolation_matrix = self._calculate_interpolation_matrix(X)
        #print("interpolation matrix", interpolation_matrix)
        
        # train, adjust weights
        self.weights = np.dot(np.linalg.pinv(interpolation_matrix), Y)

    def predict(self, X):
        """
        # Arguments
            X: numpy array of features
                of shape (num_samples, feature_dimension)
        # Returns:
            predictions: numpy array of shape (num_samples, )
        """
        # calculate interpolation matrix 
        interpolation_matrix = self._calculate_interpolation_matrix(X)
        #print("interpolation matrix", interpolation_matrix)
        
        Y = np.dot(interpolation_matrix, self.weights)
        return Y
		
		
#####################################################################

	
	
	
model = RBFN(hidden_dimension=10, sigma=1.)
model.fit(trainingSet, trainingLabelSet)
y_pred = model.predict(testSet)

# Use softmax on ouput
y_pred_softmax = np.zeros_like(y_pred)
y_pred_softmax[np.arange(len(y_pred)), y_pred.argmax(1)] = 1

# Confusion map
cm = confusion_matrix(y_test.argmax(axis=1), y_pred_softmax.round().argmax(axis=1))
plot_confusion_matrix(cm, ["No skin", "Skin"], title="Confusion map for RBFN Neurons")
