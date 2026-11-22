import numpy as np


class SVM:
    """
    Linear Soft-Margin Support Vector Machine.

    Formulation:
        J(w, b) = (lambda / 2) * ||w||^2 + (1 / m) * sum(max(0, 1 - y_i * f(x_i)))

    Decision function:
        f(x) = w.T @ x + b

    Optimization:
        Batch Gradient Descent

    Labels:
        -1 and +1
    """

    def __init__(self, iters=1000, eta=0.01, lambdaP=0.01):
        """ 
        Initialize the SVM model with hyperparameters.

        Args:
            iters (int, optional): Number of training iterations. Defaults to 1000.
            eta (float, optional): Learning rate. Defaults to 0.01.
            lambdaP (float, optional): Regularization parameter. Defaults to 0.01.
        """
        self.iters = iters
        self.eta = eta
        self.lambdaP = lambdaP
        self.w = None
        self.b = None
        self.objective_history = []
        self.hinge_loss_history = []
        self.accuracy_history = []

    # --------------------------------------------------
    def _validate_data(self, X, Y):
        """Validate input data for training.

        Args:
            X (array-like of shape (m, n)): Training features, 
                where m is the number of samples and n is the number of features.
            Y (array-like of shape (m,)): Training labels, encoded as -1 and +1.

        Raises:
            ValueError: If X is not a 2-dimensional array.
            ValueError: If Y is not a 1-dimensional array or its length does not 
                match the number of samples in X.
            ValueError: If the training data is empty.
            ValueError: If X or Y contains NaN or infinity values.
            ValueError: If Y contains labels other than -1 and +1.

        Returns:
            tuple: Validated X and Y as numpy arrays.
        """

        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        if X.ndim != 2:
            raise ValueError("X harus berupa array 2 dimensi.")

        if Y.ndim != 1 or len(Y) != len(X):
            raise ValueError("Y harus berupa array 1 dimensi dengan panjang sesuai X.")

        if len(X) == 0 or X.shape[1] == 0:
            raise ValueError("Data training tidak boleh kosong.")

        if not np.all(np.isfinite(X)) or not np.all(np.isfinite(Y)):
            raise ValueError("X dan Y tidak boleh mengandung NaN atau infinity.")

        if not np.all(np.isin(Y, [-1, 1])):
            raise ValueError("Label harus menggunakan -1 dan +1.")

        return X, Y

    # --------------------------------------------------
    def fit(self, X, Y):
        """
        Fit the SVM model to the training data.

        Args: X and Y

        Raises:
            ValueError: If the hyperparameters are not valid.

        Returns:
            self: The fitted SVM model.
        """
        X, Y = self._validate_data(X, Y)

        if self.iters <= 0 or self.eta <= 0 or self.lambdaP < 0:
            raise ValueError("Hyperparameter tidak valid.")

        m, n = X.shape

        # Initialize parameters
        self.w = np.zeros(n)
        self.b = 0.0
        self.objective_history = []
        self.hinge_loss_history = []
        self.accuracy_history = []

        # Inital metrics
        self.objective_history.append(self.objective_function(X, Y))
        self.hinge_loss_history.append(self.mean_hinge_loss(X, Y))
        self.accuracy_history.append(np.mean(self.predict(X) == Y))

        # Process optimization
        for _ in range(self.iters):

            # 1. Decision function
            output = self.decision_function(X)

            # 2. Margin
            margin = Y * output

            # 3. Identify samples that violate the margin
            condition = margin < 1

            # 4. Gradient calculation
            dw = self.lambdaP * self.w - (np.dot(X[condition].T, Y[condition]) / m)
            db = -np.sum(Y[condition]) / m

            # 5. Update weights and bias
            self.w -= self.eta * dw
            self.b -= self.eta * db

            # 6. Store objective function value for monitoring
            self.objective_history.append(self.objective_function(X, Y))
            self.hinge_loss_history.append(self.mean_hinge_loss(X, Y))
            self.accuracy_history.append(np.mean(self.predict(X) == Y))

        return self

    # --------------------------------------------------
    def decision_function(self, X):
        """
        Compute the decision function for the input samples.

        Args: X

        Raises:
            ValueError: If the model has not been trained using fit().

        Returns:
            numpy.ndarray of shape (m,): The decision function values for 
                the input samples.
        """
        if self.w is None:
            raise ValueError("Model belum dilatih. Jalankan fit() terlebih dahulu.")

        X = np.asarray(X, dtype=float)

        return np.dot(X, self.w) + self.b

    # --------------------------------------------------
    def predict(self, X):
        """
        Predict the class labels for the input samples.

        Args: X

        Returns:
            numpy.ndarray of shape (m,): Predicted class labels encoded
            as -1 and +1.
        """
        output = self.decision_function(X)

        return np.where(output >= 0, 1, -1)

    # --------------------------------------------------
    def mean_hinge_loss(self, X, Y):
        """
        Compute the mean hinge loss for the given samples and labels.

        """
        if self.w is None:
            raise ValueError("Model belum dilatih.")

        X, Y = self._validate_data(X, Y)

        output = self.decision_function(X)
        margin = Y * output

        return np.maximum(0, 1 - margin).mean()

    # --------------------------------------------------
    def objective_function(self, X, Y):
        """
        Compute the value of the objective function.
         
        """
        if self.w is None:
            raise ValueError("Model belum dilatih.")

        X, Y = self._validate_data(X, Y)

        output = self.decision_function(X)
        margin = Y * output

        hinge_loss = self.mean_hinge_loss(X, Y)

        regularization = (self.lambdaP / 2) * np.dot(self.w, self.w)

        J = regularization + hinge_loss

        return J
