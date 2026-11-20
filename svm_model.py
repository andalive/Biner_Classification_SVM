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
        self.iters = iters
        self.eta = eta
        self.lambdaP = lambdaP
        self.w = None
        self.b = None
        self.loss_history = []

    # Training
    # --------------------------------------------------
    def fit(self, X, Y):
        X =  np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        # Validasi dimensi dan label
        if X.ndim != 2:
            raise ValueError("X harus berupa array 2 dimensi.")

        if Y.ndim != 1 or len(Y) != len(X):
            raise ValueError("Y harus berupa array 1 dimensi dengan panjang sesuai X.")

        if len(X) == 0:
            raise ValueError("Data training tidak boleh kosong.")

        if not np.all(np.isin(Y, [-1, 1])):
            raise ValueError("Label harus menggunakan -1 dan +1.")

        if self.iters <= 0 or self.eta <= 0 or self.lambdaP < 0:
            raise ValueError("Hyperparameter tidak valid.")

        m, n = X.shape

        # Inisialisasi parameter
        self.w = np.zeros(n)
        self.b = 0.0
        self.loss_history = []

        # Proses optimisasi
        for _ in range(self.iters):
            output = np.dot(X, self.w) + self.b # Decision function
            margin = Y * output                 # Margin
            condition = margin < 1              # Sampel dengan margin < 1

            # Gradient bobot dan bias
            dw = self.lambdaP * self.w - (np.dot(X[condition].T, Y[condition]) / m)
            db = -np.sum(Y[condition]) / m

            # Update parameter secara simultan
            self.w -= self.eta * dw
            self.b -= self.eta * db

            # Menghitung objective function setelah update
            updated_output = np.dot(X, self.w) + self.b
            updated_margin = Y * updated_output

            hinge_loss = np.maximum(0, 1 - updated_margin).mean()
            regularization = (self.lambdaP/2) * np.dot(self.w, self.w)
            loss = regularization + hinge_loss

            self.loss_history.append(loss)

        return self

    # Decision Function
    # --------------------------------------------------
    def decision_function(self, X):
        if self.w is None:
            raise ValueError("Model belum dilatih. Jalankan fit() terlebih dahulu.")

        X = np.asarray(X, dtype=float)

        return np.dot(X, self.w) + self.b

    # Prediction
    # --------------------------------------------------
    def predict(self, X):
        output = self.decision_function(X)

        return np.where(output >= 0, 1, -1)

    # Objective Function
    # --------------------------------------------------
    def objective_function(self, X, Y):
        if self.w is None:
            raise ValueError("Model belum dilatih.")

        X = np.asarray(X, dtype=float)
        Y = np.asarray(Y, dtype=float)

        output = self.decision_function(X)
        margin = Y * output
        hinge_loss = np.maximum(0, 1 - margin).mean()
        regularization = (self.lambdaP/2) * np.dot(self.w, self.w)

        return regularization + hinge_loss
