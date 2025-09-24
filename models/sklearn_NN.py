from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.preprocessing import StandardScaler
import numpy as np
from .model import Model

class NNRegressor(Model):
    """
    A wrapper for the MLPRegressor from sklearn with hyperparameter tuning using GridSearchCV.
    """

    def __init__(self):
        """
        Initialize the NNRegressor with the given parameters.
        """
        self.Xscaler = StandardScaler()
        self.yscaler = StandardScaler()

    def fit(self, X, y, outer_folds=5, inner_folds=3, 
            epochs=10000, random_state=42, verbose=1):
        """
        Perform k-fold cross-validation with fresh hyperparameter tuning on each fold.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        outer_folds : int, default=5
            Number of folds for cross-validation.
        inner_folds : int, default=3
            Number of folds for hyperparameter tuning in each outer fold.
        epochs : int, default=10000
            Maximum number of epochs for training.
        random_state : int, default=42
            Random state for reproducibility.
        verbose : int, default=0
            Verbosity level (0: no output, 1: summary, 2: detailed)
            
        Returns
        -------
        list
            List containing test scores for each fold
        """
        # Initialize the cross-validation folds
        kf = KFold(n_splits=outer_folds, shuffle=True, random_state=random_state)

        # Generate random indices
        random_idxs = np.arange(len(X))
        np.random.seed(random_state)
        np.random.shuffle(random_idxs)
        
        # Scale the full dataset
        X = self.Xscaler.fit_transform(X)
        y = self.yscaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        if verbose > 0:
            print(f"Performing {outer_folds}-fold cross-validation")
        
        # Reset scores and models for each fold
        self.scores = []
        self.models = []
        self.best_params_list = []

        # For each fold
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            if verbose > 0:
                print(f"\n{'='*50}")
                print(f"Fold {fold+1}/{outer_folds}")
                print(f"{'='*50}")
            
            # Split data based on the indices
            X_train, y_train = X[random_idxs[train_idx]], y[random_idxs[train_idx]]
            X_test, y_test =   X[random_idxs[test_idx]],  y[random_idxs[test_idx]]

            if verbose > 0:
                print(f"Using {len(X_train)} samples for training")
                print(f"Using {len(X_test)} samples for testing")

            # Define the parameter grid for hyperparameter tuning
            param_grid = {
                'hidden_layer_sizes': [(5), (10), (10, 5), (10, 10)],
                'activation': ['tanh', 'relu', 'logistic'],
                'alpha': [0.0001, 0.001, 0.01],
                'solver': ['sgd', 'adam'],
            }

            tuner = GridSearchCV(
                MLPRegressor(
                    early_stopping=True,
                    validation_fraction=0.1,
                    max_iter=epochs,
                    random_state=random_state
                ),
                param_grid,
                n_jobs=-1,
                cv=inner_folds,
                refit=True,
                scoring='r2',
            )

            tuner.fit(X_train, y_train)

            # Evaluate on the test set
            score = tuner.score(X_test, y_test)
            self.scores.append(score)
            self.models.append(tuner.best_estimator_)
            self.best_params_list.append(tuner.best_params_)
            
            if verbose > 0:
                print(f"Best parameters: {tuner.best_params_}")
                print(f"Fold score (R²): {score:.4f}")
        
        # Report average score across all outer folds
        if verbose > 0:
            print(f"\nAverage score across {outer_folds} folds: {np.mean(self.scores):.4f} ± {np.std(self.scores):.4f}")
            print(f"Standard error: {np.std(self.scores)/np.sqrt(outer_folds):.4f}")
        
        return self.scores

    def predict(self, X):
        """
        Predict using the trained models.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data to predict.
            
        Returns
        -------
        array-like, shape (n_samples,)
            Predicted values.
        """
        # Scale the input data
        X = self.Xscaler.transform(X)
        
        # Average predictions from all models (ensemble approach)
        predictions = np.mean([model.predict(X) for model in self.models], axis=0)
        
        # Inverse transform to get back to the original scale
        return self.yscaler.inverse_transform(predictions.reshape(-1, 1)).ravel()

        
        
        
class WarmStartNNRegressor(Model):
    """
    A wrapper for the MLPRegressor from sklearn with hyperparameter tuning using GridSearchCV.
    """

    def __init__(self, epochs=1000, random_state=42, params=None):
        """
        Initialize the NNRegressor with the given parameters.
        """
        self.Xscaler = StandardScaler()
        self.yscaler = StandardScaler()
        self.epochs = epochs
        self.random_state = random_state

        # Use provided params or default values
        if params is None:
            self.params = {
            'hidden_layer_sizes': (10, 10),
            'activation': 'relu',
            'alpha': 0.001,
            'solver': 'adam',
            }
        else:
            self.params = params
        
        # Initialize the model with the default parameters
        self.model = MLPRegressor(
            **self.params,
            max_iter=epochs,
            warm_start=True,
            early_stopping=True,
            n_iter_no_change=50,
            validation_fraction=0.1,
            random_state=random_state
        )


    def fit(self, X, y):
        """
        Perform k-fold cross-validation with fresh hyperparameter tuning on each fold.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.
        y : array-like, shape (n_samples,)
            Target values.
        outer_folds : int, default=5
            Number of folds for cross-validation.
        inner_folds : int, default=3
            Number of folds for hyperparameter tuning in each outer fold.
        epochs : int, default=10000
            Maximum number of epochs for training.
        random_state : int, default=42
            Random state for reproducibility.
        verbose : int, default=0
            Verbosity level (0: no output, 1: summary, 2: detailed)
            
        Returns
        -------
        list
            List containing test scores for each fold
        """
        # Scale the full dataset
        X = self.Xscaler.fit_transform(X)
        y = self.yscaler.fit_transform(y.reshape(-1, 1)).ravel()
        
        # Generate training and testing indices
        idxs = np.arange(len(X))
        np.random.seed(self.random_state)
        np.random.shuffle(idxs)
        Ntrain = int(0.8 * len(X))
        train_idx = idxs[:Ntrain]
        test_idx = idxs[Ntrain:]
        
        # Split data based on the indices
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
            
        # Execute model fitting
        self.model.fit(X_train, y_train)

        # Evaluate on the test set
        score = self.model.score(X_test, y_test)
        print(f"Fold score (R²): {score:.4f}")
        
        return 0

    def predict(self, X):
        """
        Predict using the trained models.
        
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Input data to predict.
            
        Returns
        -------
        array-like, shape (n_samples,)
            Predicted values.
        """
        # Scale the input data
        X = self.Xscaler.transform(X)
        
        # Average predictions from all models (ensemble approach)
        predictions = self.model.predict(X)
        
        # Inverse transform to get back to the original scale
        return self.yscaler.inverse_transform(predictions.reshape(-1, 1)).ravel()
