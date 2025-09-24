from abc import ABC, abstractmethod



class Model(ABC):
    '''Abstract class for a model used inside the RL agent
    '''
    
    @abstractmethod
    def fit(self, X, y):
        '''Fit the model to the data
        Parameters
        ----------
        X : array_like
            Features to fit the model to
        y : array_like
            Target variable to fit the model to
        '''
        pass
    
    
    @abstractmethod
    def predict(self, X):
        '''Predict using the model
        Parameters
        ----------
        X : array_like
            Features to predict on
        Returns
        -------
        array_like
            Predicted values
        '''
        pass