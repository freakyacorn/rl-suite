from abc import ABC, abstractmethod


class Policy(ABC):
    '''Base class for all policies
    '''
        
    @abstractmethod
    def choose(self, actions, state, model):
        pass