from abc import ABC, abstractmethod


class Environment(ABC):
    """Abstract base class for environments in reinforcement learning"""

    @abstractmethod
    def calculate_rewards(self):
        """Calculate and return rewards based on current state"""
        pass

    @abstractmethod
    def calculate_states(self):
        """Calculate and return current states"""
        pass

    @abstractmethod
    def initialize_simulations(self):
        """Initialize all simulations and return initial states"""
        pass

    @abstractmethod
    def equilibrate_simulations(self):
        """Run equilibration steps"""
        pass

    @abstractmethod
    def modify_simulations(self, actions):
        """Apply actions to simulations"""
        pass

    @abstractmethod
    def calculate_additional_metrics(self):
        """Calculate any additional metrics of interest"""
        pass

    @abstractmethod
    def get_state_dict(self):
        """Return the dictionary of state variables"""
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        """Load the state dictionary into the environment"""
        pass

    @abstractmethod
    def get_possible_actions(self, states=None):
        """Return the list of possible actions"""
        pass
