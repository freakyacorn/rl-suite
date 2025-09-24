import numpy as np
from .policy import Policy

class EpsilonGreedy(Policy):
    '''Class to pick an action according to an epsilon greedy policy
    
    Attributes
    ----------
    epsilon : float
        Epsilon rate for the epsilon greedy policy. This is the rate at
        which random actions are chosen over following the optimal policu
    '''
    
    def __init__(self, epsilon, anneal=None):
        '''Constructor
        
        Parameters
        ----------
        epsilon : float
            Epsilon rate for the epsilon greedy policy. This is the rate at
            which random actions are chosen over following the optimal policy
        '''

        self.epsilon = epsilon
        self.anneal = anneal
        

    def choose(self, states, actions, model, training_round):
        '''Choose an action given a list of actions, state, and model
        Parameters
        ----------
        actions : array_like
            List of actions that can be chosen
        state : float/array_like
            Numerical representation of current state
        model : tensorflow model object
            Model being used to represent Q-value function
        Returns
        -------
        action : array_like
            Action(s) to be taken
        '''
        # Convert to numpy arrays
        states = np.array(states)

        # Compute current epsilon, applying annealing if specified
        current_epsilon = self.epsilon * self.anneal ** training_round if self.anneal is not None else self.epsilon
        
        # Create state-action pairs
        chosen_actions = [0] * len(states)
        for i, state in enumerate(states):
            # Get actions for this state
            state_actions = np.array(actions[i])
            
            # Skip if no actions for this state
            if len(state_actions) == 0:
                raise ValueError("No actions available for this state")
                
            # Repeat the state for each action
            repeated_state = np.repeat(state.reshape(1, -1), len(state_actions), axis=0)
            
            # Convert actions to correct shape (make column vectors if they're 1D)
            if state_actions.ndim == 1:
                state_actions = state_actions.reshape(-1, 1)
            
            # Combine states with their actions
            state_action_pairs = np.hstack([repeated_state, state_actions])
            
            # Add to our collection
            expected_rewards = model.predict(state_action_pairs)
            
            if np.random.rand() < current_epsilon:
                # Choose a random action
                idx = np.random.randint(0, len(state_actions))
            else:
                # Choose the action with the maximum expected reward
                idx = np.argmax(expected_rewards)
            # print(f"Chosen action index: {idx}")

            chosen_actions[i] = state_actions[idx].tolist()
        
        return chosen_actions