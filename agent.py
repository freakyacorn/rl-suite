import numpy as np
from ambition.random import create_ascii_histogram

class Agent:
    '''Reinforcement learning agent class
    
    Attributes
    ----------
    policy : object
        Policy for selecting actions
    memory : dict
        Dictionary containing memory as NumPy arrays
    '''
    
    def __init__(self, brain, policy, discount_rate=0.97) -> None:
        '''Constructor
        
        Parameters
        ----------
        policy : object
            Policy for selection actions
        brain : object
            Function approximator that learns state-action-reward space
        '''
        self.policy = policy
        self.brain = brain
        self.discount_rate = discount_rate
        
        # Initialize with None - we'll create empty arrays on first data
        self.memory = {'old_states': None, 'actions': None, 
                      'rewards': None, 'new_states': None}
        self.is_initialized = False
        self.training_round = 0

    
    def train(self, environment):
        '''Train agent given current memory

        possible_actions : object
            Possible actions for the new state
        '''
        if not self.is_initialized or len(self.memory['old_states']) == 0:
            print("No experiences to train on")
            return
            
        # Use numpy arrays directly
        old_states = self.memory['old_states']
        actions = self.memory['actions']
        rewards = self.memory['rewards']
        new_states = self.memory['new_states']
        
        # Create training features by combining old states and actions
        Xtrain = np.hstack([old_states, actions])
        
        # Calculate greedy reward in new states and discount them
        if self.training_round == 0:
            # For the first training round, there is no previous experience
            # so we cannot calculate discounted future rewards
            discounted_reward = np.zeros_like(rewards)
        else:
            # After the first training round, we can calculate discounted future rewards
            greedy_reward = self.__greedy_reward(new_states, environment.get_possible_actions(states=new_states))
            discounted_reward = self.discount_rate * greedy_reward
        
        # ytrain is reward plus discounted future reward
        ytrain = discounted_reward + rewards
        print('ytrain')
        # Print ytrain without scientific notation
        print(create_ascii_histogram(ytrain))
        
        # Fit the model to current data
        print('Beginning training')
        self.brain.fit(Xtrain, ytrain)
        self.training_round += 1
        print('Training complete')

    
    def choose_actions(self, states, possible_actions):
        '''Choose action given a state
        
        Parameters
        ----------
        states : array_like
            Current states, must be a 2D array where each row is a state

        Returns
        -------
        actions : numpy.ndarray
            Actions to execute, 2D array where each row is an action
        '''
        # Convert to numpy array if not already
        states = np.asarray(states)
        
        # Ensure states is a 2D array
        if states.ndim != 2:
            raise ValueError(
                "States must be a 2D array where each row is a state. "
                "For a single state, use [[state_values]] format. "
                f"Got shape {states.shape} instead."
            )
            
        if self.training_round == 0:
            # Random actions for first round
            idxs = [np.random.choice(len(row)) for row in possible_actions]
            actions = [row[idx] for row, idx in zip(possible_actions, idxs)]
        else:
            # Use policy to choose actions
            actions = self.policy.choose(states, possible_actions, self.brain, self.training_round)
            
        return actions

    
    def record(self, old_states, actions, rewards, new_states):
        '''Record new training data
        
        Parameters
        ----------
        old_states : array_like
            Must be a 2D array where each row is a state vector.
            Example: [[1, 2, 3], [4, 5, 6]] for two states
        
        actions : array_like
            Must be a 2D array where each row is an action vector.
            Example: [[0], [1]] for two single-dimensional actions
        
        rewards : array_like
            Rewards received for each state-action pair
        
        new_states : array_like
            Must be a 2D array where each row is the resulting state vector.
            Example: [[2, 3, 4], [5, 6, 7]] for two states
        '''
        # Convert to numpy arrays for validation
        old_states = np.asarray(old_states)
        actions = np.asarray(actions)
        rewards = np.asarray(rewards)
        new_states = np.asarray(new_states)
        
        # Validate inputs in a loop
        for name, array, ndim in [
            ("old_states", old_states, 2),
            ("new_states", new_states, 2),
            ("actions", actions, 2),
        ]:
            if array.ndim != ndim:
                raise ValueError(
                    f"{name} must be a {ndim}D array where each row is a {name[:-1]}. "
                    f"Got shape {array.shape} instead."
                )
        
        # Initialize memory arrays if this is the first experience
        if not self.is_initialized:
            # Get dimensions from first data batch
            state_dim = old_states.shape[1]
            action_dim = actions.shape[1]
            
            # Create empty arrays with correct dimensions
            self.memory['old_states'] = np.empty((0, state_dim), dtype=np.float32)
            self.memory['actions'] = np.empty((0, action_dim), dtype=np.float32)
            self.memory['rewards'] = np.empty(0, dtype=np.float32)
            self.memory['new_states'] = np.empty((0, state_dim), dtype=np.float32)
            self.is_initialized = True
        
        # Append new data to memory using vstack/concatenate
        self.memory['old_states'] = np.vstack((self.memory['old_states'], old_states))
        self.memory['actions'] = np.vstack((self.memory['actions'], actions))
        self.memory['rewards'] = np.concatenate((self.memory['rewards'], rewards))
        self.memory['new_states'] = np.vstack((self.memory['new_states'], new_states))
    

    def __greedy_reward(self, states, possible_actions):
        '''Gets the expected reward from greedy selection
        
        Parameters
        ----------
        states : array_like
            Must be a 2D array where each row is a state vector.
            Single states must be explicitly wrapped in an outer array.
            Examples:
            - [[1, 2, 3]] for a single state with 3 dimensions
            - [[1]] for a single state with 1 dimension
            
        Returns
        -------
        rewards : ndarray
            Maximum expected reward for each state
        '''
        # Convert input to numpy array
        states = np.asarray(states)
        
        # Ensure states is a 2D array (array of arrays)
        if states.ndim != 2:
            raise ValueError(
                "States must be a 2D array where each row is a state. "
                "For a single state, use [[state_values]] format. "
                f"Got shape {states.shape} instead."
            )
        
        # Initialize rewards array
        rewards = np.zeros(len(states))
        
        # Process each state
        for i, state in enumerate(states):
            # Get possible actions for current state
            actions = possible_actions[i]
                
            # Build state-action pairs for prediction
            state_action_pairs = []
            
            for action in actions:
                # Convert to numpy array if not already
                action = np.asarray(action)
                
                # Ensure action is an array, not a scalar
                if action.ndim == 0:
                    raise ValueError(
                        f"Action must be an array, not a scalar value. "
                        f"For single-value actions, use [value] format. "
                        f"Got scalar {action} instead."
                    )
                    
                # Concatenate state and action
                state_action_pair = np.concatenate((state, action))
                state_action_pairs.append(state_action_pair)
                
            # Convert to array for batch prediction
            X = np.array(state_action_pairs)
            
            # Predict rewards for all actions
            expected_rewards = self.brain.predict(X)
            
            # Store maximum expected reward
            rewards[i] = np.max(expected_rewards)
        
        return rewards

    
    def get_state_dict(self):
        """Return the dictionary of state variables"""
        state_dict = {
            'memory': self.memory,
            'is_initialized': self.is_initialized,
            'training_round': self.training_round,
            'policy': self.policy,
            'brain': self.brain,
            'discount_rate': self.discount_rate
        }

        return state_dict

        
    def load_state_dict(self, state_dict):
        """Load the state dictionary into the agent"""
        self.memory = state_dict['memory']
        self.is_initialized = state_dict['is_initialized']
        self.training_round = state_dict['training_round']
        self.policy = state_dict['policy']
        self.brain = state_dict['brain']    
        self.discount_rate = state_dict['discount_rate']
        print("Agent state loaded successfully")

        