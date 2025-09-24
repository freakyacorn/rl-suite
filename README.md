# Reinforcement learning suite

## Overview

Some example code that I wrote to show people. 
It is an implementation of reinforcement learning (RL) control over molecular dynamics simulations.
Specifically, the package is based off of a deep Q-learning approach to RL.
There are the following main classes in the package.

1. `Sandbox` class. This is the main class for the package. It takes as inputs an environment and an agent. The main method is the run() method, which can be executed from a previous run's round checkpoint.
2. `Environment` class. This is an abstract class that contains all of the information specific to the kind of simulation over which you are trying to exert RL control. The specific inputs to the environment may vary, but will almost certainly include some notion of a "target" and a list of "actions". You can check out the various environments I have created in the `environments` directory.
3. `Agent` class. The agent is the class that chooses from the list of actions in the environment based on the current state of the environment. It needs a model and policy as its inputs.
4. `Model` class. This class is a wrapper class for any surrogate model that can be used to representing the value function. Right now there are options to use a PyTorch based NN and a scikit-learn based NN with hyperparameter tuning.
5. `Policy` class. This class contains the "policy" used in action selection by the agent. Right now I only have one implementation of a policy (epsilon-greedy)

The script `example.py` outlines what a concerted usage of these objects might look like.
After imports and arguments, an environment is initialized (in this case a folding peptide nucleic acid system). 
Then, before initializing the agent, I initialize the policy and model. 
These are fed as inputs when initializing the agent (next step). 
Now that I have an environment and an agent, I input these into the Sandbox and call its run method. 

In this example, I am running on a high-performance-computing cluster with multi-core nodes. 
As a result, I run many (in this case 112) simulations in parallel, each of which the agent can "explore" independently. 
The biggest bottleneck in this setup with respect to computational efficiency is the discrepancy between the embarrassingly parallel nature of the simulations and the training of the model within the agent. 
While model training is fast compared to the simulations, it leaves many cpus idle. 
Currently, my workaround is to use fewer cpu cores (longer simulation times because not all run in parallel) to reduce idle cores during model training. 
If anyone has any good ideas for this, please let me know.
