from ambition.rl.src.environments.pna_folding import IncrementalPNAFolding
from ambition.rl.src.policies.epsilon_greedy import EpsilonGreedy
from ambition.rl.src.models.sklearn_NN import NNRegressor, WarmStartNNRegressor
from ambition.rl.src.models.torch_NN import NeuralNetwork
from ambition.rl.src.agent import Agent
from ambition.rl.src.sandbox import Sandbox
from ambition.code_snapshot import code_snapshot
import numpy as np
import sys


def main(args):
    # Capture a code snapshot
    code_snapshot()

    # Get the argument
    target = args[0]

    # Initialize the environment
    environment = IncrementalPNAFolding(
        "/home/wo6860/my_package/ambition/rl/src/data_files/pna_folding/toy_sequence.txt",
        target,
        # [[x] for x in sorted(list(np.linspace(-5, 5, 41)) + [-0.01, 0.01])],
        [[x] for x in sorted(list(np.linspace(0, 12, 121)))],
        n_simulations=112,
        sim_steps_per_round=1000000,
    )

    # Initialize the policy
    policy = EpsilonGreedy(epsilon=0.05, anneal=1.00)

    # Initialize the model
    model_type = args[1] if len(args) > 1 else "sklearn"
    if model_type == "sklearn":
        model = NNRegressor()
    elif model_type == "warm_start":
        model = WarmStartNNRegressor()
    else:  # Default to pytorch based model
        model = NeuralNetwork(8)

    # Initialize the agent
    agent = Agent(brain=model, policy=policy, discount_rate=0.9)

    # Initialize the sandbox
    sandbox = Sandbox(environment, agent, rounds=199, visualize=True)

    # Execute the training rounds, resume from checkpoint if provided
    checkpoint_path = "checkpoints/checkpoint.pkl" if "resume" in args else None
    sandbox.run(checkpoint_path=checkpoint_path)


if __name__ == "__main__":
    main(sys.argv[1:])
