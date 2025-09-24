import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import time
from ambition.random import seconds_to_hms
from ambition.random import create_ascii_histogram


class Sandbox:
    def __init__(
        self,
        environment,
        agent,
        rounds=199,
        save_dir="checkpoints",
        visualize=False,
        vis_dir="visualizations",
    ):
        self.environment = environment
        self.agent = agent
        self.rounds = rounds
        self.current_round = 0
        self.save_dir = save_dir
        self.visualize = visualize
        self.vis_dir = vis_dir
        self.results_history = {
            "old_states": [],
            "actions": [],
            "rewards": [],
            "new_states": [],
        }  # For tracking metrics

        self.training_time = 0
        self.simulation_time = 0

        # Create save directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        if not os.path.exists(vis_dir) and visualize:
            os.makedirs(vis_dir)

    def initialize(self, checkpoint_path=None):
        """Initialize training from scratch or resume from checkpoint"""
        if checkpoint_path:
            self._load_checkpoint(checkpoint_path)
            self.rounds += self.current_round  # Add rounds if resuming from checkpoint
        else:
            # Initialize environment and get initial states
            self.environment.initialize_simulations()
            s0 = time.perf_counter()
            print("Performing initial equilibration...")
            self.environment.equilibrate_simulations()
            self.simulation_time += time.perf_counter() - s0
            self.states = self.environment.calculate_states()

    def execute_round(self):
        """Execute a single training round"""
        # Agent chooses actions based on current states
        actions = self.agent.choose_actions(
            self.states, self.environment.get_possible_actions()
        )

        # Apply actions to simulation scripts
        self.environment.modify_simulations(actions)

        # Equilibrate simulations
        s0 = time.perf_counter()
        print("Equilibrating simulations...")
        self.environment.equilibrate_simulations()
        print("Done!\n")
        self.simulation_time += time.perf_counter() - s0

        # Get new states and rewards
        new_states = self.environment.calculate_states()
        rewards = self.environment.calculate_rewards()

        print("!" * 50)
        print("old states")
        print(create_ascii_histogram([s[0] for s in self.states]))
        print("")
        print("actions")
        print(create_ascii_histogram([a[0] for a in actions]))
        print("")
        print("rewards")
        print(create_ascii_histogram([r for r in rewards]))
        print("")
        print("new states")
        print(create_ascii_histogram([ns[0] for ns in new_states]))
        print("")

        # Agent records experience
        self.agent.record(self.states, actions, rewards, new_states)

        # Update current states and results history to track progress
        self._update_results(
            old_states=self.states,
            actions=actions,
            rewards=rewards,
            new_states=new_states,
        )
        self.states = new_states

        # Train agent after collecting data for this round
        # Requires environment to calculate possible actions for new states
        s0 = time.perf_counter()
        self.agent.train(
            self.environment,
        )
        self.training_time += time.perf_counter() - s0

        # Update round counter
        self.current_round += 1

        # Save checkpoint each round
        self._save_checkpoint()

    def run(self, checkpoint_path=None):
        """Run all training rounds"""
        t0 = time.perf_counter()
        self.initialize(checkpoint_path=checkpoint_path)
        for i in range(self.current_round, self.rounds):
            print(f"Beginning round {i} of training")
            print("--------------------------------------")
            self.execute_round()
            # Log progress
            if self.current_round % 5 == 0:
                print(f"Completed round {self.current_round}/{self.rounds}\n\n")

        print("\n\nSimulation complete!\n")
        t1 = time.perf_counter()
        hours, minutes, seconds = seconds_to_hms(t1 - t0)
        print(f"  Total time: {hours}h {minutes}m {seconds}s")
        print(f"  Training load: {(self.training_time / (t1 - t0)) * 100:.2f}%")
        print(f"  Simulation load: {(self.simulation_time / (t1 - t0)) * 100:.2f}%")

    def _update_results(self, old_states, actions, rewards, new_states):
        """Update results history with metrics from current round"""
        self.results_history["old_states"].append(old_states)
        self.results_history["actions"].append(actions)
        self.results_history["rewards"].append(rewards)
        self.results_history["new_states"].append(new_states)

        # Record any other environment-specific metrics
        metrics = self.environment.calculate_additional_metrics()
        if metrics is not None:
            for key, value in metrics.items():
                if key not in self.results_history:
                    self.results_history[key] = []
                self.results_history[key].append(value)

        if self.visualize and self.current_round:
            # Plot the average of each thing in results history vs round
            rounds = list(range(0, self.current_round + 1))
            for key, values in self.results_history.items():
                values_array = np.squeeze(values)

                # If values_array is 2D, plot the distribution
                if values_array.ndim == 2:
                    lower_quartile = np.percentile(values_array, 25, axis=-1)
                    median = np.percentile(values_array, 50, axis=-1)
                    upper_quartile = np.percentile(values_array, 75, axis=-1)

                    plt.figure(figsize=(8, 5))
                    plt.plot(rounds, median, label=f"Median {key}")
                    plt.fill_between(
                        rounds,
                        lower_quartile,
                        upper_quartile,
                        alpha=0.3,
                        label=f"25-75% {key}",
                    )
                    plt.xlabel("Round")
                    plt.ylabel(f"{key}")
                    plt.xlim(0, self.rounds + 1)
                    plt.title(f"{key} vs Round")
                    plt.legend()

                    # Save the plot to vis_dir
                    plot_path = os.path.join(self.vis_dir, f"{key}_vs_round.png")
                    plt.savefig(plot_path)
                    plt.close()

                # If values_array is 1D, plot the values
                if values_array.ndim == 1:
                    plt.figure(figsize=(8, 5))
                    plt.plot(rounds, values_array, label=f"{key} per round")
                    plt.xlabel("Round")
                    plt.ylabel(f"{key}")
                    plt.xlim(0, self.rounds + 1)
                    plt.title(f"{key} vs Round")
                    plt.legend()

                    # Save the plot to vis_dir
                    plot_path = os.path.join(self.vis_dir, f"{key}_vs_round.png")
                    plt.savefig(plot_path)
                    plt.close()

    def _save_checkpoint(self):
        """Save current state of training"""
        checkpoint = {
            "current_round": self.current_round,
            "results_history": self.results_history,
            "agent_state": self.agent.get_state_dict(),  # Agent would need to implement this
            "environment_state": self.environment.get_state_dict(),  # Environment would need to implement this
        }

        # Save the whole checkpoint state
        filename = os.path.join(self.save_dir, f"checkpoint.pkl")
        with open(filename, "wb") as f:
            pickle.dump(checkpoint, f)

        # Also save a separate file with just the results
        filename = os.path.join(self.save_dir, f"results.pkl")
        with open(filename, "wb") as f:
            pickle.dump(self.results_history, f)

        print(f"Saved checkpoint after round {self.current_round - 1}\n")

    def _load_checkpoint(self, checkpoint_path):
        """Load training state from checkpoint"""
        with open(checkpoint_path, "rb") as f:
            checkpoint = pickle.load(f)

        self.current_round = checkpoint["current_round"]
        self.results_history = checkpoint["results_history"]

        # Load agent and environment states
        self.agent.load_state_dict(checkpoint["agent_state"])
        self.environment.load_state_dict(checkpoint["environment_state"])

        # Get current states from environment
        self.states = self.environment.calculate_states()

        print(f"Resumed training from round {self.current_round}")
        print(f"Running an additional {self.rounds} rounds\n")
