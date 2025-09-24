from .environment import Environment
from .utils import write_universe
from abc import ABC
import os
import subprocess
import numpy as np
import MDAnalysis as mda
from joblib import Parallel, delayed
from functools import partial


class PNAFolding(Environment, ABC):
    """Lennard-Jones polymer of degree polymerization 30
    with agent control over temperature"""

    def __init__(
        self,
        sequence_file,
        target,
        action_list,
        n_simulations=112,
        sim_steps_per_round=1000000,
        dump_style="dcd",
        simulation_dir=os.path.join(os.getcwd(), "simulations"),
    ):
        """
        Initialize the Lennard-Jones temperature environment.

        Parameters
        ----------
        target : float or list
            Target temperature(s) for the environment.
        n_simulations : int
            Number of simulations to run.
        sim_steps_per_round : int
            Number of simulation steps per round.
        """
        super().__init__()
        self.sequence_file = sequence_file
        if target not in ("tripod", "hairpin"):
            raise ValueError("Target must be either 'tripod' or 'hairpin'.")
        self.target = target
        if isinstance(action_list, list) and all(
            isinstance(sublist, list) for sublist in action_list
        ):
            self.action_list = action_list
        else:
            raise ValueError("action_list must be a 2D array or a list of lists.")
        self.n_simulations = n_simulations
        self.sim_steps_per_round = sim_steps_per_round
        self.dump_style = dump_style
        self.simulation_dir = simulation_dir
        self.states = []
        self.temperatures = []
        self.seeds = []

    def initialize_simulations(self):
        """Initialize all simulations and return initial states"""
        commands = []
        paths = []
        self.temperatures = list(np.random.uniform(3.0, 7.0, self.n_simulations))
        self.seeds = np.random.randint(0, 1000000, self.n_simulations)
        for i in range(self.n_simulations):
            poly_dir = os.path.join(self.simulation_dir, f"poly{i}")
            paths.append(poly_dir)
            commands.append(
                [
                    "python",
                    "/home/wo6860/projects/pna/sum2024_project/generate.py",
                    self.sequence_file,
                    "--output",
                    poly_dir,
                    "--temperature",
                    str(self.temperatures[i]),
                    "--nsteps",
                    str(self.sim_steps_per_round),
                    "--seed",
                    str(self.seeds[i]),
                    "--coords_freq",
                    "10000",
                    "--dump_style",
                    self.dump_style,
                    "--database",
                    "db_high_tm_delta.py",
                ]
            )

        def begin_wrapper(command):
            subprocess.run(command)
            tiger_submit_file = os.path.join(command[4], "tiger.submit")
            if os.path.exists(tiger_submit_file):
                os.remove(tiger_submit_file)
            if os.path.exists(os.path.join(command[4], "sys.data")):
                subprocess.run(
                    [
                        "cp",
                        os.path.join(command[4], "sys.data"),
                        os.path.join(command[4], "start_config.data"),
                    ]
                )
            return 0

        Parallel(n_jobs=-1)(delayed(begin_wrapper)(command) for command in commands)

        write_universe(self.simulation_dir, paths, "in.pna")

    def equilibrate_simulations(self):
        """Run equilibration"""
        run_dir = os.path.join(self.simulation_dir, "in.universe")
        available_cpus = int(os.environ.get("SLURM_NTASKS"))
        command = f"srun /home/wo6860/software/lammps-stable_29Aug2024/build_pna/lmp_intel -in {run_dir} -partition {available_cpus}x1 -plog none -pscreen none"
        subprocess.run(command.split(), cwd=self.simulation_dir)

    def modify_simulations(self, actions):
        """Apply actions to simulations"""
        # Change the temperatures based on actions
        # self.temperatures = [t + action[0] for t, action in zip(self.temperatures, actions)]
        self.temperatures = [
            action[0] for action in actions
        ]  # Assuming actions is a list of lists with one action each

        # Modify lammps scripts to reflect new temperatures
        for i in range(self.n_simulations):
            with open(f"{self.simulation_dir}/poly{i}/in.pna", "r") as file:
                lines = file.readlines()

            with open(f"{self.simulation_dir}/poly{i}/in.pna", "w") as file:
                for line in lines:
                    if line.startswith("variable            T0"):
                        file.write(
                            f"variable            T0             index    {self.temperatures[i]}\n"
                        )
                    else:
                        file.write(line)

    def get_possible_actions(self, states=None):
        """
        Return a list of numpy arrays where each array represents the possible actions
        for a given simulation based on its current temperature.
        """
        # State is temperature for each simulation, limit actions based on temperature
        if states is None:
            print("No states provided, using current temperatures.")
            temperatures = self.temperatures
        else:
            print("Using provided states for action calculation.")
            temperatures = [s[0] for s in states]

        possible_actions = [0] * len(temperatures)
        for i, T in enumerate(temperatures):
            # valid_actions = [action for action in self.action_list if 0.0 < T + action]
            valid_actions = self.action_list
            possible_actions[i] = valid_actions
        return possible_actions

    def load_state_dict(self, state_dict):
        """Load the state dictionary into the environment"""
        self.sequence_file = state_dict["sequence_file"]
        self.target = state_dict["target"]
        self.action_list = state_dict["action_list"]
        self.n_simulations = state_dict["n_simulations"]
        self.sim_steps_per_round = state_dict["sim_steps_per_round"]
        self.dump_style = state_dict["dump_style"]
        self.simulation_dir = state_dict["simulation_dir"]
        self.states = state_dict["states"]
        self.temperatures = state_dict["temperatures"]
        self.seeds = state_dict["seeds"]
        print("State dictionary loaded successfully.")

        return 0

    def get_state_dict(self):
        """Return the dictionary of state variables"""
        state_dict = {
            "sequence_file": self.sequence_file,
            "target": self.target,
            "action_list": self.action_list,
            "n_simulations": self.n_simulations,
            "sim_steps_per_round": self.sim_steps_per_round,
            "dump_style": self.dump_style,
            "simulation_dir": self.simulation_dir,
            "states": self.states,
            "temperatures": self.temperatures,
            "seeds": self.seeds,
        }

        return state_dict


class AllOrNothingPNAFolding(PNAFolding):
    """PNA Folding environment with all-or-nothing reward"""

    def calculate_states(self):
        """Return current states. Temperature must be first item"""
        commands = []
        for i in range(self.n_simulations):
            structure_path = os.path.join(
                self.simulation_dir, f"poly{i}/start_config.data"
            )
            trajectory_path = os.path.join(
                self.simulation_dir, f"poly{i}/coords.{self.dump_style}"
            )
            commands.append((structure_path, trajectory_path))

        func = partial(
            pna_cv,
            fraction=0.25,
            mode=self.target,
            dump_style=self.dump_style,
            average=True,
            classify=True,
            cutoff=5.0,
        )

        half_available_cpus = int(os.environ.get("SLURM_NTASKS", 1)) // 2
        states = Parallel(n_jobs=half_available_cpus, timeout=60)(
            delayed(func)(*command) for command in commands
        )
        self.states = [[t] + [s] for t, s in zip(self.temperatures, states)]
        return self.states

    def calculate_rewards(self):
        """Calculate and return rewards based on current state"""
        temperature_penalty = [
            -1 * (max(0, t - 10) + max(0, 2.5 - t)) for t in self.temperatures
        ]
        return [
            s[1] + temperature_penalty[i] for i, s in enumerate(self.states)
        ]  # Assuming states are structured as [[state_value], ...]

    def calculate_additional_metrics(self):
        """Any other calculations of interest. Return a dictionary
        with keys as metric names and values as the calculated metric."""
        additional_metrics = {}
        additional_metrics["temperatures"] = [
            [t] for t in self.temperatures
        ]  # Wrap in list to match state structure
        additional_metrics["percent_on_target"] = [
            [sum(self.calculate_rewards()) / self.n_simulations * 100]
        ]
        additional_metrics["pna_cv"] = [
            s[1:] for s in self.states
        ]  # Extract PNA CVs from states
        return additional_metrics


class IncrementalPNAFolding(PNAFolding):
    """PNA Folding environment with incremental reward"""

    def calculate_states(self):
        """Return current states. Temperature must be first item"""
        commands = []
        for i in range(self.n_simulations):
            structure_path = os.path.join(
                self.simulation_dir, f"poly{i}/start_config.data"
            )
            trajectory_path = os.path.join(
                self.simulation_dir, f"poly{i}/coords.{self.dump_style}"
            )
            commands.append((structure_path, trajectory_path))

        func = partial(
            pna_cv,
            fraction=0.25,
            mode=self.target,
            dump_style=self.dump_style,
            average=True,
            classify=False,
            cutoff=5.0,
        )

        half_available_cpus = int(os.environ.get("SLURM_NTASKS", 1)) // 2
        states = Parallel(n_jobs=half_available_cpus, timeout=60)(
            delayed(func)(*command) for command in commands
        )
        self.states = [[t] + list(s) for t, s in zip(self.temperatures, states)]
        return self.states

    def calculate_rewards(self):
        """Calculate and return rewards based on current state"""
        # Extract distances from states, assuming states are structured as [[temperature], [dist_12, dist_34, ...], ...]
        pna_cv = [s[1:] for s in self.states]

        # See if the PNA is in the target state
        classified_states = [
            classify_pna(state, self.target, cutoff=5.0) for state in pna_cv
        ]

        # Calculate rewards based on distances and classification
        rewards = [0] * self.n_simulations
        for i, (state, classification) in enumerate(zip(pna_cv, classified_states)):
            # For a given target, the goal is to minimize certain distances and maximize others
            # For example, in a tripod state, we want distances 12 and 34 to be small, while 14 and 23 should be large
            if self.target == "tripod":
                rewards[i] += (
                    -1 * state[0]
                    + -1 * state[1]
                    + state[2]
                    + state[3]
                    + state[4]
                    + state[5]
                )
            elif self.target == "hairpin":
                rewards[i] += (
                    state[0]
                    + state[1]
                    + -1 * state[2]
                    + -1 * state[3]
                    + state[4]
                    + state[5]
                )

            # # If the PNA is in the target state, add a bonus
            # rewards[i] += 100 * classification

            # Also penalize for raising or lowering the temperature too much
            rewards[i] += -100 * (
                max(0, self.temperatures[i] - 10) + max(0, 2.5 - self.temperatures[i])
            )

        return rewards

    def calculate_additional_metrics(self):
        """Any other calculations of interest. Return a dictionary
        with keys as metric names and values as the calculated metric."""
        additional_metrics = {}
        additional_metrics["temperatures"] = [
            [t] for t in self.temperatures
        ]  # Wrap in list to match state structure
        classified_states = [
            classify_pna(state[1:], self.target, cutoff=5.0) for state in self.states
        ]
        additional_metrics["percent_on_target"] = [
            [sum(classified_states) / self.n_simulations * 100]
        ]
        return additional_metrics


class GroupedPNAFolding(PNAFolding):
    """PNA folding environment where agent controls groups of simulations
    instead of individually controlling each simulation"""

    def __init__(
        self,
        sequence_file,
        target,
        action_list,
        n_simulations=112,
        sim_steps_per_round=1000000,
        dump_style="dcd",
        simulation_dir=os.path.join(os.getcwd(), "simulations"),
        group_size=4,
    ):
        """
        Initialize the grouped PNA folding environment.

        Parameters
        ----------
        group_size : int
            Number of simulations in each group.
        """
        super().__init__(
            sequence_file,
            target,
            action_list,
            n_simulations,
            sim_steps_per_round,
            dump_style,
            simulation_dir,
        )
        if n_simulations % group_size != 0:
            raise ValueError("n_simulations must be divisible by group_size.")
        self.group_size = group_size
        self.n_groups = n_simulations // group_size

    def initialize_simulations(self):
        """Initialize all simulations and return initial states"""
        commands = []
        paths = []
        self.temperatures = list(np.random.uniform(3.0, 7.0, self.n_groups))
        self.seeds = np.random.randint(0, 1000000, self.n_simulations)
        for i in range(self.n_simulations):
            poly_dir = os.path.join(self.simulation_dir, f"poly{i}")
            paths.append(poly_dir)
            commands.append(
                [
                    "python",
                    "/home/wo6860/projects/pna/sum2024_project/generate.py",
                    self.sequence_file,
                    "--output",
                    poly_dir,
                    "--temperature",
                    str(self.temperatures[i // self.group_size]),
                    "--nsteps",
                    str(self.sim_steps_per_round),
                    "--seed",
                    str(self.seeds[i]),
                    "--coords_freq",
                    "10000",
                    "--dump_style",
                    self.dump_style,
                    "--database",
                    "db_high_tm_delta.py",
                ]
            )

        def begin_wrapper(command):
            subprocess.run(command)
            tiger_submit_file = os.path.join(command[4], "tiger.submit")
            if os.path.exists(tiger_submit_file):
                os.remove(tiger_submit_file)
            if os.path.exists(os.path.join(command[4], "sys.data")):
                subprocess.run(
                    [
                        "cp",
                        os.path.join(command[4], "sys.data"),
                        os.path.join(command[4], "start_config.data"),
                    ]
                )
            return 0

        Parallel(n_jobs=-1)(delayed(begin_wrapper)(command) for command in commands)

        write_universe(self.simulation_dir, paths, "in.pna")

    def modify_simulations(self, actions):
        """Apply actions to simulations"""
        # Change the temperatures based on actions
        # self.temperatures = [t + action[0] for t, action in zip(self.temperatures, actions)]
        self.temperatures = [
            action[0] for action in actions
        ]  # Assuming actions is a list of lists with one action each

        # Modify lammps scripts to reflect new temperatures
        for i in range(self.n_simulations):
            with open(f"{self.simulation_dir}/poly{i}/in.pna", "r") as file:
                lines = file.readlines()

            with open(f"{self.simulation_dir}/poly{i}/in.pna", "w") as file:
                for line in lines:
                    if line.startswith("variable            T0"):
                        file.write(
                            f"variable            T0             index    {self.temperatures[i // self.group_size]}\n"
                        )
                    else:
                        file.write(line)

    def calculate_states(self):
        """Return current states. Current states is the concatenation of
        the temperature and something I haven't figured out yet"""

        # Start by calculating the pna cv for each simulation and classifying them
        commands = []
        for i in range(self.n_simulations):
            structure_path = os.path.join(
                self.simulation_dir, f"poly{i}/start_config.data"
            )
            trajectory_path = os.path.join(
                self.simulation_dir, f"poly{i}/coords.{self.dump_style}"
            )
            commands.append((structure_path, trajectory_path))

        func = partial(
            pna_cv,
            fraction=0.25,
            mode=self.target,
            dump_style=self.dump_style,
            average=True,
            classify=True,
            cutoff=5.0,
        )

        half_available_cpus = int(os.environ.get("SLURM_NTASKS", 1)) // 2
        individual_states = Parallel(n_jobs=half_available_cpus, timeout=60)(
            delayed(func)(*command) for command in commands
        )

        # Group the states by group size and compute mean for each group
        group_states = [0] * self.n_groups
        for i in range(self.n_groups):
            group_percent_on_target = (
                np.mean(
                    individual_states[i * self.group_size : (i + 1) * self.group_size]
                )
                * 100
            )
            group_states[i] = group_percent_on_target.tolist()

        # Combine temperature and group state as a list of lists
        # Each state is now [temperature, group_state]
        self.states = [
            [float(t)] + [float(s)] for t, s in zip(self.temperatures, group_states)
        ]
        return self.states

    def calculate_rewards(self):
        """Reward is the percentage of simulations in the target state
        within the group plus a temperature based penalty"""

        # Calculate temperature penalty, there are only as many
        # temperatures as there are groups
        temperature_penalty = [
            -100 * (max(0, t - 10) + max(0, 2.5 - t)) for t in self.temperatures
        ]

        # Calculate rewards for each group
        rewards = [0] * self.n_groups
        for i in range(self.n_groups):
            rewards[i] = self.states[i][1] + temperature_penalty[i]

        return rewards

    def calculate_additional_metrics(self):
        """Any other calculations of interest. Return a dictionary
        with keys as metric names and values as the calculated metric."""
        additional_metrics = {}
        additional_metrics["temperatures"] = [
            [t] for t in self.temperatures
        ]  # Wrap in list to match state structure
        additional_metrics["percent_on_target"] = [
            [self.states[i][1] for i in range(self.n_groups)]
        ]
        return additional_metrics


def pna_cv(
    structure,
    trajectory,
    fraction=0.25,
    mode="tripod",
    dump_style="dcd",
    average=True,
    classify=False,
    cutoff=5.0,
):
    """
    Calculate average group-to-group distances over a fraction of the trajectory
    using MDAnalysis, for a polymer in a LAMMPS data file and a DCD trajectory.

    Parameters
    ----------
    data_file : str
        Path to the LAMMPS data file (structure).
    dcd_file : str
        Path to the DCD trajectory file.
    fraction : float, optional
        Fraction of the trajectory to analyze from the end.
        e.g., 0.5 means only the last half of frames. Default is 1.0 (all frames).
    mode : {'tripod', 'hairpin'}, optional
        - 'tripod': Measure average distance between groups 1–2 and groups 3–4.
        - 'hairpin': Measure average distance between groups 1–4 and groups 2–3.

    Returns
    -------
    tuple of float
        If mode='tripod', returns (avg_12, avg_34).
        If mode='hairpin', returns (avg_14, avg_23).

    Notes
    -----
    - Groups (1, 2, 3, 4) are defined by atom indices:
      * Group 1: atoms 1–16   -> (index 0:16)   in 0-based indexing
      * Group 2: atoms 33–48  -> (index 32:48)
      * Group 3: atoms 129–144 -> (index 128:144)
      * Group 4: atoms 161–176 -> (index 160:176)
    - Distances are computed between centers of mass of each group.
    - Ensure MDAnalysis is installed in your environment.
    """
    if classify and not average:
        raise ValueError("'average' must be True if 'classify' is True.")

    # Load the universe
    u = mda.Universe(
        structure,
        trajectory,
        atom_style="id resid type charge x y z vx vy vz",
        topology_format="DATA",
        format=dump_style.upper(),
    )

    # Define the four groups based on their atom indices (0-based)
    group1 = u.select_atoms("index 0:16")
    group2 = u.select_atoms("index 32:48")
    group3 = u.select_atoms("index 128:144")
    group4 = u.select_atoms("index 160:176")

    # Determine which frames to analyze (only the last 'fraction' portion)
    n_frames = len(u.trajectory)
    start_frame = int(n_frames * (1.0 - fraction))

    # Lists to collect distances
    dist_12, dist_34 = [], []
    dist_14, dist_23 = [], []
    dist_13, dist_24 = [], []

    # Iterate through the selected frames
    for ts in u.trajectory[start_frame:]:
        # Account for periodic boundary conditions
        delta = ts.positions - ts.positions[0]
        pbc_vectors = np.where(
            delta > ts.dimensions[0] / 2,
            delta - ts.dimensions[0],
            np.where(delta < -ts.dimensions[0] / 2, delta + ts.dimensions[0], delta),
        )
        # Compute the centers of mass for each group
        com1 = np.sum(pbc_vectors[group1.indices].T * group1.masses, axis=1) / np.sum(
            group1.masses
        )
        com2 = np.sum(pbc_vectors[group2.indices].T * group2.masses, axis=1) / np.sum(
            group2.masses
        )
        com3 = np.sum(pbc_vectors[group3.indices].T * group3.masses, axis=1) / np.sum(
            group3.masses
        )
        com4 = np.sum(pbc_vectors[group4.indices].T * group4.masses, axis=1) / np.sum(
            group4.masses
        )

        dist_12.append(np.linalg.norm(com1 - com2))
        dist_34.append(np.linalg.norm(com3 - com4))
        dist_14.append(np.linalg.norm(com1 - com4))
        dist_23.append(np.linalg.norm(com2 - com3))
        dist_13.append(np.linalg.norm(com1 - com3))
        dist_24.append(np.linalg.norm(com2 - com4))

    if average:
        dist_12 = np.mean(dist_12)
        dist_34 = np.mean(dist_34)
        dist_14 = np.mean(dist_14)
        dist_23 = np.mean(dist_23)
        dist_13 = np.mean(dist_13)
        dist_24 = np.mean(dist_24)

    # Calculate and return the average distances
    state = (dist_12, dist_34, dist_14, dist_23, dist_13, dist_24)
    if classify:
        return classify_pna(state, mode, cutoff)
    return state


def classify_pna(state, mode, cutoff):
    """Classify the PNA as tripod or hairpin state. If in the desired
    specified, returns 1, else returns 0."""
    dist_12, dist_34, dist_14, dist_23, dist_13, dist_24 = state
    if mode == "tripod":
        if (
            dist_12 < cutoff
            and dist_34 < cutoff
            and dist_14 > cutoff
            and dist_23 > cutoff
            and dist_13 > cutoff
            and dist_24 > cutoff
        ):
            return 1
        return 0
    elif mode == "hairpin":
        if (
            dist_12 > cutoff
            and dist_34 > cutoff
            and dist_14 < cutoff
            and dist_23 < cutoff
            and dist_13 > cutoff
            and dist_24 > cutoff
        ):
            return 1
        return 0
