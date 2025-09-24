from .environment import Environment
from .utils import write_world
import os
import subprocess
import numpy as np
import MDAnalysis as mda
from joblib import Parallel, delayed
from functools import partial


class LjTemperature(Environment):
    """Lennard-Jones polymer of degree polymerization 30
    with agent control over temperature"""

    def __init__(
        self,
        target,
        action_list,
        n_simulations=100,
        sim_steps_per_round=1000000,
        dump_style="dcd",
        simulation_dir=os.path.join(os.getcwd(), "simulations"),
        data_dir="/home/wo6860/projects/rl_idp/src/data_files/lj_dp30",
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
        if isinstance(target, (int, float)):
            self.target = np.full(n_simulations, target)
        elif isinstance(target, list) and len(target) == n_simulations:
            self.target = np.array(target)
        else:
            raise ValueError(
                "Target must be a single value or a list with length equal to n_simulations."
            )
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
        self.data_dir = data_dir
        self.temperatures = []
        self.seeds = []

    def calculate_rewards(self):
        """Calculate and return rewards based on current state"""
        # Calculate radius of gyration for each simulation
        rg = self._calculate_rg()

        # Reward is the negative absolute difference between rg and target
        return [-1 * abs(rg[i] - self.target[i]) for i in range(self.n_simulations)]

    def calculate_states(self):
        """Return current states"""
        return [[t] for t in self.temperatures]

    def initialize_simulations(self):
        """Initialize all simulations and return initial states"""
        for i in range(self.n_simulations):
            poly_dir = os.path.join(self.simulation_dir, f"poly{i}")
            if not os.path.exists(poly_dir):
                os.makedirs(poly_dir, exist_ok=True)
            subprocess.run(
                ["cp", "-f", os.path.join(self.data_dir, "sys.data"), poly_dir]
            )
            subprocess.run(
                ["cp", "-f", os.path.join(self.data_dir, "sys.settings"), poly_dir]
            )

        self.temperatures = np.random.randint(200, 501, self.n_simulations)
        self.seeds = np.random.randint(0, 1000000, self.n_simulations)
        paths = []
        for i, (T, seed) in enumerate(zip(self.temperatures, self.seeds)):
            path = os.path.join(self.simulation_dir, f"poly{i}/start.lmp")
            paths.append(os.path.join(self.simulation_dir, f"poly{i}"))
            _write_lmp(
                path, self.sim_steps_per_round, T, seed, self.dump_style, rewrite=False
            )

        write_world(self.simulation_dir, paths, "start.lmp")

    def equilibrate_simulations(self):
        """Run equilibration"""
        run_dir = os.path.join(self.simulation_dir, "in.world")
        command = f"srun lmp_intel -in {run_dir} -partition {self.n_simulations}x1 -plog none -pscreen none"
        subprocess.run(command.split(), cwd=self.simulation_dir)

    def modify_simulations(self, actions):
        """Apply actions to simulations"""
        self.temperatures += np.asarray([action[0] for action in actions])

        for i, (T, seed) in enumerate(zip(self.temperatures, self.seeds)):
            path = os.path.join(self.simulation_dir, f"poly{i}/start.lmp")
            _write_lmp(
                path, self.sim_steps_per_round, T, seed, self.dump_style, rewrite=True
            )

    def calculate_additional_metrics(self):
        """Return the list of possible actions"""
        additional_metrics = {}

        # We also want to track Rg
        additional_metrics["rg"] = self._calculate_rg()

        return additional_metrics

    def get_possible_actions(self, states=None):
        """
        Return a list of numpy arrays where each array represents the possible actions
        for a given simulation based on its current temperature.
        """
        # State is temperature for each simulation, limit actions based on temperature
        if states is None:
            states = self.temperatures

        possible_actions = [0] * len(states)
        for i, T in enumerate(states):
            valid_actions = [
                action for action in self.action_list if 100 <= T + action <= 600
            ]
            possible_actions[i] = valid_actions
        return possible_actions

    def _calculate_rg(self):
        """Calculate the radius of gyration for each simulation"""
        commands = []
        for i in range(self.n_simulations):
            structure = os.path.join(self.simulation_dir, f"poly{i}", "sys.data")
            trajectory = os.path.join(
                self.simulation_dir, f"poly{i}", f"coords.{self.dump_style}"
            )

            commands.append([structure, trajectory])

        func = partial(_calc_rg, step=1, start=0.5, dump_style=self.dump_style)

        return Parallel(n_jobs=-1)(
            delayed(func)(structure, trajectory) for structure, trajectory in commands
        )

    def load_state_dict(self, state_dict):
        return super().load_state_dict(state_dict)

    def get_state_dict(self):
        return super().get_state_dict()


def _write_lmp(path, nsteps, T, seed, dump_style="dcd", rewrite=True):
    """
    Write a LAMMPS input script for a simulation of a Lennard-Jones chain.

    Parameters
    ----------
    path : str
        The file path where the LAMMPS input script will be written.
    nsteps : int
        The number of simulation steps to run.
    T : float
        The initial, target, and final temperature for the simulation.
    """

    if dump_style == "dcd":
        dump = f"dump crds all dcd ${{coords_freq}} coords.dcd"
    elif dump_style == "xyz":
        dump = f"dump crds all xyz ${{coords_freq}} coords.xyz"

    if rewrite:
        velocity = ""
    else:
        velocity = f"velocity all create ${{Tinit}} ${{vseed1}} mom yes rot yes\n"

    if rewrite:
        data = "restart.data"
    else:
        data = "sys.data"

    with open(path, "w") as f:
        f.write(f"""# VARIABLES
variable        data_name      index 	{data}
variable        settings_name  index    sys.settings
variable        nsteps         index    {nsteps}
variable        coords_freq    index    500
variable        Tinit          index    {T}
variable        T0	           index    {T}
variable        Tf	           index    {T}
variable        Tdamp          index    1000
variable        vseed1         index    {seed}
variable        vseed2         index    {seed}

#===========================================================
# SYSTEM DEFINITION
#===========================================================
units		real	# m = grams/mole, x = Angstroms, E = kcal/mole
dimension	3	# 3 dimensional simulation
newton		on	# use Newton's 3rd law
boundary	p p p	# shrink wrap conditions
atom_style	full    # molecular + charge

#===========================================================
# FORCE FIELD DEFINITION
#===========================================================
pair_style     lj/cut 50
pair_modify    shift yes
bond_style     hybrid harmonic
special_bonds  fene
angle_style    none
dihedral_style none
kspace_style   none
improper_style none                 # no impropers

#===========================================================
# SETUP SIMULATIONS
#===========================================================
# READ IN COEFFICIENTS/COORDINATES/TOPOLOGY
read_data ${{data_name}}
include ${{settings_name}}

# SET RUN PARAMETERS
neighbor 3.5 multi
comm_style tiled              #could be removed
timestep   10
run_style	verlet 		# Velocity-Verlet integrator

# DECLARE RELEVANT OUTPUT VARIABLES
variable        my_step   equal   step
variable        my_temp   equal   temp
variable        my_rho    equal   density
variable        my_pe     equal   pe
variable        my_ke     equal   ke
variable        my_etot   equal   etotal
variable        my_ent    equal   enthalpy
variable        my_P      equal   press
variable        my_vol    equal   vol

#===========================================================
# PERFORM ENERGY MINIMIZATION
#===========================================================

#===========================================================
# SET OUTPUTS
#===========================================================
fix  averages all ave/time 100 1 100 v_my_temp v_my_etot v_my_pe v_my_ke v_my_ent v_my_P v_my_rho file thermo.avg
{dump}
dump_modify crds sort id

#===========================================================
# RUN DYNAMICS
#===========================================================
{velocity}fix lang     all langevin ${{T0}} ${{Tf}} ${{Tdamp}} ${{vseed2}}
fix dynamics all nve
fix bal      all balance 1000 1.0 shift xyz 10 1.1
run             ${{nsteps}}
write_data      restart.data pair ij
unfix dynamics""")


def _calc_rg(structure, trajectory, step=1, start=0.5, dump_style="dcd"):
    """Calculate radius of gyration of a single chain polymer simulation

    Parameters
    ----------
    structure : lammps data file
        Path to lammps data file containing simulation topology
    trajectory : lammps dcd trajectory file
        Path to lammps binary trajectory file

    Returns
    -------
    rg : float
        Mean radius of gyration in simulation
    """

    # initialize universe
    if dump_style == "dcd":
        u = mda.Universe(
            structure,
            trajectory,
            atom_style="id resid type charge x y z vx vy vz",
            topology_format="DATA",
            format="DCD",
        )
    elif dump_style == "xyz":
        u = mda.Universe(
            structure,
            trajectory,
            atom_style="id resid type charge x y z vx vy vz",
            topology_format="DATA",
            format="XYZ",
        )

    # create atom groups
    chain = u.select_atoms("resid 1")
    if dump_style == "dcd":
        L = u.dimensions[0]
    elif dump_style == "xyz":
        with open(structure, "r") as f:
            for line in f:
                if "xlo xhi" in line:
                    xlo, xhi = map(float, line.split()[:2])
                    break
            L = xhi - xlo
    start = int(start * len(u.trajectory))

    # Initilize rg array
    rg = [0] * len(u.trajectory[start::step])

    # Go through each timestep
    for i, ts in enumerate(u.trajectory[start::step]):
        # Calculate particle positions relative to reference atom
        delta = chain.positions - chain[0].position

        # Account for periodic boundary conditions
        pbc_vectors = np.where(
            delta > L / 2, delta - L, np.where(delta < -L / 2, delta + L, delta)
        )

        # Calculate center of mass
        masses = chain.masses
        com = np.sum(pbc_vectors.T * masses, axis=1) / np.sum(masses)

        # Calculate distances from COM based on pbc vectors
        distances = np.zeros(len(pbc_vectors))
        for k in range(len(pbc_vectors)):
            for j in range(3):
                distances[k] += (pbc_vectors[k, j] - com[j]) ** 2

        # Calculate mass weighted rg
        rg[i] = np.sqrt(np.sum(distances * masses) / np.sum(masses))

    return np.mean(rg)
