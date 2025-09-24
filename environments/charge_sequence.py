from .environment import Environment
from abc import ABC, abstractmethod
from .utils import write_universe, LammpsData
import os
import subprocess
import numpy as np
import MDAnalysis as mda
from joblib import Parallel, delayed
from functools import partial
import random

class ChargeSequence(Environment, ABC):
    '''Base class for charge sequence environments, the two
    main ones being control over SCD and control over sequence.'''
    
    
    def __init__(self, target, action_list, n_simulations=112, sim_steps_per_round=1000000,
                 dump_style='dcd', simulation_dir=os.path.join(os.getcwd(), 'simulations'),
                 data_dir='/home/wo6860/my_package/ambition/rl/src/data_files/charge_sequence',
                 T=300):
        '''
        Initialize the polyampholyte environment.

        Parameters
        ----------
        target : float or list
            Target Rg for the environment.
        action_list : list of lists
            List of possible actions, where each action is a list of SCD changes.
        n_simulations : int
            Number of simulations to run.
        sim_steps_per_round : int
            Number of simulation steps per round.
        dump_style : str
            Style of the dump file (e.g., 'dcd', 'xyz').
        simulation_dir : str
            Directory where simulations will be stored.
        data_dir : str
            Directory where initial data files are stored.
        '''
        super().__init__()
        if isinstance(target, (int, float)):
            self.target = np.full(n_simulations, target)
        elif isinstance(target, list) and len(target) == n_simulations:
            self.target = np.array(target)
        else:
            raise ValueError("Target must be a single value or a list with length equal to n_simulations.")
        if isinstance(action_list, list) and all(isinstance(sublist, list) for sublist in action_list):
            self.action_list = action_list
        else:
            raise ValueError("action_list must be a 2D array or a list of lists.")
        self.n_simulations = n_simulations
        self.sim_steps_per_round = sim_steps_per_round
        self.dump_style = dump_style
        self.simulation_dir = simulation_dir
        self.data_dir = data_dir
        self.T = T
        self.states = []
        self.data_files = []
        self.seeds = []

    
    def calculate_rewards(self):
        '''Calculate and return rewards based on current state'''
        # Calculate radius of gyration for each simulation
        rg = self._calculate_rg()

        # Reward is the negative absolute difference between rg and target
        return [-1 * abs(rg[i] - self.target[i]) for i in range(self.n_simulations)]
    

    def initialize_simulations(self):
        '''Initialize all simulations and calculate initial states'''
        self.data_files = []
        for i in range(self.n_simulations):
            poly_dir = os.path.join(self.simulation_dir, f'poly{i}')
            if not os.path.exists(poly_dir):
                os.makedirs(poly_dir, exist_ok=True)

            # Instantiate data file object
            data = ChargeSequenceData(os.path.join(self.data_dir, 'sys.data'))
            N = data.num_atoms
            half_N = N // 2
            array = [1] * half_N + [-1] * (N - half_N)

            # Generate random charge distribution and modify data file
            random.shuffle(array)
            for atom, charge in enumerate(array):
                data.modify_atom(atom+1, charge=charge)
            data.filename = os.path.join(poly_dir, 'sys.data')

            # Finally, write data file and append to list of data files
            data.write()
            self.data_files.append(data)

            # also copy the settings file
            subprocess.run(['cp', '-f', os.path.join(self.data_dir, 'sys.settings'), poly_dir])

        self.states = self.calculate_states()
        self.seeds = np.random.randint(0, 1000000, self.n_simulations)
        paths = []
        for i, seed in enumerate(self.seeds):
            path = os.path.join(self.simulation_dir, f'poly{i}/start.lmp')
            paths.append(os.path.join(self.simulation_dir, f'poly{i}'))
            _write_lmp(path, self.sim_steps_per_round, self.T, seed, self.dump_style)

        write_universe(self.simulation_dir, paths, 'start.lmp')
    
    
    def equilibrate_simulations(self):
        '''Run equilibration'''
        run_dir = os.path.join(self.simulation_dir, 'in.universe')
        available_cpus = int(os.environ.get('SLURM_NTASKS'))
        command = f'srun lmp_intel -in {run_dir} -partition {available_cpus}x1 -plog none -pscreen none'
        subprocess.run(command.split(), cwd=self.simulation_dir)

        return 0
    
    
    def calculate_additional_metrics(self):
        '''Return the list of possible actions'''
        additional_metrics = {}

        # We also want to track Rg
        additional_metrics['rg'] = self._calculate_rg()

        return additional_metrics

        
    def get_possible_actions(self, states=None):
        '''
        Return a list of numpy arrays where each array represents the possible actions
        for a given simulation based on its current temperature.
        '''
        # State is temperature for each simulation, limit actions based on temperature
        if states is None:
            states = self.states

        possible_actions = [0] * len(states)
        for i, s in enumerate(states):
            valid_actions = self.action_list # All actions are valid for this environment
            possible_actions[i] = valid_actions
        return possible_actions

    
    def _calculate_rg(self):
        '''Calculate the radius of gyration for each simulation'''
        commands = []
        for i in range(self.n_simulations):
            structure  = os.path.join(self.simulation_dir, f'poly{i}', 'sys.data')
            trajectory = os.path.join(self.simulation_dir, f'poly{i}', f'coords.{self.dump_style}')

            commands.append([structure, trajectory])

        func = partial(_calc_rg, step=1, start=0.5, dump_style=self.dump_style)

        return Parallel(n_jobs=-1)(delayed(func)(structure, trajectory) for structure, trajectory in commands)

        
    def get_state_dict(self):
        '''Return the dictionary of state variables'''
        state_dict = {
            'states': self.states,
            'data_files': [data.filename for data in self.data_files],
            'seeds': self.seeds,
            'simulation_dir': self.simulation_dir,
            'data_dir': self.data_dir,
            'T': self.T,
            'target': self.target.tolist(),
            'action_list': self.action_list,
        }

        return state_dict

    
    def load_state_dict(self, state_dict):
        '''Load the state dictionary into the environment'''
        self.states = state_dict['states']
        self.data_files = [ChargeSequenceData(filename) for filename in state_dict['data_files']]
        self.seeds = state_dict['seeds']
        self.simulation_dir = state_dict['simulation_dir']
        self.data_dir = state_dict['data_dir']
        self.T = state_dict['T']
        self.target = np.array(state_dict['target'])
        self.action_list = state_dict['action_list']
        print("Environment state loaded successfully")

        return 0
    


class SCDChargeSequence(ChargeSequence):
    '''Charge sequence environment where the agent controls the Sequence Charge Decorator (SCD).'''
    def modify_simulations(self, actions):
        '''Apply actions to simulations'''

        # Modify the scd based on the action
        self.states = [[state[0] + action[0]] for state, action in zip(self.states, actions)]

        # Generate new charge sequence based on the SCD
        charges = Parallel(n_jobs=-1)(delayed(_generate_SCD)(scd[0]) for scd in self.states)

        # Rewrite the charges in the data files
        for data, charge in zip(self.data_files, charges):
            data.rewrite_charges(charge)
            data.write()

        return 0
    
    
    def calculate_states(self):
        '''Return current states'''
        if len(self.states) == 0:
            return [[_calc_SCD(data.return_charges())] for data in self.data_files]
        else:
            return self.states


        
class DirectChargeSequence(ChargeSequence):
    '''Charge sequence environment where the agent controls the sequence directly
    by swapping the positions of two charges.'''
    def calculate_states(self):
        '''Return current states'''
        states = []
        for data in self.data_files:
            charges = data.return_charges()
            if charges[0] == -1:
                charges = [-c for c in charges]
            states.append(charges)
        return states

    
    def modify_simulations(self, actions):
        '''Apply actions to simulations'''

        # Swap charges at indices given by the action
        for data, action in zip(self.data_files, actions):
            data.swap_charges(action)
            data.write()


    
class ChargeSequenceData(LammpsData):
    '''Modification to the LammpsData class for more explicit handling of charge sequences.'''
    def rewrite_charges(self, charges):
        '''Rewrites the charges of atoms based on a list of charges.'''
        sorted_atoms = sorted(self.atoms, key=lambda x: x['id'])
        for atom, charge in zip(sorted_atoms, charges):
            atom['charge'] = charge

    def return_charges(self):
        '''Return the charges of the current charge sequence.'''
        return [atom['charge'] for atom in sorted(self.atoms, key=lambda x: x['id'])]

    def swap_charges(self, idxs):
        """
        Swap charges between two positions indicated by a one-hot binary vector.
        If idxs contains two 1's, swap charges at those positions (0-based index).
        If idxs contains zero or one 1's, do nothing.
        """
        indices = [i for i, val in enumerate(idxs) if val == 1]
        if len(indices) == 2:
            sorted_atoms = sorted(self.atoms, key=lambda x: x['id'])
            atom1 = sorted_atoms[indices[0]]
            atom2 = sorted_atoms[indices[1]]
            atom1['charge'], atom2['charge'] = atom2['charge'], atom1['charge']
        elif len(indices) == 0:
            pass  # No swap
        else:
            print(f"Invalid action: more than two positions selected for swap.")



def _calc_SCD(sequence):
    """
    Calculates the Sequence Charge Decorator (SCD) for a given sequence of 'R' (charge +1) and 'D' (charge -1).
    The SCD is the sum over i<j of (charge_i * charge_j * (i-j)^(1/2)), normalized by sequence length.
    """

    # Convert to list if it's a string and convert to numerical charges
    seq = list(sequence)
    if 'R' and 'D' in seq:
        charge_dict = {'R': 1, 'D': -1}
        seq = [charge_dict[aa] for aa in seq]
    
    # compute SCD
    scd = 0.0
    for i in range(1, len(seq)):
        for j in range(i):
            scd += seq[i] * seq[j] * (i - j) ** 0.5
    return scd / len(seq)

def _mcmove(sequence, old_scd, T, target=None):
    """
    Perform a single Monte Carlo move on the sequence:
    - Pick two random indices and swap them.
    - Calculate the new SCD.
    - Decide to accept or reject based on difference from old SCD (and target if provided).
    """
    seq = list(sequence)  # ensure it's mutable
    length = len(seq)

    # Choose two random positions to swap
    i, j = np.random.choice(length, 2, replace=False)
    seq[i], seq[j] = seq[j], seq[i]

    new_scd = _calc_SCD(seq)

    # If a target is given, measure the difference from the target
    if target is not None:
        diff = abs(new_scd - target) - abs(old_scd - target)
    else:
        diff = new_scd - old_scd

    # If diff is negative or we pass the MC acceptance criterion, accept
    if diff <= 0:
        return ''.join(seq), new_scd
    else:
        # Metropolis criterion
        if np.random.rand() < np.exp(-diff / T):
            return ''.join(seq), new_scd
        else:
            # reject swap
            return sequence, old_scd

def _generate_initial_sequence(length=30, positives=15):
    """
    Generate an initial sequence with `positives` times 'R' and `length - positives` times 'D'.
    Then shuffle it randomly.
    """
    seq = ['R'] * positives + ['D'] * (length - positives)
    np.random.shuffle(seq)
    return ''.join(seq)

def _simulated_annealing(sequence, timesteps=100000, T=0.5, target=None):
    """
    Perform a Monte Carlo simulation (simulated annealing) to adjust the sequence
    toward a specified target SCD (if provided).
    """
    seq = sequence
    scd = _calc_SCD(seq)

    for step in range(timesteps):
        seq, scd = _mcmove(seq, scd, T, target=target)

        # Optionally lower temperature over time (simulated annealing)
        # This reduces acceptance rate as we progress
        if step % (timesteps // 200) == 0:
            T *= 0.99

    return seq, scd

def _generate_SCD(target_scd):
    # Parameters
    length = 14            # total length of sequence
    positives = 7         # how many R's (positive charges)
    timesteps = 50000      # total Monte Carlo steps
    T = 0.5                # initial temperature

    # Generate an initial random sequence
    sequence = _generate_initial_sequence(length=length, positives=positives)

    # Run simulated annealing to move toward the target SCD
    final_sequence, final_scd = _simulated_annealing(sequence, timesteps=timesteps, T=T, target=target_scd)
    charge_dict = {'R': 1, 'D': -1}

    charges = [charge_dict[aa] for aa in final_sequence]
    return charges



def _calc_rg(structure, trajectory, step=1, start=0.5, dump_style='dcd'):
    '''Calculate radius of gyration of a single chain polymer simulation
    
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
    '''

    # initialize universe
    if dump_style == 'dcd':
        u = mda.Universe(structure, trajectory, \
                        atom_style='id resid type charge x y z vx vy vz', \
                        topology_format='DATA', format='DCD')
    elif dump_style == 'xyz':
        u = mda.Universe(structure, trajectory, \
                        atom_style='id resid type charge x y z vx vy vz', \
                        topology_format='DATA', format='XYZ')

    # create atom groups
    chain = u.select_atoms('resid 1')
    if dump_style == 'dcd':
        L = u.dimensions[0]
    elif dump_style == 'xyz':
        with open(structure, 'r') as f:
            for line in f:
                if 'xlo xhi' in line:
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
        pbc_vectors = np.where(delta > L/2, delta-L, np.where(delta < -L/2, delta+L, delta))

        # Calculate center of mass
        masses = chain.masses
        com = np.sum(pbc_vectors.T*masses,axis=1) / np.sum(masses)

        # Calculate distances from COM based on pbc vectors
        distances = np.zeros(len(pbc_vectors))
        for k in range(len(pbc_vectors)):
            for j in range(3):
                distances[k] += (pbc_vectors[k,j] - com[j])**2

        # Calculate mass weighted rg
        rg[i] = np.sqrt(np.sum(distances * masses) / np.sum(masses))

    return np.mean(rg)

    
def _write_lmp(path, nsteps, T, seed, dump_style='dcd'):
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

    if dump_style == 'dcd':
        dump = f'dump crds all dcd ${{coords_freq}} coords.dcd'
    elif dump_style == 'xyz':
        dump = f'dump crds all xyz ${{coords_freq}} coords.xyz'
        

    with open(path,'w') as f:
        f.write(f'''# VARIABLES
variable        data_name      index 	sys.data
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
pair_style     lj/cut/coul/cut 50
pair_modify    shift yes
bond_style     hybrid harmonic
special_bonds  fene
angle_style    none
dihedral_style none
kspace_style   none
improper_style none                 # no impropers
dielectric     80.0

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
velocity all create ${{Tinit}} ${{vseed1}} mom yes rot yes
fix lang     all langevin ${{T0}} ${{Tf}} ${{Tdamp}} ${{vseed2}}
fix dynamics all nve
fix bal      all balance 1000 1.0 shift xyz 10 1.1
run             ${{nsteps}}
write_data      sys.data pair ij
unfix dynamics''')