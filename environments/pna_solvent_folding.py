from .environment import Environment
from .utils import write_universe, LammpsSettings
from abc import ABC
import os
import subprocess
import numpy as np
import MDAnalysis as mda
from joblib import Parallel, delayed
from functools import partial


class PNAFolding(Environment, ABC):
    '''Lennard-Jones polymer of degree polymerization 30 
    with agent control over temperature'''
    
    
    def __init__(self, sequence_file, target, action_list, temperature,
                 n_simulations=112, sim_steps_per_round=1000000, dump_style='dcd', 
                 simulation_dir=os.path.join(os.getcwd(), 'simulations'),
                 ):
        '''
        Initialize the Lennard-Jones temperature environment.

        Parameters
        ----------
        target : float or list
            Target temperature(s) for the environment.
        n_simulations : int
            Number of simulations to run.
        sim_steps_per_round : int
            Number of simulation steps per round.
        '''
        super().__init__()
        self.sequence_file = sequence_file
        if target not in ('tripod', 'hairpin'):
            raise ValueError("Target must be either 'tripod' or 'hairpin'.")
        self.target = target
        if isinstance(action_list, list) and all(isinstance(sublist, list) for sublist in action_list):
            self.action_list = action_list
        else:
            raise ValueError("action_list must be a 2D array or a list of lists.")
        self.temperature = temperature
        self.n_simulations = n_simulations
        self.sim_steps_per_round = sim_steps_per_round
        self.dump_style = dump_style
        self.simulation_dir = simulation_dir
        self.states = []
        self.interaction_strengths = []
        self.settings_files = []
        self.seeds = []

    
    def initialize_simulations(self):
        '''Initialize all simulations and return initial states'''
        commands = []
        paths = []

        self.seeds = np.random.randint(0, 1000000, self.n_simulations)
        for i in range(self.n_simulations):
            poly_dir = os.path.join(self.simulation_dir, f'poly{i}')
            paths.append(poly_dir)

            # Execute Sai's script to create initial scripts
            commands.append(['python', '/home/wo6860/projects/pna/sum2024_project/generate.py', 
                             self.sequence_file,
                            '--output', poly_dir,
                            '--temperature', str(self.temperature),
                            '--nsteps', str(self.sim_steps_per_round),
                            '--seed', str(self.seeds[i]),
                            '--coords_freq', '10000',
                            '--dump_style', self.dump_style,
                            '--database', 'db.py',
                            ])

        # Wrapper command for executing Sai's script in parallel
        def begin_wrapper(command):
            subprocess.run(command)
            tiger_submit_file = os.path.join(command[4], 'tiger.submit')
            if os.path.exists(tiger_submit_file):
                os.remove(tiger_submit_file)
            if os.path.exists(os.path.join(command[4], 'sys.data')):
                subprocess.run(['cp', os.path.join(command[4], 'sys.data'), os.path.join(command[4], 'start_config.data')])
            return 0

        # Run the commands in parallel
        Parallel(n_jobs=-1)(delayed(begin_wrapper)(command) for command in commands)

        # Now that simulation files have been created, establish the interaction strengths
        # Set the interaction strength of the AT pairs first from 0 to 100
        at_strength = np.random.uniform(0, 100, self.n_simulations)
        # Make the GC strength 100 minus AT strength
        # The idea is that, if you can't change the monomer chemistry, it is likely
        # difficult to decouple solvent quality
        self.interaction_strengths = [[x, 100 - x] for x in at_strength]
        
        # for loop to modify the newly created sys.settings files    
        for i, interaction_strength in enumerate(self.interaction_strengths):

            # Initialize the settings file object
            settings_file_path = os.path.join(self.simulation_dir, f'poly{i}', 'sys.settings')
            settings = LammpsSettings(settings_file_path)

            # Modify the appropriate nonbonded interactions in the settings file
            interaction_strength = self.interaction_strengths[i]
            at_strength, gc_strength = interaction_strength[0], interaction_strength[1]
            settings.modify_non_bonded(2, 3, 'lj/gromacs', [at_strength, 0.3, 0.57, 0.6])
            settings.modify_non_bonded(4, 5, 'lj/gromacs', [gc_strength, 0.3, 0.57, 0.6])
            settings.write()

            self.settings_files.append(settings)

        write_universe(self.simulation_dir, paths, 'in.pna')

    
    def equilibrate_simulations(self):
        '''Run equilibration'''
        run_dir = os.path.join(self.simulation_dir, 'in.universe')
        available_cpus = int(os.environ.get('SLURM_NTASKS'))
        command = f'srun /home/wo6860/software/lammps-stable_29Aug2024/build_pna/lmp_intel -in {run_dir} -partition {available_cpus}x1 -plog none -pscreen none'
        subprocess.run(command.split(), cwd=self.simulation_dir)
    
    
    def modify_simulations(self, actions):
        '''Apply actions to simulations'''
        # Change the interactions based on actions
        self.interaction_strengths = [[action[0], 100 - action[0]] for action in actions]

        # Modify lammps settings files to reflect new interactions
        for settings, interaction_strength in zip(self.settings_files, self.interaction_strengths):
            at_strength, gc_strength = interaction_strength[0], interaction_strength[1]
            settings.modify_non_bonded(2, 3, 'lj/gromacs', [at_strength, 0.3, 0.57, 0.6])
            settings.modify_non_bonded(4, 5, 'lj/gromacs', [gc_strength, 0.3, 0.57, 0.6])
            settings.write()
    
    
    def get_possible_actions(self, states=None):
        '''
        Return a list of numpy arrays where each array represents the possible actions
        for a given simulation based on its current temperature.
        '''
        # State is temperature for each simulation, limit actions based on temperature
        if states is None:
            print("No states provided, using current interaction strengths.")
            interaction_strengths = self.interaction_strengths
        else:
            print("Using provided states for action calculation.")
            interaction_strengths = [[s[0], s[1]] for s in states]

        possible_actions = [0] * len(interaction_strengths)
        print(f'{len(possible_actions)=}')
        print(f'{len(interaction_strengths)=}')
        for i, interaction_strength in enumerate(interaction_strengths):
            # For now, all of the actions will be valid actions as long as 
            # the new actions are directly choosing the interaction strength
            valid_actions = self.action_list
            possible_actions[i] = valid_actions

        return possible_actions

    
    def load_state_dict(self, state_dict):
        '''Load the state dictionary into the environment'''
        self.sequence_file = state_dict['sequence_file']
        self.target = state_dict['target']
        self.action_list = state_dict['action_list']
        self.n_simulations = state_dict['n_simulations']
        self.sim_steps_per_round = state_dict['sim_steps_per_round']
        self.dump_style = state_dict['dump_style']
        self.simulation_dir = state_dict['simulation_dir']
        self.states = state_dict['states']
        self.temperature = state_dict['temperature']
        self.seeds = state_dict['seeds']
        self.interaction_strengths = state_dict['interaction_strengths']
        self.settings_files = state_dict['settings_files']
        print("State dictionary loaded successfully.")

        return 0
    
    
    def get_state_dict(self):
        '''Return the dictionary of state variables'''
        state_dict = {
            'sequence_file': self.sequence_file,
            'target': self.target,
            'action_list': self.action_list,
            'n_simulations': self.n_simulations,
            'sim_steps_per_round': self.sim_steps_per_round,
            'dump_style': self.dump_style,
            'simulation_dir': self.simulation_dir,
            'states': self.states,
            'temperature': self.temperature,
            'seeds': self.seeds,
            'interaction_strengths': self.interaction_strengths,
            'settings_files': self.settings_files,
        }

        return state_dict

        
class AllOrNothingPNAFolding(PNAFolding):
    '''PNA Folding environment with all-or-nothing reward'''
    
    def calculate_states(self):
        '''Return current states. Interaction strengths must be first two items'''
        commands = []
        for i in range(self.n_simulations):
            structure_path = os.path.join(self.simulation_dir, f'poly{i}/start_config.data')
            trajectory_path = os.path.join(self.simulation_dir, f'poly{i}/coords.{self.dump_style}')
            commands.append((structure_path, trajectory_path))

        func = partial(pna_cv, 
                       fraction=0.25, 
                       mode=self.target, 
                       dump_style=self.dump_style,
                       average=True,
                       classify=True,
                       cutoff=5.0,
                       )

        half_available_cpus = int(os.environ.get('SLURM_NTASKS', 1)) // 2
        states = Parallel(n_jobs=half_available_cpus, timeout=60)(
            delayed(func)(*command) for command in commands
        )
        self.states = [i + [s] for i, s in zip(self.interaction_strengths, states)]
        return self.states
    
    
    def calculate_rewards(self):
        '''Calculate and return rewards based on current state'''
        return [s[2] for s in self.states]  # Assuming states are structured as [[state_value], ...]

    
    def calculate_additional_metrics(self):
        '''Any other calculations of interest. Return a dictionary
        with keys as metric names and values as the calculated metric.'''
        additional_metrics = {}
        additional_metrics['interaction_strengths'] = self.interaction_strengths  
        additional_metrics['percent_on_target'] = [[sum(self.calculate_rewards()) / self.n_simulations * 100]]
        return additional_metrics

        
class IncrementalPNAFolding(PNAFolding):
    '''PNA Folding environment with incremental reward'''
    
    def calculate_states(self):
        '''Return current states. Interaction strengths must be first two items'''
        commands = []
        for i in range(self.n_simulations):
            structure_path = os.path.join(self.simulation_dir, f'poly{i}/start_config.data')
            trajectory_path = os.path.join(self.simulation_dir, f'poly{i}/coords.{self.dump_style}')
            commands.append((structure_path, trajectory_path))

        func = partial(pna_cv, 
                       fraction=0.25, 
                       mode=self.target, 
                       dump_style=self.dump_style,
                       average=True,
                       classify=False,
                       cutoff=5.0,
                       )

        half_available_cpus = int(os.environ.get('SLURM_NTASKS', 1)) // 2
        states = Parallel(n_jobs=half_available_cpus, timeout=60)(
            delayed(func)(*command) for command in commands
        )
        self.states = [i + list(s) for i, s in zip(self.interaction_strengths, states)]
        return self.states
    
    
    def calculate_rewards(self):
        '''Calculate and return rewards based on current state'''
        # Extract distances from states, assuming states are structured as 
        # [interaction_strength_1, interaction_strength_2, dist_12, dist_34, ...]
        pna_cv = [s[2:] for s in self.states] 

        # See if the PNA is in the target state
        classified_states = [classify_pna(state, self.target, cutoff=5.0) for state in pna_cv]

        # Calculate rewards based on distances and classification
        rewards = [0] * self.n_simulations
        for i, (state, classification) in enumerate(zip(pna_cv, classified_states)):

            # For a given target, the goal is to minimize certain distances and maximize others
            # For example, in a tripod state, we want distances 12 and 34 to be small, while 14 and 23 should be large
            if self.target == 'tripod':
                rewards[i] += -1 * state[0] + -1 * state[1] + state[2] + state[3] + state[4] + state[5]
            elif self.target == 'hairpin':
                rewards[i] += state[0] + state[1] + -1 * state[2] + -1 * state[3] + state[4] + state[5]

            # # If the PNA is in the target state, add a bonus
            # rewards[i] += 100 * classification 

        return rewards

    
    def calculate_additional_metrics(self):
        '''Any other calculations of interest. Return a dictionary
        with keys as metric names and values as the calculated metric.'''
        additional_metrics = {}
        additional_metrics['interaction_strengths'] = self.interaction_strengths  
        classified_states = [classify_pna(state[2:], self.target, cutoff=5.0) for state in self.states]
        additional_metrics['percent_on_target'] = [[sum(classified_states) / self.n_simulations * 100]]
        additional_metrics['pna_cv'] = [s[2:] for s in self.states]  # Extract PNA CVs from states
        return additional_metrics

        
        
class GroupedPNAFolding(PNAFolding):
    '''PNA folding environment where agent controls groups of simulations
    instead of individually controlling each simulation'''

    def __init__(self, sequence_file, target, action_list, temperature,
                 n_simulations=112, sim_steps_per_round=1000000, dump_style='dcd', 
                 simulation_dir=os.path.join(os.getcwd(), 'simulations'),
                 group_size=4):
        '''
        Initialize the grouped PNA folding environment.

        Parameters
        ----------
        group_size : int
            Number of simulations in each group.
        '''
        super().__init__(sequence_file, target, action_list, temperature,
                         n_simulations, sim_steps_per_round, dump_style, simulation_dir)
        if n_simulations % group_size != 0:
            raise ValueError("n_simulations must be divisible by group_size.")
        self.group_size = group_size
        self.n_groups = n_simulations // group_size

        
    def initialize_simulations(self):
        '''Initialize all simulations and return initial states'''
        commands = []
        paths = []
        self.seeds = np.random.randint(0, 1000000, self.n_simulations)
        for i in range(self.n_simulations):
            poly_dir = os.path.join(self.simulation_dir, f'poly{i}')
            paths.append(poly_dir)
            commands.append(['python', '/home/wo6860/projects/pna/sum2024_project/generate.py', 
                             self.sequence_file,
                            '--output', poly_dir,
                            '--temperature', str(self.temperature),
                            '--nsteps', str(self.sim_steps_per_round),
                            '--seed', str(self.seeds[i]),
                            '--coords_freq', '10000',
                            '--dump_style', self.dump_style,
                            '--database', 'db.py',
                            ])

        def begin_wrapper(command):
            subprocess.run(command)
            tiger_submit_file = os.path.join(command[4], 'tiger.submit')
            if os.path.exists(tiger_submit_file):
                os.remove(tiger_submit_file)
            if os.path.exists(os.path.join(command[4], 'sys.data')):
                subprocess.run(['cp', os.path.join(command[4], 'sys.data'), os.path.join(command[4], 'start_config.data')])
            return 0

        Parallel(n_jobs=-1)(delayed(begin_wrapper)(command) for command in commands)

        # Now that simulation files have been created, establish the interaction strengths
        # Set the interaction strength of the AT pairs first from 0 to 100
        at_strength = np.random.uniform(0, 100, self.n_groups)
        # Make the GC strength 100 minus AT strength
        # The idea is that, if you can't change the monomer chemistry, it is likely
        # difficult to decouple solvent quality
        self.interaction_strengths = [[x, 100 - x] for x in at_strength]
        
        # for loop to modify the newly created sys.settings files    
        for i in range(self.n_simulations):

            # Initialize the settings file object
            settings_file_path = os.path.join(self.simulation_dir, f'poly{i}', 'sys.settings')
            settings = LammpsSettings(settings_file_path)

            # Modify the appropriate nonbonded interactions in the settings file
            interaction_strength = self.interaction_strengths[i // self.group_size]
            at_strength, gc_strength = interaction_strength[0], interaction_strength[1]
            settings.modify_non_bonded(2, 3, 'lj/gromacs', [at_strength, 0.3, 0.57, 0.6])
            settings.modify_non_bonded(4, 5, 'lj/gromacs', [gc_strength, 0.3, 0.57, 0.6])
            settings.write()

            self.settings_files.append(settings)

        write_universe(self.simulation_dir, paths, 'in.pna')


    def modify_simulations(self, actions):
        '''Apply actions to simulations'''
        # Change the interactions based on actions
        self.interaction_strengths = [[action[0], 100 - action[0]] for action in actions]

        # Modify lammps settings files to reflect new interactions
        for i, settings in enumerate(self.settings_files):
            interaction_strength = self.interaction_strengths[i // self.group_size]
            at_strength, gc_strength = interaction_strength[0], interaction_strength[1]
            settings.modify_non_bonded(2, 3, 'lj/gromacs', [at_strength, 0.3, 0.57, 0.6])
            settings.modify_non_bonded(4, 5, 'lj/gromacs', [gc_strength, 0.3, 0.57, 0.6])
            settings.write()

                        
    def calculate_states(self):
        '''Return current states. Current states is the concatenation of
        the temperature and something I haven't figured out yet'''

        # Start by calculating the pna cv for each simulation and classifying them
        commands = []
        for i in range(self.n_simulations):
            structure_path = os.path.join(self.simulation_dir, f'poly{i}/start_config.data')
            trajectory_path = os.path.join(self.simulation_dir, f'poly{i}/coords.{self.dump_style}')
            commands.append((structure_path, trajectory_path))

        func = partial(pna_cv, 
                       fraction=0.25, 
                       mode=self.target, 
                       dump_style=self.dump_style,
                       average=True,
                       classify=True,
                       cutoff=5.0,
                       )

        # Only use half of the available CPUs so that we don't get OOM error
        # (MDAnalysis is memory intensive)
        half_available_cpus = int(os.environ.get('SLURM_NTASKS', 1)) // 2
        individual_states = Parallel(n_jobs=half_available_cpus, timeout=60)(
            delayed(func)(*command) for command in commands
        )
        
        # Group the states by group size and compute mean for each group
        # In this calculation, the state becomes the percentage of groups
        # in the target state.
        group_states = [0] * self.n_groups
        for i in range(self.n_groups):
            group_percent_on_target = np.mean(individual_states[i * self.group_size:(i + 1) * self.group_size]) * 100
            group_states[i] = group_percent_on_target.item()

        # Combine interaction strengths and group state as a list of lists
        # Each state is now [interaction_strength_at, interaction_strength_gc, group_state]
        self.states = [i + [s] for i, s in zip(self.interaction_strengths, group_states)]
        return self.states


    def calculate_rewards(self):
        '''Reward is the percentage of simulations in the target state
        within the group'''
        # Calculate rewards for each group
        rewards = [0] * self.n_groups
        for i in range(self.n_groups):
            rewards[i] = self.states[i][2]

        return rewards
    
    
    def calculate_additional_metrics(self):
        '''Any other calculations of interest. Return a dictionary
        with keys as metric names and values as the calculated metric.'''
        additional_metrics = {}
        additional_metrics['interaction_strengths'] = self.interaction_strengths
        additional_metrics['percent_on_target'] = [[self.states[i][2] for i in range(self.n_groups)]]
        return additional_metrics


    
def pna_cv(structure, trajectory, fraction=0.25, mode='tripod', dump_style='dcd', average=True, classify=False, cutoff=5.0):
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
    u = mda.Universe(structure, trajectory, \
                    atom_style='id resid type charge x y z vx vy vz', \
                    topology_format='DATA', format=dump_style.upper())

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
        pbc_vectors = np.where(delta > ts.dimensions[0]/2, delta-ts.dimensions[0], np.where(delta < -ts.dimensions[0]/2, delta+ts.dimensions[0], delta))
        # Compute the centers of mass for each group
        com1 = np.sum(pbc_vectors[group1.indices].T * group1.masses, axis=1) / np.sum(group1.masses)
        com2 = np.sum(pbc_vectors[group2.indices].T * group2.masses, axis=1) / np.sum(group2.masses)
        com3 = np.sum(pbc_vectors[group3.indices].T * group3.masses, axis=1) / np.sum(group3.masses)
        com4 = np.sum(pbc_vectors[group4.indices].T * group4.masses, axis=1) / np.sum(group4.masses)

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
    '''Classify the PNA as tripod or hairpin state. If in the desired
    specified, returns 1, else returns 0.'''
    dist_12, dist_34, dist_14, dist_23, dist_13, dist_24 = state
    if mode == 'tripod':
        if dist_12 < cutoff and dist_34 < cutoff and dist_14 > cutoff and dist_23 > cutoff and dist_13 > cutoff and dist_24 > cutoff:
            return 1
        return 0 
    elif mode == 'hairpin':
        if dist_12 > cutoff and dist_34 > cutoff and dist_14 < cutoff and dist_23 < cutoff and dist_13 > cutoff and dist_24 > cutoff:
            return 1
        return 0 