import os


def write_world(path, subdirectories, lammps_input, name="in.world"):
    """
    Write the LAMMPS input file for a world simulation.

    Args:
        path (str): The path to the directory where the input file will be written.
        subdirectories (list): A list of subdirectories to be included in the LAMMPS simulation.
        lammps_input (str): Name of the lammps input file to use.

    Returns:
        None
    """

    with open(os.path.join(path, name), "w") as p:
        p.write(f"""variable d world {" ".join(subdirectories)}
shell cd $d
log log.lammps
include {str(lammps_input)}""")


def write_universe(path, subdirectories, lammps_input, name="in.universe"):
    """
    Write the universe file for LAMMPS simulation.

    Args:
        path (str): The path to the directory where the universe file will be created.
        subdirectories (list): A list of subdirectories to be included in the universe file.
        lammps_input (str): Name of the lammps input file to use.
    """

    with open(os.path.join(path, name), "w") as p:
        p.write(f"""variable d universe {" ".join(reversed(subdirectories))}
shell cd $d
log log.lammps
include {str(lammps_input)}
clear
shell cd {str(path)}
next d
jump {name}""")


class LammpsData:
    """Class to read and write LAMMPS data files."""

    def __init__(self, filename):
        """Input is path to the LAMMPS data file."""
        self.filename = filename
        self.read()

    def read(self, filename=None):
        """Reads the LAMMPS data file and stores the information, including optional PairIJ Coeffs & Velocities."""
        self.header = ""
        self.num_atoms = 0
        self.num_bonds = 0
        self.atom_types = 0
        self.bond_types = 0
        self.xlo, self.xhi = 0.0, 0.0
        self.ylo, self.yhi = 0.0, 0.0
        self.zlo, self.zhi = 0.0, 0.0
        self.masses = {}
        self.atoms = []
        self.bonds = []

        # Dictionaries for optional sections
        # If "PairIJ Coeffs" is present, we'll store them as pair_coeffs[(type_i, type_j)] = [param1, param2, ...]
        self.pair_coeffs = {}
        # If "Velocities" is present, we'll store them as velocities[atom_id] = (vx, vy, vz)
        self.velocities = {}

        if filename is None:
            filename = self.filename

        with open(filename, "r") as file:
            lines = file.readlines()

        # 1. Parse basic info
        self.header = lines[0].rstrip("\n")
        self.num_atoms = int(lines[2].split()[0])
        self.atom_types = int(lines[3].split()[0])
        self.num_bonds = int(lines[4].split()[0])
        self.bond_types = int(lines[5].split()[0])

        # 2. Box bounds
        self.xlo, self.xhi = map(float, lines[7].split()[:2])
        self.ylo, self.yhi = map(float, lines[8].split()[:2])
        self.zlo, self.zhi = map(float, lines[9].split()[:2])

        # We'll search for line indices for "Masses\n", "PairIJ Coeffs\n", "Atoms\n", "Velocities\n", "Bonds\n"
        # to decide where each section starts and ends.
        def find_section(name):
            try:
                return lines.index(name + "\n")
            except ValueError:
                return None  # Section not found

        def find_section(name):
            for i, line in enumerate(lines):
                if line.startswith(name):
                    return i
            return None

        idx_masses = find_section("Masses")
        idx_pairij = find_section("PairIJ Coeffs")
        idx_atoms = find_section("Atoms")
        idx_velocities = find_section("Velocities")
        idx_bonds = find_section("Bonds")

        # 3. Parse Masses
        # In this template, "Masses" is followed by a blank line, so the actual lines start 2 after index
        # The next section might be "PairIJ Coeffs" or "Atoms"
        if idx_masses is not None:
            start_mass = idx_masses + 2
            # The section ends just before the next known section
            potential_stops = []
            if idx_pairij is not None:
                potential_stops.append(idx_pairij - 1)
            if idx_atoms is not None:
                potential_stops.append(idx_atoms - 1)
            end_mass = min(potential_stops) if potential_stops else start_mass
            for line in lines[start_mass:end_mass]:
                if not line.strip():
                    break
                atom_id, mass = line.split()[:2]
                self.masses[int(atom_id)] = float(mass)

        # 4. Parse PairIJ Coeffs (if present)
        # Similar logic: "PairIJ Coeffs" lines start 2 after idx_pairij, end right before "Atoms" or next known section
        if idx_pairij is not None:
            start_pairij = idx_pairij + 2
            # It ends just before "Atoms"
            end_pairij = idx_atoms - 1 if idx_atoms is not None else start_pairij

            # Example line for PairIJ Coeffs might look like: "1 1 0.2 2.5"
            # We'll parse the first 2 as types, the rest as parameters
            for line in lines[start_pairij:end_pairij]:
                if not line.strip():
                    break
                parts = line.split()
                type_i = int(parts[0])
                type_j = int(parts[1])
                params = list(map(float, parts[2:]))
                self.pair_coeffs[(type_i, type_j)] = params

        # 5. Parse Atoms
        if idx_atoms is not None:
            start_atoms = idx_atoms + 2
            # End is just before "Velocities" or "Bonds"
            potential_stops = []
            if idx_velocities is not None:
                potential_stops.append(idx_velocities - 1)
            if idx_bonds is not None:
                potential_stops.append(idx_bonds - 1)
            end_atoms = min(potential_stops) if potential_stops else start_atoms

            for line in lines[start_atoms:end_atoms]:
                if not line.strip():
                    break
                atom_data = list(map(float, line.split()))
                self.atoms.append(
                    {
                        "id": int(atom_data[0]),
                        "type": int(atom_data[1]),
                        "mol": int(atom_data[2]),
                        "charge": atom_data[3],
                        "x": atom_data[4],
                        "y": atom_data[5],
                        "z": atom_data[6],
                        # vx, vy, vz stored as ints in original code, but usually they are floats
                        # We'll keep them as ints for consistency with the original code
                        "vx": int(atom_data[7]),
                        "vy": int(atom_data[8]),
                        "vz": int(atom_data[9]),
                    }
                )

        # 6. Parse Velocities (if present)
        if idx_velocities is not None:
            start_vel = idx_velocities + 2
            # Ends just before "Bonds"
            stop_vel = idx_bonds - 1 if idx_bonds is not None else start_vel
            # Example line: "atom_id vx vy vz"
            for line in lines[start_vel:stop_vel]:
                if not line.strip():
                    break
                parts = line.split()
                atom_id = int(parts[0])
                vx = float(parts[1])
                vy = float(parts[2])
                vz = float(parts[3])
                self.velocities[atom_id] = (vx, vy, vz)

        # 7. Parse Bonds
        if idx_bonds is not None:
            start_bonds = idx_bonds + 2
            for line in lines[start_bonds:]:
                if not line.strip():
                    break
                bond_data = list(map(int, line.split()))
                self.bonds.append(
                    {
                        "id": bond_data[0],
                        "type": bond_data[1],
                        "atom1": bond_data[2],
                        "atom2": bond_data[3],
                    }
                )

    def modify_atom(self, atom_id, **kwargs):
        """Modify any attribute of a specific atom."""
        for atom in self.atoms:
            if atom["id"] == atom_id:
                for key, value in kwargs.items():
                    if key in atom:
                        atom[key] = value
                    else:
                        print(f"Invalid attribute: {key}")
                return
        print(f"Atom with ID {atom_id} not found.")

    def write(self, filename=None):
        """Writes the data back to a LAMMPS data file, preserving optional PairIJ Coeffs & Velocities if present."""
        if filename is None:
            filename = self.filename

        with open(filename, "w") as file:
            # 1. Header
            file.write(self.header + "\n\n")

            # 2. Basic info
            file.write(f"{self.num_atoms} atoms\n")
            file.write(f"{self.atom_types} atom types\n")
            file.write(f"{self.num_bonds} bonds\n")
            file.write(f"{self.bond_types} bond types\n\n")

            # 3. Box bounds
            file.write(f"{self.xlo:<18.15e}   {self.xhi:>18.15e} xlo xhi\n")
            file.write(f"{self.ylo:<18.15e}   {self.yhi:>18.15e} ylo yhi\n")
            file.write(f"{self.zlo:<18.15e}   {self.zhi:>18.15e} zlo zhi\n\n")

            # 4. Masses
            file.write("Masses\n\n")
            for atom_type, mass in self.masses.items():
                file.write(f"{atom_type}  {mass:>9.5f}\n")

            # 5. PairIJ Coeffs (if any)
            if self.pair_coeffs:
                file.write("\nPairIJ Coeffs\n\n")
                for (type_i, type_j), params in self.pair_coeffs.items():
                    # Convert params to string
                    param_str = " ".join(f"{p:g}" for p in params)
                    file.write(f"{type_i} {type_j} {param_str}\n")

            # 6. Atoms
            file.write("\nAtoms\n\n")
            for atom in self.atoms:
                file.write(
                    f"{atom['id']:<5} {atom['type']:>5} {atom['mol']:>4} "
                    f"{atom['charge']:>8.5f} {atom['x']:>15.10f} {atom['y']:>15.10f} "
                    f"{atom['z']:>15.10f} {atom['vx']} {atom['vy']} {atom['vz']}\n"
                )

            # 7. Velocities (if any)
            if self.velocities:
                file.write("\nVelocities\n\n")
                for atom_id, (vx, vy, vz) in self.velocities.items():
                    file.write(f"{atom_id} {vx} {vy} {vz}\n")

            # 8. Bonds
            file.write("\nBonds\n\n")
            for bond in self.bonds:
                file.write(
                    f"{bond['id']:<2} {bond['type']} {bond['atom1']} {bond['atom2']}\n"
                )


class LammpsSettings:
    """Class to read and write LAMMPS settings files."""

    def __init__(self, filename):
        """Input is path to the LAMMPS settings file."""
        self.filename = filename
        self.non_bonded = {}
        self.bonds = {}
        self.angles = {}
        self.dihedrals = {}

        self.read()

    def read(self, filename=None):
        """Reads the LAMMPS settings file."""
        if filename is None:
            filename = self.filename

        with open(filename, "r") as file:
            lines = file.readlines()
        self.header = lines[0].strip()

        for line in lines:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            if line.startswith("pair_coeff"):
                self._parse_non_bonded(line)
            elif line.startswith("bond_coeff"):
                self._parse_bonds(line)
            elif line.startswith("angle_coeff"):
                self._parse_angles(line)
            elif line.startswith("dihedral_coeff"):
                self._parse_dihedrals(line)

    def _parse_non_bonded(self, line):
        """Parses non-bonded coefficients from a line."""
        parts = line.split()
        # print(parts)
        # print(self.non_bonded)
        if (int(parts[1]), int(parts[2])) not in self.non_bonded:
            self.non_bonded[(int(parts[1]), int(parts[2]))] = {
                "style": parts[3],
                "params": list(map(float, parts[4:])),
            }

    def _parse_bonds(self, line):
        """Parses bond coefficients from a line."""
        parts = line.split()
        if parts[1] not in self.bonds:
            self.bonds[parts[1]] = {
                "style": parts[2],
                "params": list(map(float, parts[3:])),
            }

    def _parse_angles(self, line):
        """Parses angle coefficients from a line."""
        parts = line.split()
        if parts[1] not in self.angles:
            self.angles[parts[1]] = {
                "style": parts[2],
                "params": list(map(float, parts[3:])),
            }

    def _parse_dihedrals(self, line):
        """Parses dihedral coefficients from a line."""
        parts = line.split()
        if parts[1] not in self.dihedrals:
            self.dihedrals[parts[1]] = {
                "params": list(map(float, parts[2:])),
            }

    def write(self, filename=None):
        """Writes the settings back to a LAMMPS settings file."""
        if filename is None:
            filename = self.filename

        with open(filename, "w") as file:
            file.write(self.header + "\n\n")

            # Write non-bonded coefficients first
            file.write("# ~~~~~Non-bonded~~~~~\n# e.g., pair_coeff type eps sig\n")
            for (type1, type2), data in self.non_bonded.items():
                params_str = " ".join(f"{p:10.5f}" for p in data["params"])
                file.write(
                    f"pair_coeff {type1:<5} {type2:<5} {data['style']:<15} {params_str}\n"
                )

            # Write bond coefficients
            file.write("\n# ~~~~~Bonds~~~~~\n# e.g., bond_coeff type k  r0\n")
            for bond_type, data in self.bonds.items():
                params_str = " ".join(f"{p:10.5f}" for p in data["params"])
                file.write(
                    f"bond_coeff {bond_type:<5} {data['style']:<15} {params_str}\n"
                )

            # Write angle coefficients
            file.write(
                "\n# ~~~~~Angles~~~~~\n# e.g., angle_coeff type k(kcal/mol/rad**2)  theta_0 (degrees)\n"
            )
            for angle_type, data in self.angles.items():
                params_str = " ".join(f"{p:10.5f}" for p in data["params"])
                file.write(
                    f"angle_coeff {angle_type:<5} {data['style']:<15} {params_str}\n"
                )

            # Write dihedral coefficients
            file.write(
                "\n# ~~~~~Dihedrals~~~~~\n# e.g., dihedral_coeff type k(kcal/mol) n d weighting_factor\n"
            )
            for dihedral_type, data in self.dihedrals.items():
                params_str = " ".join(f"{p:10.5f}" for p in data["params"][:-2])
                # For harmonic style dihedral, the last two parameters must be integers
                # If we want to use a different dihedral style, we'll need to change this
                params_str += " ".join(f"{int(p):5d}" for p in data["params"][-2:])
                file.write(f"dihedral_coeff {dihedral_type:<5} {params_str}\n")

    def modify_non_bonded(self, type1, type2, style, params):
        """Modify non-bonded coefficients."""
        self.non_bonded[(type1, type2)] = {
            "style": style,
            "params": params,
        }
