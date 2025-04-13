"""Modified from ASE calculator for the LAMMPS classical MD code"""
# lammps.py (2011/03/29)
#
# Copyright (C) 2009 - 2011 Joerg Meyer, joerg.meyer@ch.tum.de
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this file; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301
# USA or see <http://www.gnu.org/licenses/>.

import os
import shlex
import shutil
import subprocess
import warnings
from re import IGNORECASE
from re import compile as re_compile
from tempfile import NamedTemporaryFile, mkdtemp
from tempfile import mktemp as uns_mktemp
from threading import Thread
from typing import Any

import numpy as np
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.lammps import CALCULATION_END_MARK, Prism, convert, write_lammps_in
from ase.data import atomic_masses, chemical_symbols
from ase.io.lammpsdata import write_lammps_data
from ase.io.lammpsrun import read_lammps_dump

__all__ = ["LAMMPS"]


class LAMMPS(Calculator):
    """LAMMPS (https://lammps.sandia.gov/) calculator

    The environment variable :envvar:`ASE_LAMMPSRUN_COMMAND` must be defined to
    tell ASE how to call a LAMMPS binary. This should contain the path to the
    lammps binary, or more generally, a command line possibly also including an
    MPI-launcher command.

    For example (in a Bourne-shell compatible environment):

    .. code-block:: bash

        export ASE_LAMMPSRUN_COMMAND=/path/to/lmp_binary

    or possibly something similar to

    .. code-block:: bash

        export ASE_LAMMPSRUN_COMMAND="/path/to/mpirun --np 4 lmp_binary"

    Parameters
    ----------
    files : list[str]
        List of files needed by LAMMPS. Typically a list of potential files.
    parameters : dict[str, Any]
        Dictionary of settings to be passed into the input file for calculation.
    specorder : list[str]
        Within LAAMPS, atoms are identified by an integer value starting from 1.
        This variable allows the user to define the order of the indices
        assigned to the atoms in the calculation, with the default
        if not given being alphabetical
    keep_tmp_files : bool, default: False
        Retain any temporary files created. Mostly useful for debugging.
    tmp_dir : str, default: None
        path/dirname (default None -> create automatically).
        Explicitly control where the calculator object should create
        its files. Using this option implies 'keep_tmp_files=True'.
    no_data_file : bool, default: False
        Controls whether an explicit data file will be used for feeding
        atom coordinates into lammps. Enable it to lessen the pressure on
        the (tmp) file system. THIS OPTION MIGHT BE UNRELIABLE FOR CERTAIN
        CORNER CASES (however, if it fails, you will notice...).
    keep_alive : bool, default: True
        When using LAMMPS as a spawned subprocess, keep the subprocess
        alive (but idling when unused) along with the calculator object.
    always_triclinic : bool, default: False
        Force LAMMPS to treat the cell as tilted, even if the cell is not
        tilted, by printing ``xy``, ``xz``, ``yz`` in the data file.
    reduce_cell : bool, default: False
        If True, reduce cell to have shorter lattice vectors.
    write_velocities : bool, default: False
        If True, forward ASE velocities to LAMMPS.
    verbose: bool, default: False
        If True, print additional debugging output to STDOUT.

    Examples:
    --------
    Provided that the respective potential file is in the working directory,
    one can simply run (note that LAMMPS needs to be compiled to work with EAM
    potentials)

    ::

        from ase import Atom, Atoms
        from ase.build import bulk
        from ase.calculators.lammpsrun import LAMMPS

        parameters = {"pair_style": "eam/alloy", "pair_coeff": ["* * NiAlH_jea.eam.alloy H Ni"]}

        files = ["NiAlH_jea.eam.alloy"]

        Ni = bulk("Ni", cubic=True)
        H = Atom("H", position=Ni.cell.diagonal() / 2)
        NiH = Ni + H

        lammps = LAMMPS(parameters=parameters, files=files)

        NiH.calc = lammps
        print("Energy ", NiH.get_potential_energy())
    """

    name = "lammpsrun"
    implemented_properties = ["energy", "free_energy", "forces", "stress", "energies"]

    # parameters to choose options in LAMMPSRUN
    ase_parameters: dict[str, Any] = dict(
        specorder=None,
        atorder=True,
        always_triclinic=False,
        reduce_cell=False,
        keep_alive=True,
        keep_tmp_files=False,
        no_data_file=False,
        tmp_dir=None,
        files=[],  # usually contains potential parameters
        verbose=False,
        write_velocities=False,
        binary_dump=True,  # bool - use binary dump files (full
        # precision but long long ids are casted to
        # double)
        lammps_options="-echo log -screen none -log /dev/stdout",
        trajectory_out=None,  # file object, if is not None the
        # trajectory will be saved in it
    )

    # parameters forwarded to LAMMPS
    lammps_parameters = dict(
        boundary=None,  # bounadry conditions styles
        units="metal",  # str - Which units used; some potentials
        # require certain units
        atom_style="atomic",
        special_bonds=None,
        # potential informations
        pair_style="lj/cut 2.5",
        pair_coeff=["* * 1 1"],
        masses=None,
        pair_modify=None,
        # variables controlling the output
        thermo_args=[
            "step",
            "temp",
            "press",
            "cpu",
            "pxx",
            "pyy",
            "pzz",
            "pxy",
            "pxz",
            "pyz",
            "ke",
            "pe",
            "etotal",
            "vol",
            "lx",
            "ly",
            "lz",
            "atoms",
        ],
        dump_properties=[
            "id",
            "type",
            "x",
            "y",
            "z",
            "vx",
            "vy",
            "vz",
            "fx",
            "fy",
            "fz",
        ],
        dump_period=1,  # period of system snapshot saving (in MD steps)
    )

    default_parameters = dict(ase_parameters, **lammps_parameters)

    def __init__(self, label="lammps", **kwargs):
        super().__init__(label=label, **kwargs)

        self.prism = None
        self.calls = 0
        self.forces = None
        # thermo_content contains data "written by" thermo_style.
        # It is a list of dictionaries, each dict (one for each line
        # printed by thermo_style) contains a mapping between each
        # custom_thermo_args-argument and the corresponding
        # value as printed by lammps. thermo_content will be
        # re-populated by the read_log method.
        self.thermo_content = []

        if self.parameters["tmp_dir"] is not None:
            # If tmp_dir is pointing somewhere, don't remove stuff!
            self.parameters["keep_tmp_files"] = True
        self._lmp_handle = None  # To handle the lmp process

        if self.parameters["tmp_dir"] is None:
            self.parameters["tmp_dir"] = mkdtemp(prefix="LAMMPS-")
        else:
            self.parameters["tmp_dir"] = os.path.realpath(self.parameters["tmp_dir"])
            if not os.path.isdir(self.parameters["tmp_dir"]):
                os.mkdir(self.parameters["tmp_dir"], 0o755)

        for f in self.parameters["files"]:
            shutil.copy(f, os.path.join(self.parameters["tmp_dir"], os.path.basename(f)))

    def get_lammps_command(self):
        cmd = self.parameters.get("command")

        if cmd is None:
            from ase.config import cfg

            envvar = f"ASE_{self.name.upper()}_COMMAND"
            cmd = cfg.get(envvar)

        if cmd is None:
            # TODO deprecate and remove guesswork
            cmd = "lammps"

        opts = self.parameters.get("lammps_options")

        if opts is not None:
            cmd = f"{cmd} {opts}"

        return cmd

    def clean(self, force=False):
        self._lmp_end()

        if not self.parameters["keep_tmp_files"] or force:
            shutil.rmtree(self.parameters["tmp_dir"])

    def check_state(self, atoms, tol=1.0e-10):
        # Transforming the unit cell to conform to LAMMPS' convention for
        # orientation (c.f. https://lammps.sandia.gov/doc/Howto_triclinic.html)
        # results in some precision loss, so we use bit larger tolerance than
        # machine precision here.  Note that there can also be precision loss
        # related to how many significant digits are specified for things in
        # the LAMMPS input file.
        return Calculator.check_state(self, atoms, tol)

    def calculate(self, atoms=None, properties=None, system_changes=None):
        if properties is None:
            properties = self.implemented_properties
        if system_changes is None:
            system_changes = all_changes
        Calculator.calculate(self, atoms, properties, system_changes)
        self.run()

    def _lmp_alive(self):
        # Return True if this calculator is currently handling a running
        # lammps process
        return self._lmp_handle and not isinstance(self._lmp_handle.poll(), int)

    def _lmp_end(self):
        # Close lammps input and wait for lammps to end. Return process
        # return value
        if self._lmp_alive():
            # !TODO: handle lammps error codes
            try:
                self._lmp_handle.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                self._lmp_handle.kill()
                self._lmp_handle.communicate()
            err = self._lmp_handle.poll()
            assert err is not None
            return err

    def set_missing_parameters(self):
        """Verify that all necessary variables are set."""
        symbols = self.atoms.get_chemical_symbols()
        # If unspecified default to atom types in alphabetic order
        if not self.parameters.get("specorder"):
            self.parameters["specorder"] = sorted(set(symbols))

        # !TODO: handle cases were setting masses actual lead to errors
        if not self.parameters.get("masses"):
            self.parameters["masses"] = []
            for type_id, specie in enumerate(self.parameters["specorder"]):
                mass = atomic_masses[chemical_symbols.index(specie)]
                self.parameters["masses"] += [f"{type_id + 1:d} {mass:f}"]

        # set boundary condtions
        if not self.parameters.get("boundary"):
            b_str = " ".join(["fp"[int(x)] for x in self.atoms.pbc])
            self.parameters["boundary"] = b_str

    def run(self, set_atoms=False):
        # !TODO: split this function
        """Method which explicitly runs LAMMPS."""
        pbc = self.atoms.get_pbc()
        if all(pbc):
            cell = self.atoms.get_cell()
        elif not any(pbc):
            # large enough cell for non-periodic calculation -
            # LAMMPS shrink-wraps automatically via input command
            #       "periodic s s s"
            # below
            cell = 2 * np.max(np.abs(self.atoms.get_positions())) * np.eye(3)
        else:
            warnings.warn(
                "semi-periodic ASE cell detected - translation "
                + "to proper LAMMPS input cell might fail"
            )
            cell = self.atoms.get_cell()
        self.prism = Prism(cell)

        self.set_missing_parameters()
        self.calls += 1

        # change into subdirectory for LAMMPS calculations
        tempdir = self.parameters["tmp_dir"]

        # setup file names for LAMMPS calculation
        label = f"{self.label}{self.calls:>06}"
        lammps_in = uns_mktemp(prefix="in_" + label, dir=tempdir)
        lammps_log = uns_mktemp(prefix="log_" + label, dir=tempdir)
        lammps_trj_fd = NamedTemporaryFile(
            prefix="trj_" + label,
            suffix=(".bin" if self.parameters["binary_dump"] else ""),
            dir=tempdir,
            delete=(not self.parameters["keep_tmp_files"]),
        )
        lammps_trj = lammps_trj_fd.name
        if self.parameters["no_data_file"]:
            lammps_data = None
        else:
            lammps_data_fd = NamedTemporaryFile(
                prefix="data_" + label,
                dir=tempdir,
                delete=(not self.parameters["keep_tmp_files"]),
                mode="w",
                encoding="utf-8",
            )
            write_lammps_data(
                lammps_data_fd,
                self.atoms,
                specorder=self.parameters["specorder"],
                force_skew=self.parameters["always_triclinic"],
                reduce_cell=self.parameters["reduce_cell"],
                velocities=self.parameters["write_velocities"],
                prismobj=self.prism,
                units=self.parameters["units"],
                atom_style=self.parameters["atom_style"],
            )
            lammps_data = lammps_data_fd.name
            lammps_data_fd.flush()

        # see to it that LAMMPS is started
        if not self._lmp_alive():
            command = self.get_lammps_command()
            # Attempt to (re)start lammps
            self._lmp_handle = subprocess.Popen(
                shlex.split(command, posix=(os.name == "posix")),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                encoding="utf-8",
            )
        lmp_handle = self._lmp_handle

        # Create thread reading lammps stdout (for reference, if requested,
        # also create lammps_log, although it is never used)
        if self.parameters["keep_tmp_files"]:
            lammps_log_fd = open(lammps_log, "w")
            fd = SpecialTee(lmp_handle.stdout, lammps_log_fd)
        else:
            fd = lmp_handle.stdout
        thr_read_log = Thread(target=self.read_lammps_log, args=(fd,))
        thr_read_log.start()

        # write LAMMPS input (for reference, also create the file lammps_in,
        # although it is never used)
        if self.parameters["keep_tmp_files"]:
            lammps_in_fd = open(lammps_in, "w")
            fd = SpecialTee(lmp_handle.stdin, lammps_in_fd)
        else:
            fd = lmp_handle.stdin
        write_lammps_in(
            lammps_in=fd,
            parameters=self.parameters,
            atoms=self.atoms,
            prismobj=self.prism,
            lammps_trj=lammps_trj,
            lammps_data=lammps_data,
        )

        if self.parameters["keep_tmp_files"]:
            lammps_in_fd.close()

        # Wait for log output to be read (i.e., for LAMMPS to finish)
        # and close the log file if there is one
        thr_read_log.join()
        if self.parameters["keep_tmp_files"]:
            lammps_log_fd.close()

        if not self.parameters["keep_alive"]:
            self._lmp_end()

        exitcode = lmp_handle.poll()
        if exitcode and exitcode != 0:
            raise RuntimeError(f"LAMMPS exited in {tempdir} with exit code: {exitcode}.")

        # A few sanity checks
        if len(self.thermo_content) == 0:
            raise RuntimeError("Failed to retrieve any thermo_style-output")
        if int(self.thermo_content[-1]["atoms"]) != len(self.atoms):
            # This obviously shouldn't happen, but if prism.fold_...() fails,
            # it could
            raise RuntimeError("Atoms have gone missing")

        trj_atoms = read_lammps_dump(
            infileobj=lammps_trj,
            order=self.parameters["atorder"],
            index=-1,
            prismobj=self.prism,
            specorder=self.parameters["specorder"],
        )

        if set_atoms:
            self.atoms = trj_atoms.copy()

        self.forces = trj_atoms.get_forces()
        # !TODO: trj_atoms is only the last snapshot of the system; Is it
        #        desirable to save also the inbetween steps?
        if self.parameters["trajectory_out"] is not None:
            # !TODO: is it advisable to create here temporary atoms-objects
            self.trajectory_out.write(trj_atoms)

        tc = self.thermo_content[-1]
        self.results["energy"] = convert(tc["pe"], "energy", self.parameters["units"], "ASE")
        self.results["free_energy"] = self.results["energy"]
        self.results["forces"] = convert(
            self.forces.copy(), "force", self.parameters["units"], "ASE"
        )
        stress = np.array([-tc[i] for i in ("pxx", "pyy", "pzz", "pyz", "pxz", "pxy")])

        # We need to apply the Lammps rotation stuff to the stress:
        xx, yy, zz, yz, xz, xy = stress
        stress_tensor = np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])
        stress_atoms = self.prism.tensor2_to_ase(stress_tensor)
        stress_atoms = stress_atoms[[0, 1, 2, 1, 0, 0], [0, 1, 2, 2, 2, 1]]
        stress = stress_atoms

        self.results["stress"] = convert(stress, "pressure", self.parameters["units"], "ASE")

        lammps_trj_fd.close()
        if not self.parameters["no_data_file"]:
            lammps_data_fd.close()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self._lmp_end()

    def read_lammps_log(self, fileobj):
        # !TODO: somehow communicate 'thermo_content' explicitly
        """Method which reads a LAMMPS output log file."""
        # read_log depends on that the first (three) thermo_style custom args
        # can be capitalized and matched against the log output. I.e.
        # don't use e.g. 'ke' or 'cpu' which are labeled KinEng and CPU.
        mark_re = r"^\s*" + r"\s+".join(
            [x.capitalize() for x in self.parameters["thermo_args"][0:3]]
        )
        _custom_thermo_mark = re_compile(mark_re)

        # !TODO: regex-magic necessary?
        # Match something which can be converted to a float
        f_re = r"([+-]?(?:(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?|nan|inf))"
        n_args = len(self.parameters["thermo_args"])
        # Create a re matching exactly N white space separated floatish things
        _custom_thermo_re = re_compile(
            r"^\s*" + r"\s+".join([f_re] * n_args) + r"\s*$", flags=IGNORECASE
        )

        thermo_content = []
        line = fileobj.readline()
        while line and line.strip() != CALCULATION_END_MARK:
            # check error
            if "ERROR:" in line:
                raise RuntimeError(f"LAMMPS exits with error message: {line}")

            # get thermo output
            if _custom_thermo_mark.match(line):
                while True:
                    line = fileobj.readline()
                    if "WARNING:" in line:
                        continue

                    bool_match = _custom_thermo_re.match(line)
                    if not bool_match:
                        break

                    # create a dictionary between each of the
                    # thermo_style args and it's corresponding value
                    thermo_content.append(
                        dict(
                            zip(
                                self.parameters["thermo_args"],
                                map(float, bool_match.groups()),
                                strict=False,
                            )
                        )
                    )
            else:
                line = fileobj.readline()

        self.thermo_content = thermo_content


class SpecialTee:
    """A special purpose, with limited applicability, tee-like thing.

    A subset of stuff read from, or written to, orig_fd,
    is also written to out_fd.
    It is used by the lammps calculator for creating file-logs of stuff
    read from, or written to, stdin and stdout, respectively.
    """

    def __init__(self, orig_fd, out_fd):
        self._orig_fd = orig_fd
        self._out_fd = out_fd
        self.name = orig_fd.name

    def write(self, data):
        self._orig_fd.write(data)
        self._out_fd.write(data)
        self.flush()

    def read(self, *args, **kwargs):
        data = self._orig_fd.read(*args, **kwargs)
        self._out_fd.write(data)
        return data

    def readline(self, *args, **kwargs):
        data = self._orig_fd.readline(*args, **kwargs)
        self._out_fd.write(data)
        return data

    def readlines(self, *args, **kwargs):
        data = self._orig_fd.readlines(*args, **kwargs)
        self._out_fd.write("".join(data))
        return data

    def flush(self):
        self._orig_fd.flush()
        self._out_fd.flush()
