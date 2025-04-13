###################################################################################
### Adapted from CHGNet (https://github.com/CederGroupHub/chgnet/blob/main/chgnet/data/dataset.py)
###################################################################################
from __future__ import annotations

import functools
import os
import random
import warnings
from typing import TYPE_CHECKING

import torch
from chgnet import utils
from chgnet.graph import CrystalGraph, CrystalGraphConverter
from pymatgen.core.structure import Structure
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

if TYPE_CHECKING:
    from chgnet import TrainTask

warnings.filterwarnings("ignore")
TORCH_DTYPE = torch.float32


class StructureJsonData(Dataset):
    """Read structure and targets from a JSON file.
    This class is used to load the MPtrj dataset.
    """

    def __init__(
        self,
        data: str | dict,
        *,
        graph_converter: CrystalGraphConverter | None = CrystalGraphConverter(
            atom_graph_cutoff=6, bond_graph_cutoff=3
        ),
        targets: TrainTask = "efsm",
        energy_key: str = "energy_per_atom",
        force_key: str = "force",
        stress_key: str = "stress",
        magmom_key: str = "magmom",
        shuffle: bool = True,
    ) -> None:
        """Initialize the dataset by reading JSON files.

        Args:
            data (str | dict): file path or dir name that contain all the JSONs
            graph_converter (CrystalGraphConverter): Converts pymatgen.core.Structure
                to CrystalGraph object.
            targets ("ef" | "efs" | "efm" | "efsm"): The training targets.
                Default = "efsm"
            energy_key (str, optional): the key of energy in the labels.
                Default = "energy_per_atom"
            force_key (str, optional): the key of force in the labels.
                Default = "force"
            stress_key (str, optional): the key of stress in the labels.
                Default = "stress"
            magmom_key (str, optional): the key of magmom in the labels.
                Default = "magmom"
            shuffle (bool): whether to shuffle the sequence of dataset
                Default = True
        """
        if isinstance(data, str):
            self.data = {}
            if os.path.isdir(data):
                for json_path in os.listdir(data):
                    if json_path.endswith(".json"):
                        print(f"Importing: {json_path}")
                        self.data.update(utils.read_json(os.path.join(data, json_path)))
            else:
                print(f"Importing: {data}")
                self.data.update(utils.read_json(data))
        elif isinstance(data, dict):
            self.data = data
        else:
            raise TypeError(f"data must be JSON path or dictionary, got {type(data)}")

        self.keys = [(mp_id, graph_id) for mp_id, dct in self.data.items() for graph_id in dct]
        if shuffle:
            random.shuffle(self.keys)
        print(f"{len(self.data)} MP IDs, {len(self)} structures imported")
        self.graph_converter = graph_converter
        self.energy_key = energy_key
        self.force_key = force_key
        self.stress_key = stress_key
        self.magmom_key = magmom_key
        self.targets = targets
        self.failed_idx: list[int] = []
        self.failed_graph_id: dict[str, str] = {}

    def __len__(self) -> int:
        """Get the number of structures with targets in the dataset."""
        return len(self.keys)

    @functools.cache  # Cache loaded structures
    def __getitem__(self, idx: int) -> tuple[CrystalGraph, dict[str, Tensor]]:
        """Get one item in the dataset.

        Returns:
            crystal_graph (CrystalGraph): graph of the crystal structure
            targets (dict): dictionary of targets. i.e. energy, force, stress, magmom
        """
        if idx not in self.failed_idx:
            mp_id, graph_id = self.keys[idx]
            try:
                struct = Structure.from_dict(self.data[mp_id][graph_id]["structure"])
                if self.graph_converter is not None:
                    end_struct = self.graph_converter(struct, graph_id=graph_id, mp_id=mp_id)
                else:
                    end_struct = struct
                    end_struct.properties["mp_id"] = mp_id
                    end_struct.properties["graph_id"] = graph_id

                targets = {}
                for key in self.targets:
                    if key == "e":
                        energy = self.data[mp_id][graph_id][self.energy_key]
                        targets["e"] = torch.tensor(energy, dtype=TORCH_DTYPE)
                    elif key == "f":
                        force = self.data[mp_id][graph_id][self.force_key]
                        targets["f"] = torch.tensor(force, dtype=TORCH_DTYPE)
                    elif key == "s":
                        stress = self.data[mp_id][graph_id][self.stress_key]
                        # Convert VASP stress
                        targets["s"] = torch.tensor(stress, dtype=TORCH_DTYPE) * (-0.1)
                    elif key == "m":
                        mag = self.data[mp_id][graph_id][self.magmom_key]
                        # use absolute value for magnetic moments
                        if mag is None:
                            targets["m"] = None
                        else:
                            targets["m"] = torch.abs(torch.tensor(mag, dtype=TORCH_DTYPE))

                return end_struct, targets

            # Omit structures with isolated atoms. Return another randomly selected
            # structure
            except Exception:
                structure = Structure.from_dict(self.data[mp_id][graph_id]["structure"])
                self.failed_graph_id[graph_id] = structure.composition.formula
                self.failed_idx.append(idx)
                idx = random.randint(0, len(self) - 1)
                return self.__getitem__(idx)
        else:
            idx = random.randint(0, len(self) - 1)
            return self.__getitem__(idx)

    def get_train_val_test_loader(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        *,
        train_key: list[str] | None = None,
        val_key: list[str] | None = None,
        test_key: list[str] | None = None,
        batch_size=32,
        num_workers=0,
        pin_memory=True,
    ) -> tuple[DataLoader, DataLoader, DataLoader]:
        """Partition the Dataset using materials id,
        randomly select the train_keys, val_keys, test_keys by train val test ratio,
        or use pre-defined train_keys, val_keys, and test_keys to create train, val,
        test loaders.

        Args:
            train_ratio (float): The ratio of the dataset to use for training
                Default = 0.8
            val_ratio (float): The ratio of the dataset to use for validation
                Default: 0.1
            train_key (List(str), optional): a list of mp_ids for train set
            val_key (List(str), optional): a list of mp_ids for val set
            test_key (List(str), optional): a list of mp_ids for test set
            batch_size (int): batch size
                Default = 32
            num_workers (int): The number of worker processes for loading the data
                see torch Dataloader documentation for more info
                Default = 0
            pin_memory (bool): Whether to pin the memory of the data loaders
                Default: True

        Returns:
            train_loader, val_loader, test_loader
        """
        train_data, val_data, test_data = {}, {}, {}
        if train_key is None:
            mp_ids = list(self.data)
            random.shuffle(mp_ids)
            n_train = int(train_ratio * len(mp_ids))
            n_val = int(val_ratio * len(mp_ids))
            train_key = mp_ids[:n_train]
            val_key = mp_ids[n_train : n_train + n_val]
            test_key = mp_ids[n_train + n_val :]
        for mp_id in train_key:
            train_data[mp_id] = self.data.pop(mp_id)
        train_dataset = StructureJsonData(
            data=train_data,
            graph_converter=self.graph_converter,
            targets=self.targets,
            energy_key=self.energy_key,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collate_graphs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        for mp_id in val_key:
            val_data[mp_id] = self.data.pop(mp_id)
        val_dataset = StructureJsonData(
            data=val_data,
            graph_converter=self.graph_converter,
            targets=self.targets,
            energy_key=self.energy_key,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            collate_fn=collate_graphs,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        )
        if test_key is not None:
            for mp_id in test_key:
                test_data[mp_id] = self.data.pop(mp_id)
            test_dataset = StructureJsonData(
                data=test_data,
                graph_converter=self.graph_converter,
                targets=self.targets,
                energy_key=self.energy_key,
            )
            test_loader = DataLoader(
                test_dataset,
                batch_size=batch_size,
                collate_fn=collate_graphs,
                num_workers=num_workers,
                pin_memory=pin_memory,
            )
        else:
            test_loader = None
        return train_loader, val_loader, test_loader


def collate_graphs(batch_data: list) -> tuple[list[CrystalGraph], dict[str, Tensor]]:
    """Collate of list of (graph, target) into batch data.

    Args:
        batch_data (list): list of (graph, target(dict))

    Returns:
        graphs (List): a list of graphs
        targets (Dict): dictionary of targets, where key and values are:
            e (Tensor): energies of the structures [batch_size]
            f (Tensor): forces of the structures [n_batch_atoms, 3]
            s (Tensor): stresses of the structures [3*batch_size, 3]
            m (Tensor): magmom of the structures [n_batch_atoms]
    """
    graphs = [graph for graph, _ in batch_data]
    all_targets = {key: [] for key in batch_data[0][1]}
    all_targets["e"] = torch.tensor([targets["e"] for _, targets in batch_data], dtype=TORCH_DTYPE)

    for _, targets in batch_data:
        for target, value in targets.items():
            if target != "e":
                all_targets[target].append(value)

    return graphs, all_targets
