import numpy as np
import torch
from ase.atoms import Atoms
from nff.data import Dataset, collate_dicts
from nff.utils.cuda import batch_detach, batch_to
from torch.utils.data import DataLoader
from torch_scatter import scatter_sum

# from vssr.calculators.base import EnsembleModel

OUTPUT_KEYS = ["energy", "forces", "energy_grad", "embedding"]


def get_nff_prediction(
    model,
    dset: Dataset,
    batch_size: int = 10,
    device: str = "cuda",
    output_keys=OUTPUT_KEYS,
    **kwargs,
):
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        collate_fn=collate_dicts,
    )
    model.to(device)
    model.eval()
    predicted, num_atoms = [], []
    for batch in loader:
        batch = batch_to(batch, device=device)
        pred = model(batch, requires_embedding=True)
        batch = batch_detach(batch)
        pred_detached = batch_detach(pred)
        num_atoms.extend(batch["num_atoms"])

        predicted.append(pred_detached)
    # if isinstance(model, EnsembleModel):
    #     print("Getting std for values")
    #     output_keys = output_keys + ["energy_std", "forces_std"]
    #     predicted = {k: torch.concat([p[k] for p in predicted]) for k in predicted[0].keys() if k in output_keys}
    # else:
    print("Single Model")
    predicted = {
        k: torch.concat([p[k] for p in predicted]) for k in predicted[0].keys() if k in output_keys
    }
    predicted["num_atoms"] = torch.as_tensor(num_atoms).view(-1)
    if "forces" in output_keys and "energy_grad" in predicted:
        predicted["forces"] = -predicted["energy_grad"]
    elif "energy_grad" in output_keys and "forces" in predicted:
        predicted["energy_grad"] = -predicted["forces"]
    return predicted


def get_prediction(
    model,
    dset: Dataset | list[Atoms],
    batch_size: int = 10,
    device: str = "cuda",
    requires_grad: bool = False,
    **kwargs,
) -> tuple[dict, dict]:
    # if "PaiNN" in model.__repr__() or "SchNet" in model.__repr__():
    predicted = get_nff_prediction(model, dset, batch_size=batch_size, device=device, **kwargs)

    if isinstance(dset, Dataset):
        target = {
            "energy": dset.props["energy"],
            "energy_grad": dset.props["energy_grad"],
        }
    else:
        target = {"energy": [atoms.get_potential_energy() for atoms in dset]}
        target["energy_grad"] = [-atoms.get_forces(apply_constraint=False) for atoms in dset]

    target["energy_grad"] = np.concatenate(target["energy_grad"], axis=0)

    target["energy"] = torch.tensor(target["energy"]).to(predicted["energy"].device)
    target["forces"] = -torch.tensor(target.pop("energy_grad")).to(predicted["energy"].device)

    return target, predicted


def get_errors(predicted: dict, target: dict, mae=True, rmse=True, r2=True, max_error=True) -> dict:
    pred_energy = predicted["energy"].detach().cpu().numpy()
    targ_energy = target["energy"].detach().cpu().numpy()

    pred_forces = predicted["forces"].detach().cpu().numpy()
    targ_forces = target["forces"].detach().cpu().numpy()
    if pred_energy.ndim > 1 and pred_energy.shape != targ_energy.shape:
        pred_energy = pred_energy.mean(-1)
    if pred_forces.ndim > 2 and pred_forces.shape != targ_forces.shape:
        pred_forces = pred_forces.mean(-1)

    errors = {"energy": {}, "forces": {}}
    if mae:
        mae_energy = np.mean(np.abs(pred_energy - targ_energy))
        mae_forces = np.mean(np.abs(pred_forces - targ_forces))
        errors["energy"]["mae"] = mae_energy
        errors["forces"]["mae"] = mae_forces

    if rmse:
        rmse_energy = np.sqrt(np.mean((pred_energy - targ_energy) ** 2))
        rmse_forces = np.sqrt(np.mean((pred_forces - targ_forces) ** 2))
        errors["energy"]["rmse"] = rmse_energy
        errors["forces"]["rmse"] = rmse_forces

    if r2:
        r2_energy = 1 - np.sum((pred_energy - targ_energy) ** 2) / np.sum(
            (targ_energy - np.mean(targ_energy)) ** 2
        )
        r2_forces = 1 - np.sum((pred_forces - targ_forces) ** 2) / np.sum(
            (targ_forces - np.mean(targ_forces)) ** 2
        )
        errors["energy"]["r2"] = r2_energy
        errors["forces"]["r2"] = r2_forces

    if max_error:
        max_error_energy = np.max(np.abs(pred_energy - targ_energy))
        max_error_forces = np.max(np.abs(pred_forces - targ_forces))
        errors["energy"]["max_error"] = max_error_energy
        errors["forces"]["max_error"] = max_error_forces

    return errors


def get_embedding(
    model,
    dset: Dataset,
    batch_size: int,
    device: str,
) -> torch.Tensor:
    loader = DataLoader(
        dset,
        batch_size=batch_size,
        collate_fn=collate_dicts,
        shuffle=False,
    )

    embedding = []
    for batch in loader:
        batch = batch_to(batch, device=device)
        emb = model(batch, requires_embedding=True)["embedding"]
        emb = emb.detach().cpu()
        batch = batch_detach(batch)
        center_idx = batch.get("center_idx", None)
        if center_idx is not None:
            # print(center_idx, emb.shape[0])
            dummy_tensor = torch.tensor([0], device=emb.device, dtype=batch["num_atoms"].dtype)
            num_atoms_added = torch.cumsum(batch["num_atoms"], dim=0)
            added_tensor = torch.cat([dummy_tensor, num_atoms_added[:-1]], dim=0)
            # print(batch["num_atoms"],  num_atoms_added[:-1], added_tensor,)
            center_idx = center_idx + added_tensor
            # print(center_idx)
            system_emb = emb[center_idx]
        else:
            N = batch["num_atoms"].detach().cpu().tolist()
            batch_idx = torch.arange(len(N)).repeat_interleave(torch.LongTensor(N)).to(emb.device)
            system_emb = scatter_sum(emb, index=batch_idx, dim=0)
        # if True in torch.isnan(system_emb):
        #     print(batch["num_atoms"][~torch.isnan(system_emb)[:,0]])
        embedding.append(system_emb)

    embedding = torch.concat(embedding, dim=0)

    return embedding  # (n_atoms, n_atom_basis)


def get_prediction_and_errors(
    model, dset: Dataset | list[Atoms], batch_size: int, device: str
) -> tuple[dict, dict, dict]:
    target, predicted = get_prediction(model, dset, batch_size, device)

    target = batch_detach(target)
    predicted = batch_detach(predicted)

    errors = get_errors(predicted, target, mae=True, rmse=True, r2=True, max_error=True)

    return target, predicted, errors


def get_system_val(
    val: list[torch.Tensor], num_atoms: list[torch.Tensor], order: str
) -> torch.Tensor:
    if len(val) == len(num_atoms):
        # It's already a system value
        return val

    splits = torch.split(val, list(num_atoms))
    # Determine the maximum length
    max_length = max(t.size(0) for t in splits)
    padded_tensors = []
    masks = []
    for t in splits:
        padded_length = max_length - t.size(0)
        padded_tensors.append(torch.nn.functional.pad(t, (0, padded_length), "constant", 0))
        # Create a mask that is 1 where data is valid and 0 where it's padded
        mask = torch.ones_like(t, dtype=torch.bool)
        mask = torch.nn.functional.pad(mask, (0, padded_length), "constant", 0)
        masks.append(mask)
    stack_split = torch.stack(padded_tensors, dim=0)
    stacked_masks = torch.stack(masks, dim=0)
    valid_values = stack_split * stacked_masks
    if order == "system_sum":
        system_val = valid_values.sum(dim=-1)
        system_val = system_val.squeeze()
    elif order == "system_max":
        system_val = valid_values.max(dim=-1).values
        system_val = system_val.squeeze()
    elif order == "system_min":
        system_val = valid_values.min(dim=-1).values
        system_val = system_val.squeeze()
    elif order == "system_mean":
        system_val = valid_values.sum(dim=-1) / stacked_masks.sum(dim=-1)
        system_val = system_val.squeeze()
    elif order == "system_mean_squared":
        # valid_values = stack_split * stacked_masks
        system_val = (valid_values**2).sum(dim=-1) / stacked_masks.sum(dim=-1)
        system_val = system_val.squeeze()
    elif order == "system_root_mean_squared":
        # valid_values = stack_split * stacked_masks
        system_val = (valid_values**2).sum(dim=-1) / stacked_masks.sum(dim=-1)
        system_val = system_val.squeeze() ** 0.5
    return system_val


def get_residual(
    targ: dict,
    pred: dict,
    num_atoms: list[int],
    quantity: str = "forces",
    order: str = "system_mean",
) -> torch.Tensor:
    assert pred[quantity].shape == targ[quantity].shape
    # pred[quantity] = pred[quantity].mean(-1)

    res = targ[quantity] - pred[quantity]
    res = abs(res)

    # if quantity == "energy":
    #     return res
    if quantity == "forces" or quantity == "energy_grad":
        res = torch.norm(res, dim=-1)
        if "system" in order:
            system_res = get_system_val(res, num_atoms, order)
            return system_res
    return res
