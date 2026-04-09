###########################################################################################
# Script for evaluating qdiv predictions on configurations in an xyz file
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import argparse

import ase.io
import numpy as np
import torch

from mace import data
from mace.tools import torch_geometric, torch_tools, utils


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate per-atom qdiv predictions with a trained AtomicQdivMACE model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--configs", help="path to XYZ configurations", required=True)
    parser.add_argument("--model", help="path to model", required=True)
    parser.add_argument("--output", help="output path", required=True)
    parser.add_argument(
        "--device",
        help="select device",
        type=str,
        choices=["cpu", "cuda"],
        default="cpu",
    )
    parser.add_argument(
        "--default_dtype",
        help="set default dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
    )
    parser.add_argument("--batch_size", help="batch size", type=int, default=64)
    parser.add_argument(
        "--qdiv_key",
        help="key in atoms.arrays for reference qdiv values (for error reporting)",
        type=str,
        default="REF_qdiv",
    )
    parser.add_argument(
        "--info_prefix",
        help="prefix for output keys",
        type=str,
        default="MACE_",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run(args)


def run(args: argparse.Namespace) -> None:
    torch_tools.set_default_dtype(args.default_dtype)
    device = torch_tools.init_device(args.device)

    # Load model
    model = torch.load(f=args.model, map_location=args.device)
    model = model.to(device)
    for param in model.parameters():
        param.requires_grad = False

    # Load data
    atoms_list = ase.io.read(args.configs, index=":")
    configs = [data.config_from_atoms(atoms) for atoms in atoms_list]

    z_table = utils.AtomicNumberTable([int(z) for z in model.atomic_numbers])

    try:
        heads = model.heads
    except AttributeError:
        heads = None

    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            data.AtomicData.from_config(
                config, z_table=z_table, cutoff=float(model.r_max), heads=heads
            )
            for config in configs
        ],
        batch_size=args.batch_size,
        shuffle=False,
        drop_last=False,
    )

    # Collect predictions
    qdiv_collection = []
    for batch in data_loader:
        batch = batch.to(device)
        output = model(batch.to_dict())
        qdivs = np.split(
            torch_tools.to_numpy(output["qdiv"]),
            indices_or_sections=batch.ptr[1:],
            axis=0,
        )
        qdiv_collection.append(qdivs[:-1])  # drop last as it's empty

    qdiv_list = [q for sublist in qdiv_collection for q in sublist]
    assert len(atoms_list) == len(qdiv_list)

    # Check for reference qdiv and compute errors
    has_ref = all(args.qdiv_key in atoms.arrays for atoms in atoms_list)
    if has_ref:
        all_ref = np.concatenate(
            [atoms.arrays[args.qdiv_key] for atoms in atoms_list]
        )
        all_pred = np.concatenate(qdiv_list)
        rmse = np.sqrt(np.mean((all_ref - all_pred) ** 2))
        mae = np.mean(np.abs(all_ref - all_pred))
        print(f"Qdiv RMSE: {rmse:.6f} e")
        print(f"Qdiv MAE:  {mae:.6f} e")

    # Store predictions and write output
    for atoms, qdiv in zip(atoms_list, qdiv_list):
        atoms.calc = None
        atoms.arrays[args.info_prefix + "qdiv"] = qdiv

    ase.io.write(args.output, images=atoms_list, format="extxyz")


if __name__ == "__main__":
    main()
