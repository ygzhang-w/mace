import logging
from typing import Any, Union, Dict, List, Optional

import torch
from prettytable import PrettyTable

from mace.tools import evaluate


def custom_key(key):
    """
    Helper function to sort the keys of the data loader dictionary
    to ensure that the training set, and validation set
    are evaluated first
    """
    if key == "train":
        return (0, key)
    if key == "valid":
        return (1, key)
    return (2, key)


def create_error_table(
    table_type: str,
    all_data_loaders: dict,
    model: torch.nn.Module,
    loss_fn: torch.nn.Module,
    output_args: Dict[str, bool],
    log_wandb: bool,
    device: str,
    distributed: bool = False,
    skip_heads: Optional[List[str]] = None,
) -> PrettyTable:
    if log_wandb:
        import wandb
    skip_heads = skip_heads or []
    table = PrettyTable()
    if table_type == "TotalRMSE":
        table.field_names = [
            "config_type",
            "RMSE E / meV",
            "RMSE F / meV / A",
            "relative F RMSE %",
        ]
    elif table_type == "PerAtomRMSE":
        table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
        ]
    elif table_type == "PerAtomRMSE_ei":
        table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
            "RMSE Ei / meV",
        ]
    elif table_type == "PerAtomRMSEstressvirials":
        table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "relative F RMSE %",
            "RMSE Stress (Virials) / meV / A (A^3)",
        ]
    elif table_type == "PerAtomMAEstressvirials":
        table.field_names = [
            "config_type",
            "MAE E / meV / atom",
            "MAE F / meV / A",
            "relative F MAE %",
            "MAE Stress (Virials) / meV / A (A^3)",
        ]
    elif table_type == "TotalMAE":
        table.field_names = [
            "config_type",
            "MAE E / meV",
            "MAE F / meV / A",
            "relative F MAE %",
        ]
    elif table_type == "PerAtomMAE":
        table.field_names = [
            "config_type",
            "MAE E / meV / atom",
            "MAE F / meV / A",
            "relative F MAE %",
        ]
    elif table_type == "PerAtomMAE_ei":
        table.field_names = [
            "config_type",
            "MAE E / meV / atom",
            "MAE F / meV / A",
            "relative F MAE %",
            "MAE Ei / meV",
        ]
    elif table_type == "DipoleRMSE":
        table.field_names = [
            "config_type",
            "RMSE MU / mDebye / atom",
            "relative MU RMSE %",
        ]
    elif table_type == "DipoleMAE":
        table.field_names = [
            "config_type",
            "MAE MU / mDebye / atom",
            "relative MU MAE %",
        ]
    elif table_type == "DipolePolarRMSE":
        table.field_names = [
            "config_type",
            "RMSE MU / me A / atom",
            "relative MU RMSE %",
            "RMSE ALPHA e A^2 / V / atom",
        ]
    elif table_type == "EnergyDipoleRMSE":
        table.field_names = [
            "config_type",
            "RMSE E / meV / atom",
            "RMSE F / meV / A",
            "rel F RMSE %",
            "RMSE MU / mDebye / atom",
            "rel MU RMSE %",
        ]

    for name in sorted(all_data_loaders, key=custom_key):
        if any(skip_head in name for skip_head in skip_heads):
            logging.info(f"Skipping evaluation of {name} (in skip_heads list)")
            continue
        data_loader = all_data_loaders[name]
        logging.info(f"Evaluating {name} ...")
        _, metrics = evaluate(
            model,
            loss_fn=loss_fn,
            data_loader=data_loader,
            output_args=output_args,
            device=device,
        )
        if distributed:
            torch.distributed.barrier()

        del data_loader
        torch.cuda.empty_cache()
        if log_wandb:
            wandb_log_dict = {
                name
                + "_final_rmse_e_per_atom": metrics["rmse_e_per_atom"]
                * 1e3,  # meV / atom
                name + "_final_rmse_f": metrics["rmse_f"] * 1e3,  # meV / A
                name + "_final_rel_rmse_f": metrics["rel_rmse_f"],
            }
            wandb.log(wandb_log_dict)
        if table_type == "TotalRMSE":
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_e'] * 1000:8.1f}",
                    f"{metrics['rmse_f'] * 1000:8.1f}",
                    f"{metrics['rel_rmse_f']:8.2f}",
                ]
            )
        elif table_type == "PerAtomRMSE":
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_e_per_atom'] * 1000:8.1f}",
                    f"{metrics['rmse_f'] * 1000:8.1f}",
                    f"{metrics['rel_rmse_f']:8.2f}",
                ]
            )
        elif table_type == "PerAtomRMSE_ei":
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_e_per_atom'] * 1000:8.1f}",
                    f"{metrics['rmse_f'] * 1000:8.1f}",
                    f"{metrics['rel_rmse_f']:8.2f}",
                    f"{metrics['rmse_ei'] * 1000:8.1f}",
                ]
            )
        elif (
            table_type == "PerAtomRMSEstressvirials"
            and metrics["rmse_stress"] is not None
        ):
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_e_per_atom'] * 1000:8.1f}",
                    f"{metrics['rmse_f'] * 1000:8.1f}",
                    f"{metrics['rel_rmse_f']:8.2f}",
                    f"{metrics['rmse_stress'] * 1000:8.1f}",
                ]
            )
        elif (
            table_type == "PerAtomRMSEstressvirials"
            and metrics["rmse_virials"] is not None
        ):
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_e_per_atom'] * 1000:8.1f}",
                    f"{metrics['rmse_f'] * 1000:8.1f}",
                    f"{metrics['rel_rmse_f']:8.2f}",
                    f"{metrics['rmse_virials'] * 1000:8.1f}",
                ]
            )
        elif (
            table_type == "PerAtomMAEstressvirials"
            and metrics["mae_stress"] is not None
        ):
            table.add_row(
                [
                    name,
                    f"{metrics['mae_e_per_atom'] * 1000:8.1f}",
                    f"{metrics['mae_f'] * 1000:8.1f}",
                    f"{metrics['rel_mae_f']:8.2f}",
                    f"{metrics['mae_stress'] * 1000:8.1f}",
                ]
            )
        elif (
            table_type == "PerAtomMAEstressvirials"
            and metrics["mae_virials"] is not None
        ):
            table.add_row(
                [
                    name,
                    f"{metrics['mae_e_per_atom'] * 1000:8.1f}",
                    f"{metrics['mae_f'] * 1000:8.1f}",
                    f"{metrics['rel_mae_f']:8.2f}",
                    f"{metrics['mae_virials'] * 1000:8.1f}",
                ]
            )
        elif table_type == "TotalMAE":
            table.add_row(
                [
                    name,
                    f"{metrics['mae_e'] * 1000:8.1f}",
                    f"{metrics['mae_f'] * 1000:8.1f}",
                    f"{metrics['rel_mae_f']:8.2f}",
                ]
            )
        elif table_type == "PerAtomMAE":
            table.add_row(
                [
                    name,
                    f"{metrics['mae_e_per_atom'] * 1000:8.1f}",
                    f"{metrics['mae_f'] * 1000:8.1f}",
                    f"{metrics['rel_mae_f']:8.2f}",
                ]
            )
        elif table_type == "PerAtomMAE_ei":
            table.add_row(
                [
                    name,
                    f"{metrics['mae_e_per_atom'] * 1000:8.1f}",
                    f"{metrics['mae_f'] * 1000:8.1f}",
                    f"{metrics['rel_mae_f']:8.2f}",
                    f"{metrics['mae_ei'] * 1000:8.1f}",
                ]
            )
        elif table_type == "DipoleRMSE":
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_mu_per_atom'] * 1000:8.2f}",
                    f"{metrics['rel_rmse_mu']:8.1f}",
                ]
            )
        elif table_type == "DipoleMAE":
            table.add_row(
                [
                    name,
                    f"{metrics['mae_mu_per_atom'] * 1000:8.2f}",
                    f"{metrics['rel_mae_mu']:8.1f}",
                ]
            )
        elif table_type == "DipolePolarRMSE":
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_mu_per_atom'] * 1000:.2f}",
                    f"{metrics['rel_rmse_mu']:.1f}",
                    f"{metrics['rmse_polarizability_per_atom'] * 1000:.2f}",
                ]
            )
        elif table_type == "EnergyDipoleRMSE":
            table.add_row(
                [
                    name,
                    f"{metrics['rmse_e_per_atom'] * 1000:8.1f}",
                    f"{metrics['rmse_f'] * 1000:8.1f}",
                    f"{metrics['rel_rmse_f']:8.1f}",
                    f"{metrics['rmse_mu_per_atom'] * 1000:8.1f}",
                    f"{metrics['rel_rmse_mu']:8.1f}",
                ]
            )
    return table


def get_table_field_names(table_type: str) -> List[str]:
    """Get field names for a given table type."""
    field_names_map = {
        "TotalRMSE": ["config_type", "RMSE E / meV", "RMSE F / meV / A", "relative F RMSE %"],
        "PerAtomRMSE": ["config_type", "RMSE E / meV / atom", "RMSE F / meV / A", "relative F RMSE %"],
        "PerAtomRMSE_ei": ["config_type", "RMSE E / meV / atom", "RMSE F / meV / A", "relative F RMSE %", "RMSE Ei / meV"],
        "PerAtomRMSEstressvirials": ["config_type", "RMSE E / meV / atom", "RMSE F / meV / A", "relative F RMSE %", "RMSE Stress (Virials) / meV / A (A^3)"],
        "PerAtomMAEstressvirials": ["config_type", "MAE E / meV / atom", "MAE F / meV / A", "relative F MAE %", "MAE Stress (Virials) / meV / A (A^3)"],
        "TotalMAE": ["config_type", "MAE E / meV", "MAE F / meV / A", "relative F MAE %"],
        "PerAtomMAE": ["config_type", "MAE E / meV / atom", "MAE F / meV / A", "relative F MAE %"],
        "PerAtomMAE_ei": ["config_type", "MAE E / meV / atom", "MAE F / meV / A", "relative F MAE %", "MAE Ei / meV"],
        "DipoleRMSE": ["config_type", "RMSE MU / mDebye / atom", "relative MU RMSE %"],
        "DipoleMAE": ["config_type", "MAE MU / mDebye / atom", "relative MU MAE %"],
        "DipolePolarRMSE": ["config_type", "RMSE MU / me A / atom", "relative MU RMSE %", "RMSE ALPHA e A^2 / V / atom"],
        "EnergyDipoleRMSE": ["config_type", "RMSE E / meV / atom", "RMSE F / meV / A", "rel F RMSE %", "RMSE MU / mDebye / atom", "rel MU RMSE %"],
    }
    return field_names_map.get(table_type, ["config_type"])


def add_row_for_table_type(
    table: PrettyTable,
    table_type: str,
    name: str,
    metrics: Dict[str, Any],
) -> None:
    """Add a row to the table based on the table type and metrics."""
    if table_type == "TotalRMSE":
        table.add_row([
            name,
            f"{metrics['rmse_e'] * 1000:8.1f}",
            f"{metrics['rmse_f'] * 1000:8.1f}",
            f"{metrics['rel_rmse_f']:8.2f}",
        ])
    elif table_type == "PerAtomRMSE":
        table.add_row([
            name,
            f"{metrics['rmse_e_per_atom'] * 1000:8.1f}",
            f"{metrics['rmse_f'] * 1000:8.1f}",
            f"{metrics['rel_rmse_f']:8.2f}",
        ])
    elif table_type == "PerAtomRMSE_ei":
        table.add_row([
            name,
            f"{metrics['rmse_e_per_atom'] * 1000:8.1f}",
            f"{metrics['rmse_f'] * 1000:8.1f}",
            f"{metrics['rel_rmse_f']:8.2f}",
            f"{metrics.get('rmse_ei', 0) * 1000:8.1f}",
        ])
    elif table_type == "PerAtomRMSEstressvirials":
        stress_val = metrics.get('rmse_stress') or metrics.get('rmse_virials') or 0
        table.add_row([
            name,
            f"{metrics['rmse_e_per_atom'] * 1000:8.1f}",
            f"{metrics['rmse_f'] * 1000:8.1f}",
            f"{metrics['rel_rmse_f']:8.2f}",
            f"{stress_val * 1000:8.1f}",
        ])
    elif table_type == "PerAtomMAEstressvirials":
        stress_val = metrics.get('mae_stress') or metrics.get('mae_virials') or 0
        table.add_row([
            name,
            f"{metrics['mae_e_per_atom'] * 1000:8.1f}",
            f"{metrics['mae_f'] * 1000:8.1f}",
            f"{metrics.get('rel_mae_f', 0):8.2f}",
            f"{stress_val * 1000:8.1f}",
        ])
    elif table_type == "TotalMAE":
        table.add_row([
            name,
            f"{metrics['mae_e'] * 1000:8.1f}",
            f"{metrics['mae_f'] * 1000:8.1f}",
            f"{metrics.get('rel_mae_f', 0):8.2f}",
        ])
    elif table_type == "PerAtomMAE":
        table.add_row([
            name,
            f"{metrics['mae_e_per_atom'] * 1000:8.1f}",
            f"{metrics['mae_f'] * 1000:8.1f}",
            f"{metrics.get('rel_mae_f', 0):8.2f}",
        ])
    elif table_type == "PerAtomMAE_ei":
        table.add_row([
            name,
            f"{metrics['mae_e_per_atom'] * 1000:8.1f}",
            f"{metrics['mae_f'] * 1000:8.1f}",
            f"{metrics.get('rel_mae_f', 0):8.2f}",
            f"{metrics.get('mae_ei', 0) * 1000:8.1f}",
        ])
    elif table_type == "DipoleRMSE":
        table.add_row([
            name,
            f"{metrics.get('rmse_mu_per_atom', 0) * 1000:8.2f}",
            f"{metrics.get('rel_rmse_mu', 0):8.1f}",
        ])
    elif table_type == "DipoleMAE":
        table.add_row([
            name,
            f"{metrics.get('mae_mu_per_atom', 0) * 1000:8.2f}",
            f"{metrics.get('rel_mae_mu', 0):8.1f}",
        ])
    elif table_type == "DipolePolarRMSE":
        table.add_row([
            name,
            f"{metrics.get('rmse_mu_per_atom', 0) * 1000:.2f}",
            f"{metrics.get('rel_rmse_mu', 0):.1f}",
            f"{metrics.get('rmse_polarizability_per_atom', 0) * 1000:.2f}",
        ])
    elif table_type == "EnergyDipoleRMSE":
        table.add_row([
            name,
            f"{metrics['rmse_e_per_atom'] * 1000:8.1f}",
            f"{metrics['rmse_f'] * 1000:8.1f}",
            f"{metrics['rel_rmse_f']:8.1f}",
            f"{metrics.get('rmse_mu_per_atom', 0) * 1000:8.1f}",
            f"{metrics.get('rel_rmse_mu', 0):8.1f}",
        ])


def create_error_tables_for_heads(
    default_table_type: str,
    head_error_table_types: Dict[str, str],
    all_data_loaders: dict,
    model: torch.nn.Module,
    loss_fn: Union[torch.nn.Module, Dict[str, torch.nn.Module]],
    output_args: Dict[str, bool],
    log_wandb: bool,
    device: str,
    distributed: bool = False,
    skip_heads: Optional[List[str]] = None,
) -> Dict[str, PrettyTable]:
    """
    Create error tables with per-head error_table types.

    Args:
        default_table_type: Default table type to use if head not specified
        head_error_table_types: Dict mapping head names to their error_table types
        all_data_loaders: Dict of data loaders
        model: The model to evaluate
        loss_fn: Loss function (can be a single module or dict of modules per head)
        output_args: Output arguments
        log_wandb: Whether to log to wandb
        device: Device to use
        distributed: Whether using distributed training
        skip_heads: List of head names to skip

    Returns:
        Dict mapping table_type to PrettyTable
    """
    if log_wandb:
        import wandb
    skip_heads = skip_heads or []

    # Group data loaders by their table type
    table_type_to_loaders: Dict[str, Dict[str, Any]] = {}
    for name in sorted(all_data_loaders, key=custom_key):
        if any(skip_head in name for skip_head in skip_heads):
            logging.info(f"Skipping evaluation of {name} (in skip_heads list)")
            continue

        # Determine table type for this loader
        # Try to find head name in the loader name
        table_type = default_table_type
        for head_name, head_table_type in head_error_table_types.items():
            if head_name in name:
                table_type = head_table_type
                break

        if table_type not in table_type_to_loaders:
            table_type_to_loaders[table_type] = {}
        table_type_to_loaders[table_type][name] = all_data_loaders[name]

    # Create tables for each table type
    tables: Dict[str, PrettyTable] = {}
    for table_type, loaders in table_type_to_loaders.items():
        table = PrettyTable()
        table.field_names = get_table_field_names(table_type)

        for name, data_loader in loaders.items():
            logging.info(f"Evaluating {name} with table_type={table_type} ...")

            # Get loss_fn for this loader
            if isinstance(loss_fn, dict):
                # Find matching loss_fn
                loader_loss_fn = None
                for head_name, head_loss in loss_fn.items():
                    if head_name in name:
                        loader_loss_fn = head_loss
                        break
                if loader_loss_fn is None:
                    loader_loss_fn = list(loss_fn.values())[0]
            else:
                loader_loss_fn = loss_fn

            _, metrics = evaluate(
                model,
                loss_fn=loader_loss_fn,
                data_loader=data_loader,
                output_args=output_args,
                device=device,
            )
            if distributed:
                torch.distributed.barrier()

            del data_loader
            torch.cuda.empty_cache()

            if log_wandb:
                wandb_log_dict = {
                    name + "_final_rmse_e_per_atom": metrics["rmse_e_per_atom"] * 1e3,
                    name + "_final_rmse_f": metrics["rmse_f"] * 1e3,
                    name + "_final_rel_rmse_f": metrics["rel_rmse_f"],
                }
                wandb.log(wandb_log_dict)

            add_row_for_table_type(table, table_type, name, metrics)

        tables[table_type] = table

    return tables
