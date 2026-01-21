#!/usr/bin/env python
"""Script to verify the floating-point precision issue in exclusive ZBL mode."""

import torch

# Simulate the issue
def test_floating_point_precision():
    """Test if floating point precision causes edge masking issues."""
    
    # Simulate storing a minimum distance
    min_dist = 2.2592  # From the log: Pair (3, 3) at r_max=2.2592 Å
    
    # r_max stored as tensor
    r_max = torch.tensor(min_dist, dtype=torch.float64)
    
    # Now simulate edge lengths that should equal r_max
    # but might have slight floating point differences
    edge_lengths = []
    for _ in range(100):
        # Simulate computing the distance slightly differently
        x = torch.tensor([min_dist], dtype=torch.float64)
        # Add tiny floating point noise
        x = x + torch.randn(1) * 1e-15
        edge_lengths.append(x)
    
    edge_tensor = torch.cat(edge_lengths)
    
    # Check edge mask: x >= r_max
    mask = edge_tensor >= r_max
    
    print(f"r_max = {r_max.item()}")
    print(f"Number of edges: {len(edge_tensor)}")
    print(f"Edges passing mask (x >= r_max): {mask.sum().item()}")
    print(f"Edges failing mask (x < r_max): {(~mask).sum().item()}")
    
    # Show the differences
    diffs = edge_tensor - r_max
    print(f"\nDifference range: [{diffs.min().item():.2e}, {diffs.max().item():.2e}]")
    
    # Check envelope condition: x < r_max
    envelope_active = edge_tensor < r_max
    print(f"\nEdges where ZBL envelope is active: {envelope_active.sum().item()}")
    
    # Key insight: Even tiny differences can cause issues
    print("\n" + "="*60)
    print("KEY INSIGHT:")
    print("If an edge has x < r_max due to floating point errors,")
    print("it will be EXCLUDED from MACE processing and ZBL will contribute!")
    print("="*60)


if __name__ == "__main__":
    test_floating_point_precision()
