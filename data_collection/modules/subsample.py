import torch


def subsample(s, k):
    """
    Splits the input signal s into s1 and s2 based on random masks.

    Parameters:
    - s: torch.tensor of shape [1, Length], on device 'cpu' or 'mps'
    - k: int, the width of the mask

    Returns:
    - s1: torch.tensor containing elements from s assigned to mask value 0
    - s2: torch.tensor containing elements from s assigned to mask value 1
    """
    device = s.device
    Length = s.shape[1]
    num_masks = Length // k

    # Ensure num_masks is positive
    if num_masks <= 0:
        raise ValueError(f"num_masks is non-positive: {num_masks}. Check Length and k values.")

    # Generate random permutations for all masks at once
    rand_vals = torch.rand((num_masks, k), device=device)
    with open("rand_vals.txt", "w") as f:
        f.write(str(rand_vals))
    perms = torch.argsort(rand_vals, dim=1)
    with open("perms.txt", "w") as f:
        f.write(str(perms))

    # Assign labels based on the permuted order (even indices -> 0, odd indices -> 1)
    idx_in_perm_mod2 = (torch.arange(k, device=device) % 2).to(torch.int)  # Shape (k,)
    with open("idx_in_perm_mod2.txt", "w") as f:
        f.write(str(idx_in_perm_mod2))
    labels = torch.zeros((num_masks, k), dtype=torch.int, device=device)
    labels[torch.arange(num_masks, device=device)[:, None], perms] = idx_in_perm_mod2
    with open("labels.txt", "w") as f:
        f.write(str(labels))

    # Compute the indices in the original signal
    indices = torch.arange(num_masks, device=device)[:, None] * k + torch.arange(k, device=device)[None, :]

    # Flatten indices and labels
    flat_indices = indices.flatten()
    flat_labels = labels.flatten()

    # Ensure indices are within the signal length
    valid_mask = flat_indices < Length
    flat_indices = flat_indices[valid_mask]
    flat_labels = flat_labels[valid_mask]

    # Separate indices based on labels
    s1_indices = flat_indices[flat_labels == 0]
    with open("s1_indices.txt", "w") as f:
        f.write(str(s1_indices))
    s2_indices = flat_indices[flat_labels == 1]

    # Extract data from s using the indices
    s1 = s[:, s1_indices]
    s2 = s[:, s2_indices]

    return s1, s2
