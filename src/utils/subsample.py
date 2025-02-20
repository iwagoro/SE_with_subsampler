# import torch
# import numpy as np


# def subsample(s, k):
#     """
#     Splits the input signal s into s1 and s2 based on random masks.

#     Parameters:
#     - s: torch.tensor of shape [1, Length], on device 'cpu' or 'mps'
#     - k: int, the width of the mask

#     Returns:
#     - s1: torch.tensor containing elements from s assigned to mask value 0
#     - s2: torch.tensor containing elements from s assigned to mask value 1
#     """
#     device = s.device
#     s = s.squeeze(0)  # Remove the batch dimension
#     Length = s.shape[1]
#     num_masks = Length // k

#     # Ensure num_masks is positive
#     if num_masks <= 0:
#         raise ValueError(f"num_masks is non-positive: {num_masks}. Check Length and k values.")

#     # Generate random permutations for all masks at once
#     rand_vals = np.random.rand(num_masks, k)
#     perms = np.argsort(rand_vals, axis=1)

#     # Assign labels based on the permuted order (even indices -> 0, odd indices -> 1)
#     idx_in_perm_mod2 = np.arange(k) % 2  # Shape (k,)
#     labels = np.zeros((num_masks, k), dtype=int)
#     labels[np.arange(num_masks)[:, None], perms] = idx_in_perm_mod2

#     # Compute the indices in the original signal
#     indices = np.arange(num_masks)[:, None] * k + np.arange(k)[None, :]

#     # Flatten indices and labels
#     flat_indices = indices.flatten()
#     flat_labels = labels.flatten()

#     # Ensure indices are within the signal length
#     valid_mask = flat_indices < Length
#     flat_indices = flat_indices[valid_mask]
#     flat_labels = flat_labels[valid_mask]

#     # Separate indices based on labels
#     s1_indices = flat_indices[flat_labels == 0]
#     s2_indices = flat_indices[flat_labels == 1]

#     # Move s to CPU for indexing operations
#     s_cpu = s.cpu().squeeze(0)

#     # Convert indices to tensors on CPU
#     s1_indices_cpu = torch.from_numpy(s1_indices).long()
#     s2_indices_cpu = torch.from_numpy(s2_indices).long()

#     # Extract data from s_cpu using the indices
#     s1_cpu = s_cpu[s1_indices_cpu]
#     s2_cpu = s_cpu[s2_indices_cpu]

#     # # Add noise on CPU
#     # noise_level = 0.05
#     # s1_cpu = s1_cpu + noise_level * torch.randn_like(s1_cpu)
#     # s2_cpu = s2_cpu + noise_level * torch.randn_like(s2_cpu)

#     # Move the results back to the original device
#     s1 = s1_cpu.to(device).unsqueeze(0)
#     s2 = s2_cpu.to(device).unsqueeze(0)

#     # Return s1 and s2 with an added batch dimension
#     return s1.unsqueeze(0), s2.unsqueeze(0)


# import torch


# def subsample(s, k):
#     """
#     Splits the input signal s into s1 and s2 based on random masks.

#     Parameters:
#     - s: torch.tensor of shape [1, Length], on device 'cpu' or 'mps'
#     - k: int, the width of the mask

#     Returns:
#     - s1: torch.tensor containing elements from s assigned to mask value 0
#     - s2: torch.tensor containing elements from s assigned to mask value 1
#     """
#     device = s.device
#     s = s.squeeze(0)
#     Length = s.shape[1]
#     num_masks = Length // k

#     # Ensure num_masks is positive
#     if num_masks <= 0:
#         raise ValueError(f"num_masks is non-positive: {num_masks}. Check Length and k values.")

#     # Generate random permutations for all masks at once
#     rand_vals = torch.rand((num_masks, k), device=device)
#     perms = torch.argsort(rand_vals, dim=1)

#     # Assign labels based on the permuted order (even indices -> 0, odd indices -> 1)
#     idx_in_perm_mod2 = (torch.arange(k, device=device) % 2).to(torch.int)  # Shape (k,)
#     labels = torch.zeros((num_masks, k), dtype=torch.int, device=device)
#     labels[torch.arange(num_masks, device=device)[:, None], perms] = idx_in_perm_mod2

#     # Compute the indices in the original signal
#     indices = torch.arange(num_masks, device=device)[:, None] * k + torch.arange(k, device=device)[None, :]

#     # Flatten indices and labels
#     flat_indices = indices.flatten()
#     flat_labels = labels.flatten()

#     # Ensure indices are within the signal length
#     valid_mask = flat_indices < Length
#     flat_indices = flat_indices[valid_mask]
#     flat_labels = flat_labels[valid_mask]

#     # Separate indices based on labels
#     s1_indices = flat_indices[flat_labels == 0]
#     s2_indices = flat_indices[flat_labels == 1]

#     # Extract data from s using the indices
#     s1 = s[:, s1_indices].unsqueeze(0)
#     s2 = s[:, s2_indices].unsqueeze(0)

#     return s1, s2

# import torch


# def subsample(s, k):
#     """
#     Splits the input signal s into s1 and s2 based on random masks.

#     Parameters:
#     - s: torch.tensor of shape [1, Length], on device 'cpu' or 'mps'
#     - k: int, the width of the mask

#     Returns:
#     - s1: torch.tensor containing elements from s assigned to mask value 0
#     - s2: torch.tensor containing elements from s assigned to mask value 1
#     """
#     device = s.device
#     s = s.squeeze(0)
#     Length = s.shape[1]
    
    
#     even = torch.arange(0, Length, 2).to(device)
#     odd = torch.arange(1, Length, 2).to(device)

#     # Separate indices based on labels
#     s1 = s[:,even].unsqueeze(0).to(device)
#     s2 = s[:,odd].unsqueeze(0).to(device)

#     return s1, s2



# import torch
# import numpy as np
# from scipy.interpolate import CubicSpline


# def subsample(s: torch.Tensor, k: int):
    
#     s = s.squeeze(0)
#     device = s.device
#     Length = s.shape[1]
#     num_masks = Length // k

#     if num_masks <= 0:
#         raise ValueError(f"num_masks is non-positive: {num_masks}. Check Length and k values.")

#     # rand_vals は形状 (num_masks, k) の一様乱数。これを行単位で argsort して、要素のランダムな並べ替え（permutation）のインデックス perms を取得．
#     rand_vals = torch.rand((num_masks, k), device=device)
#     perms = torch.argsort(rand_vals, dim=1)

#     # マスクのラベルを決定（偶数インデックス -> 0, 奇数インデックス -> 1）
#     idx_in_perm_mod2 = (torch.arange(k, device=device) % 2).to(torch.int)
#     labels = torch.zeros((num_masks, k), dtype=torch.int, device=device)
#     labels[torch.arange(num_masks, device=device)[:, None], perms] = idx_in_perm_mod2

#     # オリジナル信号のインデックスを計算
#     indices = torch.arange(num_masks, device=device)[:, None] * k + torch.arange(k, device=device)[None, :]
#     flat_indices = indices.flatten()
#     flat_labels = labels.flatten()
#     valid_mask = flat_indices < Length
#     flat_indices = flat_indices[valid_mask]
#     flat_labels = flat_labels[valid_mask]

#     # lat_labels == 0 の要素を取り出して s1_indices とし、その位置の元データを s1_values にまとめる．s2 も同様．
#     s1_indices = flat_indices[flat_labels == 0]
#     s2_indices = flat_indices[flat_labels == 1]
#     s1_values = s[:, s1_indices]  # shape: [1, N1]
#     s2_values = s[:, s2_indices]  # shape: [1, N2]

#     # 2) スプライン補完
#     # s1_filled = fill_missing_with_spline(Length, s1_indices, s1_values, device)
#     # s2_filled = fill_missing_with_spline(Length, s2_indices, s2_values, device)
#     # 2) スプライン補完 (stretch_signal_by_two を使って補間)
#     s1_filled = stretch_signal_by_two(s1_values, device).unsqueeze(0)
#     s2_filled = stretch_signal_by_two(s2_values, device).unsqueeze(0)

#     return s1_filled, s2_filled


# def stretch_signal_by_two(signal: torch.Tensor, device: torch.device):
#     N = signal.shape[1]
#     signal_np = signal.squeeze(0).detach().cpu().numpy()


#     x = np.arange(N)
#     cs = CubicSpline(x, signal_np)

#     stretched_np = np.zeros(2 * N, dtype=signal_np.dtype)
#     # 偶数番目(0, 2, 4, ..., 2N-2)に元のサンプル
#     stretched_np[0::2] = signal_np

#     # "N-1" 点だけ補間
#     x_half = np.arange(N - 1) + 0.5  # 長さ (N-1)
#     y_half = cs(x_half)             # shape: (N-1,)

#     # 奇数番目(1, 3, 5, ..., 2N-3)のスライスの要素数も N-1 個
#     stretched_np[1:2*(N-1):2] = y_half  # shape: (N-1,)

#     return torch.tensor(stretched_np, dtype=signal.dtype).unsqueeze(0).to(device)


import torch
import numpy as np
from scipy.interpolate import CubicSpline


def subsample(s: torch.Tensor, k: int):
    device = s.device
    s = s.squeeze(0)
    Length = s.shape[1]
    num_masks = Length // k

    if num_masks <= 0:
        raise ValueError(f"num_masks is non-positive: {num_masks}. Check Length and k values.")

    # 1) subsample
    rand_vals = torch.rand((num_masks, k), device=device)
    perms = torch.argsort(rand_vals, dim=1)

    idx_in_perm_mod2 = (torch.arange(k, device=device) % 2).to(torch.int)
    labels = torch.zeros((num_masks, k), dtype=torch.int, device=device)
    labels[torch.arange(num_masks, device=device)[:, None], perms] = idx_in_perm_mod2

    indices = torch.arange(num_masks, device=device)[:, None] * k + torch.arange(k, device=device)[None, :]
    flat_indices = indices.flatten()
    flat_labels = labels.flatten()
    valid_mask = flat_indices < Length
    flat_indices = flat_indices[valid_mask]
    flat_labels = flat_labels[valid_mask]

    s1_indices = flat_indices[flat_labels == 0]
    s2_indices = flat_indices[flat_labels == 1]

    s1_values = s[:, s1_indices]  # shape: [1, N1]
    s2_values = s[:, s2_indices]  # shape: [1, N2]

    # 2) スプライン補完
    s1_filled = fill_missing_with_spline(Length, s1_indices, s1_values, device).unsqueeze(0)
    s2_filled = fill_missing_with_spline(Length, s2_indices, s2_values, device).unsqueeze(0)

    return s1_filled, s2_filled


def fill_missing_with_spline(total_length, known_indices, known_values, device):
    known_indices_np = known_indices.detach().cpu().numpy().astype(np.float64)
    known_values_np = known_values.detach().cpu().numpy()[0]  # shape: (N,)

    # ソート（CubicSplineは x が単調増加である必要がある）
    sort_perm = np.argsort(known_indices_np)
    known_indices_np = known_indices_np[sort_perm]
    known_values_np = known_values_np[sort_perm]

    # CubicSpline (float64ベース)
    cs = CubicSpline(known_indices_np, known_values_np)

    # 全インデックス[0..total_length-1]
    x_new = np.arange(total_length, dtype=np.float64)
    filled_np = cs(x_new)  # shape: (total_length,) [float64]

    # ====== 解決策：ここでfloat32にキャストする ======
    filled = torch.from_numpy(filled_np.astype(np.float32)).unsqueeze(0).to(device)

    # 既知データを上書き (同じfloat32で代入)
    # known_values_npもfloat32に変換すると確実
    known_values_t = torch.from_numpy(known_values_np.astype(np.float32)).to(device)
    filled[:, known_indices] = known_values_t

    return filled