"""
Point cloud processing and sampling utilities.
"""

import numpy as np
import torch
import open3d as o3d


def fps_rad(pcd, radius):
    """
    Farthest point sampling on numpy array with radius constraint.
    
    Args:
        pcd: (n, 3) numpy array - input point cloud
        radius: float - sampling radius constraint
        
    Returns:
        (n) numpy array - indices of sampled points
    """
    rand_idx = np.random.randint(pcd.shape[0])
    selected_indices = [rand_idx]
    dist = np.linalg.norm(pcd - pcd[rand_idx], axis=1)
    
    while dist.max() > radius:
        farthest_idx = dist.argmax()
        selected_indices.append(farthest_idx)
        dist = np.minimum(dist, np.linalg.norm(pcd - pcd[farthest_idx], axis=1))

    # Sort indices for HDF5 compatibility (indices must be in increasing order)
    selected_indices = np.sort(selected_indices)
    
    return np.array(selected_indices)



def fps_rad_tensor(points, radius):
    """
    Farthest point sampling on tensor with radius constraint.
    
    Args:
        points: [N, 3] torch tensor on GPU - input point cloud
        radius: float - sampling radius constraint
    
    Returns:
        torch.Tensor - [M] indices of selected points
    """
    N = points.shape[0]
    device = points.device
    
    # Start with random point
    rand_idx = torch.randint(0, N, (1,), device=device)
    selected_indices = torch.zeros(N, dtype=torch.long, device=device)
    selected_indices[0] = rand_idx
    n_selected = 1

    # Distance to closest selected point
    dist = torch.norm(points - points[rand_idx], dim=1)
    
    # Keep adding farthest points until radius constraint satisfied
    while dist.max() > radius:
        farthest_idx = torch.argmax(dist)
        selected_indices[n_selected] = farthest_idx
        n_selected += 1
        
        # Update distances
        new_dists = torch.norm(points - points[farthest_idx], dim=1)
        dist = torch.minimum(dist, new_dists)
    
    return selected_indices[:n_selected]

# ============================================================================
# DEPRECATED
# ============================================================================

# def fps(pcd, particle_num, init_idx=-1):
#     # pcd: (n, 3) numpy array
#     # pcd_fps: (self.particle_num, 3) numpy array
#     pcd_tensor = torch.from_numpy(pcd).float()[None, ...]
#     if init_idx == -1:
#         # init_idx = findClosestPoint(pcd, pcd.mean(axis=0))
#         pcd_fps_idx_tensor = farthest_point_sampler(pcd_tensor, particle_num)[0]
#     else:
#         pcd_fps_idx_tensor = farthest_point_sampler(pcd_tensor, particle_num, init_idx)[0]
#     pcd_fps_tensor = pcd_tensor[0, pcd_fps_idx_tensor]
#     pcd_fps = pcd_fps_tensor.numpy()
#     dist = np.linalg.norm(pcd[:, None] - pcd_fps[None, :], axis=-1)
#     dist = dist.min(axis=1)
#     return pcd_fps, dist.max()


def fps_rad_old(pcd, radius):
    """
    Farthest point sampling on numpy array with radius constraint.
    
    Args:
        pcd: (n, 3) numpy array - input point cloud
        radius: float - sampling radius constraint
        
    Returns:
        numpy array - sampled points with radius constraint
    """
    rand_idx = np.random.randint(pcd.shape[0])
    pcd_fps_lst = [pcd[rand_idx]]
    dist = np.linalg.norm(pcd - pcd_fps_lst[0], axis=1)
    while dist.max() > radius:
        pcd_fps_lst.append(pcd[dist.argmax()])
        dist = np.minimum(dist, np.linalg.norm(pcd - pcd_fps_lst[-1], axis=1))
    pcd_fps = np.stack(pcd_fps_lst, axis=0)
    return pcd_fps


def fps_rad_tensor_old(pcd_tensor, radius):
    """
    Tensor-based FPS that returns indices directly (legacy version).
    
    Args:
        pcd_tensor: torch.Tensor of shape (n, 3)
        radius: float - sampling radius
    
    Returns:
        torch.Tensor - indices of sampled points
    """
    pcd = pcd_tensor.cpu().numpy()  # Convert only once
    rand_idx = np.random.randint(pcd.shape[0])
    selected_indices = [rand_idx]
    dist = np.linalg.norm(pcd - pcd[rand_idx], axis=1)
    
    while dist.max() > radius:
        farthest_idx = dist.argmax()
        selected_indices.append(farthest_idx)
        dist = np.minimum(dist, np.linalg.norm(pcd - pcd[farthest_idx], axis=1))
    
    return torch.from_numpy(np.array(selected_indices)).long()


def fps_np(pcd, particle_num, init_idx=-1):
    """
    Farthest point sampling on numpy array with fixed number of samples.
    
    Args:
        pcd: (n, c) numpy array - input point cloud
        particle_num: int - target number of samples
        init_idx: int - initial point index (-1 for random)
        
    Returns:
        tuple: (sampled_points, max_distance)
            - sampled_points: (particle_num, c) numpy array
            - max_distance: float - maximum distance to nearest sampled point
    """
    if init_idx == -1:
        rand_idx = np.random.randint(pcd.shape[0])
    else:
        rand_idx = init_idx
    pcd_fps_lst = [pcd[rand_idx]]
    dist = np.linalg.norm(pcd - pcd_fps_lst[0], axis=1)
    while len(pcd_fps_lst) < particle_num:
        pcd_fps_lst.append(pcd[dist.argmax()])
        dist = np.minimum(dist, np.linalg.norm(pcd - pcd_fps_lst[-1], axis=1))
    pcd_fps = np.stack(pcd_fps_lst, axis=0)
    return pcd_fps, dist.max()


def recenter(pcd, sampled_pcd, r=0.02):
    """
    Recenter sampled points around local point cloud neighborhoods.
    
    Args:
        pcd: (n, 3) numpy array - full point cloud
        sampled_pcd: (particle_num, 3) numpy array - sampled points
        r: float - neighborhood radius for recentering
        
    Returns:
        numpy array - recentered sampled points
    """
    particle_num = sampled_pcd.shape[0]
    dist = np.linalg.norm(pcd[:, None, :] - sampled_pcd[None, :, :], axis=2)  # (n, particle_num)
    recenter_sampled_pcd = np.zeros_like(sampled_pcd)
    for i in range(particle_num):
        recenter_sampled_pcd[i] = pcd[dist[:, i] < r].mean(axis=0)
    return recenter_sampled_pcd


def downsample_pcd(pcd, voxel_size):
    """
    Downsample point cloud using voxel grid filtering.
    
    Args:
        pcd: (n, 3) numpy array - input point cloud
        voxel_size: float - voxel size for downsampling
        
    Returns:
        numpy array - downsampled point cloud
    """
    # convert numpy array to open3d point cloud
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)

    downpcd_o3d = pcd_o3d.voxel_down_sample(voxel_size=voxel_size)
    downpcd = np.asarray(downpcd_o3d.points)

    return downpcd


def np2o3d(pcd, color=None):
    """
    Convert numpy point cloud to Open3D format.
    
    Args:
        pcd: (n, 3) numpy array - point positions
        color: (n, 3) numpy array - point colors (optional, values in [0,1])
        
    Returns:
        o3d.geometry.PointCloud - Open3D point cloud object
    """
    pcd_o3d = o3d.geometry.PointCloud()
    pcd_o3d.points = o3d.utility.Vector3dVector(pcd)
    if color is not None:
        assert pcd.shape[0] == color.shape[0]
        assert color.max() <= 1
        assert color.min() >= 0
        pcd_o3d.colors = o3d.utility.Vector3dVector(color)
    return pcd_o3d 