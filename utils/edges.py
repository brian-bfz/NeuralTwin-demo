import torch
import numpy as np

def construct_edges_from_numpy(points, adj_thresh, topk):
    """
    Construct topological edges for points in numpy array
    
    Args:
        points: [N, 3] numpy array - coordinates of points
        adj_thresh: float - radius of neighborhood
        topk: int - maximum number of neighbors

    Returns:
        adj_matrix: [N, N] numpy array - adjacency matrix of edges
    """
    N = points.shape[0]
        
    # Compute pairwise squared distances
    diff = points[:, None, :] - points[None, :, :]
    distances_sq = np.sum(diff ** 2, axis=-1)
        
    # Threshold-based adjacency
    adj_matrix = (distances_sq < adj_thresh ** 2).astype(float)
        
    # Apply topk constraint
    topk = min(N-1, topk)
    topk_idx = np.argpartition(distances_sq, topk, axis=-1)[:, :topk]
    topk_matrix = np.zeros_like(adj_matrix)
    np.put_along_axis(topk_matrix, topk_idx, 1, axis=-1)
    adj_matrix = adj_matrix * topk_matrix

    return adj_matrix

def construct_edges_from_tensor(points, adj_thresh, topk):
    """
    Construct topological edges for points in tensor

    Args:
        points: [N, 3] torch tensor - coordinates of points
        adj_thresh: float - radius of neighborhood
        topk: int - maximum number of neighbors

    Returns:
        adj_matrix: [N, N] torch tensor - adjacency matrix of edges
    """
    N, _ = points.shape

    # Create pairwise particle combinations
    s_receiv = points[:, None, :].repeat(1, N, 1)
    s_sender = points[None, :, :].repeat(N, 1, 1)

    # Create adjacency matrix based on distance threshold
    threshold = adj_thresh * adj_thresh
    s_diff = s_receiv - s_sender  # Position differences
    dis = torch.sum(s_diff ** 2, -1)  # Squared distances (N, N)
    adj_matrix = ((dis - threshold) < 0).float()

    # Apply topk constraint
    topk = min(dis.shape[-1], topk)
    topk_idx = torch.topk(dis, k=topk, dim=-1, largest=False)[1]
    topk_matrix = torch.zeros_like(adj_matrix)
    topk_matrix.scatter_(-1, topk_idx, 1)
    adj_matrix = adj_matrix * topk_matrix

    return adj_matrix

def adj_mat_to_sparse_edge(adj_matrix):
    """
    Convert adjacency matrix to sparse edge representation
    """
    n_rels = adj_matrix.sum(dim=(1,2)).long()  # [B] - Number of edges per batch
    n_rel = n_rels.max().long().item()  # int - Maximum edges across batch
    rels = adj_matrix.nonzero()  # [E_c, 3] - each row is [batch_idx, receiver_idx, sender_idx]

    total_rels = rels.shape[0]
    if total_rels > 0:
        cs = torch.cumsum(n_rels, 0)
        cs_shifted = torch.cat([torch.zeros(1, device=adj_matrix.device, dtype=torch.long), cs[:-1]])
        rels_idx = torch.arange(total_rels, device=adj_matrix.device, dtype=torch.long) - cs_shifted.repeat_interleave(n_rels)
    else:
        rels_idx = torch.tensor([], device=adj_matrix.device, dtype=torch.long)
    return n_rel, rels, rels_idx

def construct_collision_edges(states, adj_thresh, mask, tool_mask, topk, connect_tools_all, topological_edges):
    """
    Construct collision edges between particles based on distance
    Remove edges that are already topological edges
    
    Args:
        states: (B, N, state_dim) torch tensor - particle positions
        adj_thresh: float or (B,) torch tensor - distance threshold for connections
        mask: (B, N) torch tensor - true when index is a valid particle
        tool_mask: (B, N) torch tensor - true when index is a valid tool particle
        topk: int - maximum number of neighbors per particle
        connect_tools_all: bool - if True, connect all tool particles to all object particles
        topological_edges: [B, n_object, n_object] torch tensor - adjacency matrix of topological edges 

    Returns:
        Rr: (B, n_rel, N) torch tensor - receiver matrix for collision edges
        Rs: (B, n_rel, N) torch tensor - sender matrix for collision edges
    """
    B, N, _ = states.shape
    
    # Create pairwise particle combinations for distance calculation
    s_receiv = states[:, :, None, :].repeat(1, 1, N, 1)  # Receiver particles
    s_sender = states[:, None, :, :].repeat(1, N, 1, 1)   # Sender particles

    # Calculate squared distances between all particle pairs and the squared distance threshold
    if isinstance(adj_thresh, float):
        adj_thresh = torch.tensor(adj_thresh, device=states.device, dtype=states.dtype).repeat(B)
    threshold = adj_thresh * adj_thresh
    s_diff = s_receiv - s_sender  # Position differences
    dis = torch.sum(s_diff ** 2, -1)  # Squared distances (B, N, N)
    
    # Create validity masks for particle connections
    mask_1 = mask[:, :, None].repeat(1, 1, N)  # Receiver validity
    mask_2 = mask[:, None, :].repeat(1, N, 1)  # Sender validity
    mask_12 = mask_1 * mask_2  # Both particles are valid
    dis[~mask_12] = 1e10  # Exclude invalid particle pairs
    
    # Prevent tool-to-tool connections
    tool_mask_1 = tool_mask[:, :, None].repeat(1, 1, N)
    tool_mask_2 = tool_mask[:, None, :].repeat(1, N, 1)
    tool_mask_12 = tool_mask_1 * tool_mask_2
    dis[tool_mask_12] = 1e10  # Avoid tool to tool relations

    # Create adjacency matrix based on distance threshold
    adj_matrix = ((dis - threshold[:, None, None]) < 0).float()

    # Define tool-object interaction masks
    obj_tool_mask_1 = tool_mask_1 * mask_2  # Particle sender, tool receiver

    # Apply topk constraint to limit connections per particle
    topk = min(dis.shape[-1], topk)
    topk_idx = torch.topk(dis, k=topk, dim=-1, largest=False)[1]
    topk_matrix = torch.zeros_like(adj_matrix)
    topk_matrix.scatter_(-1, topk_idx, 1)
    adj_matrix = adj_matrix * topk_matrix

    adj_matrix[obj_tool_mask_1] = 0  # Clear object-to-tool edges

    # Remove topological edges
    topological_mask = topological_edges > 0.5
    adj_matrix[topological_mask] = 0

    # Convert adjacency matrix to sparse edge representation
    n_rel, rels, rels_idx = adj_mat_to_sparse_edge(adj_matrix)

    # Create receiver and sender matrices
    Rr = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rs = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rr[rels[:, 0], rels_idx, rels[:, 1]] = 1 # receiver matrix [B, n_rel, N]
    Rs[rels[:, 0], rels_idx, rels[:, 2]] = 1 # sender matrix [B, n_rel, N]
        
    return Rr, Rs


def construct_topological_edges(adj_matrix, first_states):
    """
    Turn the adj_matrix into receiver and sender matrices 

    Args:
        adj_matrix: (B, N, N) torch tensor - adjacency matrix of topological edges
        first_states: (B, N, 3) torch tensor - first frame positions for computing edge lengths

    Returns:
        Rr: (B, n_rel, N) torch tensor - receiver matrix for topological edges
        Rs: (B, n_rel, N) torch tensor - sender matrix for topological edges
        first_edge_lengths: (B, n_rel) torch tensor - distance between receiver and sender in the first frame
    """
    B, N, _ = first_states.shape
    
    # Convert adjacency matrix to sparse edge representation
    n_rel, rels, rels_idx = adj_mat_to_sparse_edge(adj_matrix)

    # Create receiver and sender matrices
    Rr = torch.zeros((B, n_rel, N), device=first_states.device, dtype=first_states.dtype)
    Rs = torch.zeros((B, n_rel, N), device=first_states.device, dtype=first_states.dtype)
    Rr[rels[:, 0], rels_idx, rels[:, 1]] = 1 # receiver matrix [B, n_rel, N]
    Rs[rels[:, 0], rels_idx, rels[:, 2]] = 1 # sender matrix [B, n_rel, N]
    
    # Compute first frame edge lengths
    first_edge_lengths = torch.zeros((B, n_rel), device=first_states.device, dtype=first_states.dtype)
    if len(rels) > 0:
        # Get receiver and sender positions from first frame
        receiver_pos = first_states[rels[:, 0], rels[:, 1]]  # [n_edges, 3]
        sender_pos = first_states[rels[:, 0], rels[:, 2]]    # [n_edges, 3]
        edge_lengths = torch.norm(receiver_pos - sender_pos, dim=1)  # [n_edges]
        first_edge_lengths[rels[:, 0], rels_idx] = edge_lengths
        
    return Rr, Rs, first_edge_lengths

# ================================
# NOT USED
# ================================
def construct_edges_with_attrs(states, adj_thresh, mask, tool_mask, topk, connect_tools_all, topological_edges):
    """
    Construct collision edges between particles based on distance and topology
    
    Args:
        states: (B, N, state_dim) torch tensor - particle positions
        adj_thresh: float or (B,) torch tensor - distance threshold for connections
        mask: (B, N) torch tensor - true when index is a valid particle
        tool_mask: (B, N) torch tensor - true when index is a valid tool particle
        topk: int - maximum number of neighbors per particle
        connect_tools_all: bool - if True, connect all tool particles to all object particles
        topological_edges: [B, n_object, n_object] torch tensor - adjacency matrix of topological edges 

    Returns:
        Rr: (B, n_rel, N) torch tensor - receiver matrix for graph edges
        Rs: (B, n_rel, N) torch tensor - sender matrix for graph edges
        edge_attrs: (B, n_rel, 1) torch tensor - edge attributes, 1 for topological, 0 for collision
    """

    B, N, state_dim = states.shape
    
    # Create pairwise particle combinations for distance calculation
    s_receiv = states[:, :, None, :].repeat(1, 1, N, 1)  # Receiver particles
    s_sender = states[:, None, :, :].repeat(1, N, 1, 1)   # Sender particles

    # Calculate squared distances between all particle pairs and the squared distance threshold
    if isinstance(adj_thresh, float):
        adj_thresh = torch.tensor(adj_thresh, device=states.device, dtype=states.dtype).repeat(B)
    threshold = adj_thresh * adj_thresh
    s_diff = s_receiv - s_sender  # Position differences
    dis = torch.sum(s_diff ** 2, -1)  # Squared distances (B, N, N)
    
    # Create validity masks for particle connections
    mask_1 = mask[:, :, None].repeat(1, 1, N)  # Receiver validity
    mask_2 = mask[:, None, :].repeat(1, N, 1)  # Sender validity
    mask_12 = mask_1 * mask_2  # Both particles are valid
    dis[~mask_12] = 1e10  # Exclude invalid particle pairs
    
    # Prevent tool-to-tool connections
    tool_mask_1 = tool_mask[:, :, None].repeat(1, 1, N)
    tool_mask_2 = tool_mask[:, None, :].repeat(1, N, 1)
    tool_mask_12 = tool_mask_1 * tool_mask_2
    dis[tool_mask_12] = 1e10  # Avoid tool to tool relations

    # Create adjacency matrix based on distance threshold
    adj_matrix = ((dis - threshold[:, None, None]) < 0).float()

    # Define tool-object interaction masks
    obj_tool_mask_1 = tool_mask_1 * mask_2  # Particle sender, tool receiver
    obj_tool_mask_2 = tool_mask_2 * mask_1  # Tool sender, particle receiver
    obj_pad_tool_mask_1 = tool_mask_1 * (~tool_mask_2) # Tool receiver, non-tool sender

    # Apply topk constraint to limit connections per particle
    topk = min(dis.shape[-1], topk)
    topk_idx = torch.topk(dis, k=topk, dim=-1, largest=False)[1]
    topk_matrix = torch.zeros_like(adj_matrix)
    topk_matrix.scatter_(-1, topk_idx, 1)
    adj_matrix = adj_matrix * topk_matrix

    # Handle tool connectivity rules
    if connect_tools_all:
        # Only connect tools to objects if there are neighboring tool - non-tool particles in batch
        batch_mask = (adj_matrix[obj_pad_tool_mask_1].reshape(B, -1).sum(-1) > 0)[:, None, None].repeat(1, N, N)
        batch_obj_tool_mask_1 = obj_tool_mask_1 * batch_mask  # (B, N, N)
        neg_batch_obj_tool_mask_1 = obj_tool_mask_1 * (~batch_mask)  # (B, N, N)
        batch_obj_tool_mask_2 = obj_tool_mask_2 * batch_mask  # (B, N, N)
        neg_batch_obj_tool_mask_2 = obj_tool_mask_2 * (~batch_mask)  # (B, N, N)

        adj_matrix[batch_obj_tool_mask_1] = 0  # Clear object-to-tool edges
        adj_matrix[batch_obj_tool_mask_2] = 1  # Add all tool-to-object edges
        adj_matrix[neg_batch_obj_tool_mask_1] = 0
        adj_matrix[neg_batch_obj_tool_mask_2] = 0
    else:
        adj_matrix[obj_tool_mask_1] = 0  # Clear object-to-tool edges

    # Combine with topological edges
    adj_matrix = adj_matrix + topological_edges
    adj_matrix = adj_matrix.clamp(0, 1)

    # Convert adjacency matrix to sparse edge representation
    n_rels = adj_matrix.sum(dim=(1,2))  # [B] - Number of edges per batch
    n_rel = n_rels.max().long().item()  # int - Maximum edges across batch
    rels_idx = [torch.arange(n_rels[i]) for i in range(B)] # list of [n_rel] - example: [tensor([0, 1, 2, 3]), tensor([0, 1, 2])]
    rels_idx = torch.hstack(rels_idx).to(device=states.device, dtype=torch.long) # [E_c] - example: tensor([0, 1, 2, 3, 0, 1, 2])
    rels = adj_matrix.nonzero()  # [E_c, 3] - each row is [batch_idx, receiver_idx, sender_idx]
    
    # Create receiver and sender matrices
    Rr = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rs = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rr[rels[:, 0], rels_idx, rels[:, 1]] = 1 # receiver matrix [B, n_rel, N]
    Rs[rels[:, 0], rels_idx, rels[:, 2]] = 1 # sender matrix [B, n_rel, N]
    
    # Extract edge attributes: 1 for topological edges, 0 for collision edges
    # Check if each edge is topological by looking up in the topological_edges matrix
    edge_attrs_flat = topological_edges[rels[:, 0], rels[:, 1], rels[:, 2]]  # [n_rels.sum()]
    
    # Reshape to match the sparse edge format (B, n_rel, 1)
    edge_attrs = torch.zeros((B, n_rel, 1), device=states.device, dtype=states.dtype)
    edge_attrs[rels[:, 0], rels_idx, 0] = edge_attrs_flat
    
    return Rr, Rs, edge_attrs



def construct_edges_from_states(states, adj_thresh, mask, tool_mask, topk, connect_tools_all):
    """
    Construct edges between particles based on distance and tool connectivity rules (NOT USED).
    
    Args:
        states: (N, state_dim) torch tensor - particle positions
        adj_thresh: float - distance threshold for connections
        mask: (N) torch tensor - true when index is a valid particle
        tool_mask: (N) torch tensor - true when index is a valid tool particle
        topk: int - maximum number of neighbors per particle
        connect_tools_all: bool - if True, connect all tool particles to all object particles
        
    Returns:
        Rr: (n_rel, N) torch tensor - receiver matrix for graph edges
        Rs: (n_rel, N) torch tensor - sender matrix for graph edges
    """
    N, state_dim = states.shape
    
    # Create pairwise particle combinations
    s_receiv = states[:, None, :].repeat(1, N, 1)
    s_sender = states[None, :, :].repeat(N, 1, 1)

    # Calculate distances and create adjacency matrix
    threshold = adj_thresh * adj_thresh
    s_diff = s_receiv - s_sender
    dis = torch.sum(s_diff ** 2, -1)
    
    # Apply validity masks
    mask_1 = mask[:, None].repeat(1, N)
    mask_2 = mask[None, :].repeat(N, 1)
    mask_12 = mask_1 * mask_2
    dis[~mask_12] = 1e10
    
    # Prevent tool-to-tool connections
    tool_mask_1 = tool_mask[:, None].repeat(1, N)
    tool_mask_2 = tool_mask[None, :].repeat(N, 1)
    tool_mask_12 = tool_mask_1 * tool_mask_2
    dis[tool_mask_12] = 1e10

    # Define interaction masks
    obj_tool_mask_1 = tool_mask_1 * mask_2
    obj_tool_mask_2 = tool_mask_2 * mask_1

    adj_matrix = ((dis - threshold) < 0).float()

    # Apply topk constraint
    topk = min(dis.shape[-1], topk)
    topk_idx = torch.topk(dis, k=topk, dim=-1, largest=False)[1]
    topk_matrix = torch.zeros_like(adj_matrix)
    topk_matrix.scatter_(-1, topk_idx, 1)
    adj_matrix = adj_matrix * topk_matrix

    # Handle tool connectivity
    if connect_tools_all:
        adj_matrix[obj_tool_mask_1] = 0  # Clear existing tool receiver connections
        adj_matrix[obj_tool_mask_2] = 1  # Connect all object particles to all tool particles
        adj_matrix[tool_mask_12] = 0     # Avoid tool to tool relations

    # Convert to sparse representation
    n_rels = adj_matrix.sum().long().item()
    rels_idx = torch.arange(n_rels).to(device=states.device, dtype=torch.long)
    rels = adj_matrix.nonzero()
    Rr = torch.zeros((n_rels, N), device=states.device, dtype=states.dtype)
    Rs = torch.zeros((n_rels, N), device=states.device, dtype=states.dtype)
    Rr[rels_idx, rels[:, 0]] = 1
    Rs[rels_idx, rels[:, 1]] = 1
    
    return Rr, Rs


def construct_edges_from_states_batch(states, adj_thresh, mask, tool_mask, topk, connect_tools_all):
    """
    Construct edges between particles based on distance and tool connectivity rules (batch version).
    
    Args:
        states: (B, N, state_dim) torch tensor - particle positions
        adj_thresh: float or (B,) torch tensor - distance threshold for connections
        mask: (B, N) torch tensor - true when index is a valid particle
        tool_mask: (B, N) torch tensor - true when index is a valid tool particle
        topk: int - maximum number of neighbors per particle
        connect_tools_all: bool - if True, connect all tool particles to all object particles
        
    Returns:
        Rr: (B, n_rel, N) torch tensor - receiver matrix for graph edges
        Rs: (B, n_rel, N) torch tensor - sender matrix for graph edges
    """
    B, N, state_dim = states.shape
    
    # Create pairwise particle combinations for distance calculation
    s_receiv = states[:, :, None, :].repeat(1, 1, N, 1)  # Receiver particles
    s_sender = states[:, None, :, :].repeat(1, N, 1, 1)   # Sender particles

    # Calculate squared distances between all particle pairs and the squared distance threshold
    if isinstance(adj_thresh, float):
        adj_thresh = torch.tensor(adj_thresh, device=states.device, dtype=states.dtype).repeat(B)
    threshold = adj_thresh * adj_thresh
    s_diff = s_receiv - s_sender  # Position differences
    dis = torch.sum(s_diff ** 2, -1)  # Squared distances (B, N, N)
    
    # Create validity masks for particle connections
    mask_1 = mask[:, :, None].repeat(1, 1, N)  # Receiver validity
    mask_2 = mask[:, None, :].repeat(1, N, 1)  # Sender validity
    mask_12 = mask_1 * mask_2  # Both particles are valid
    dis[~mask_12] = 1e10  # Exclude invalid particle pairs
    
    # Prevent tool-to-tool connections
    tool_mask_1 = tool_mask[:, :, None].repeat(1, 1, N)
    tool_mask_2 = tool_mask[:, None, :].repeat(1, N, 1)
    tool_mask_12 = tool_mask_1 * tool_mask_2
    dis[tool_mask_12] = 1e10  # Avoid tool to tool relations

    # Create adjacency matrix based on distance threshold
    adj_matrix = ((dis - threshold[:, None, None]) < 0).float()

    # Define tool-object interaction masks
    obj_tool_mask_1 = tool_mask_1 * mask_2  # Particle sender, tool receiver
    obj_tool_mask_2 = tool_mask_2 * mask_1  # Tool sender, particle receiver
    obj_pad_tool_mask_1 = tool_mask_1 * (~tool_mask_2) # Tool receiver, non-tool sender

    # Apply topk constraint to limit connections per particle
    topk = min(dis.shape[-1], topk)
    topk_idx = torch.topk(dis, k=topk, dim=-1, largest=False)[1]
    topk_matrix = torch.zeros_like(adj_matrix)
    topk_matrix.scatter_(-1, topk_idx, 1)
    adj_matrix = adj_matrix * topk_matrix

    # Handle tool connectivity rules
    if connect_tools_all:
        # Only connect tools to objects if there are neighboring tool - non-tool particles in batch
        batch_mask = (adj_matrix[obj_pad_tool_mask_1].reshape(B, -1).sum(-1) > 0)[:, None, None].repeat(1, N, N)
        batch_obj_tool_mask_1 = obj_tool_mask_1 * batch_mask  # (B, N, N)
        neg_batch_obj_tool_mask_1 = obj_tool_mask_1 * (~batch_mask)  # (B, N, N)
        batch_obj_tool_mask_2 = obj_tool_mask_2 * batch_mask  # (B, N, N)
        neg_batch_obj_tool_mask_2 = obj_tool_mask_2 * (~batch_mask)  # (B, N, N)

        adj_matrix[batch_obj_tool_mask_1] = 0  # Clear object-to-tool edges
        adj_matrix[batch_obj_tool_mask_2] = 1  # Add all tool-to-object edges
        adj_matrix[neg_batch_obj_tool_mask_1] = 0
        adj_matrix[neg_batch_obj_tool_mask_2] = 0
    else:
        adj_matrix[obj_tool_mask_1] = 0  # Clear object-to-tool edges

    # Convert adjacency matrix to sparse edge representation
    n_rels = adj_matrix.sum(dim=(1,2))  # [B] - Number of edges per batch
    n_rel = n_rels.max().long().item()  # int - Maximum edges across batch
    rels_idx = [torch.arange(n_rels[i]) for i in range(B)] # list of [n_rel] - example: [tensor([0, 1, 2, 3]), tensor([0, 1, 2])]
    rels_idx = torch.hstack(rels_idx).to(device=states.device, dtype=torch.long) # [n_rels.sum()] - example: tensor([0, 1, 2, 3, 0, 1, 2])
    rels = adj_matrix.nonzero()  # [n_rels.sum(), 3] - each row is [batch_idx, sender_idx, receiver_idx]
    
    # Create receiver and sender matrices
    Rr = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rs = torch.zeros((B, n_rel, N), device=states.device, dtype=states.dtype)
    Rr[rels[:, 0], rels_idx, rels[:, 1]] = 1 # receiver matrix [B, n_rel, N]
    Rs[rels[:, 0], rels_idx, rels[:, 2]] = 1 # sender matrix [B, n_rel, N]
    
    return Rr, Rs

