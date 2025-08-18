import numpy as np
import cv2 
from PIL import Image, ImageEnhance
import os
import torch
from .edges import construct_collision_edges, construct_topological_edges
import open3d as o3d
import pickle
import json
from pathlib import Path

# Optional Gaussian Splatting imports (lazy-enabled via Visualizer.enable_gaussian)
from gaussian_splatting.scene.gaussian_model import GaussianModel
from gaussian_splatting.scene.cameras import Camera
from gaussian_splatting.gaussian_renderer import render as render_gaussian
from gaussian_splatting.utils.graphics_utils import focal2fov
from gs_render import remove_gaussians_with_low_opacity
from gaussian_splatting.dynamic_utils import (
    interpolate_motions_speedup,
    knn_weights_sparse,
    get_topk_indices,
    calc_weights_vals_from_indices,
)

def visualize_edges(positions, topological_edges, tool_mask, adj_thresh, topk, connect_tools_all, colors):
    """
    Create edges visualization using construct_edges_with_attrs
    
    Args:
        positions: tensor [N, 3] - particle positions
        topological_edges: tensor [N, N] - topological adjacency matrix
        tool_mask: tensor [N] - boolean mask for tool particles
        adj_thresh: float - distance threshold for collision edges
        topk: int - maximum neighbors per particle
        connect_tools_all: bool - whether to connect tools to all objects
        colors: list of [r,g,b] colors for [collision, topological] edges
        
    Returns:
        o3d.geometry.LineSet - line set for visualization or None
    """
    
    # Add batch dimension for construct_edges_with_attrs
    positions = positions.unsqueeze(0).cpu()  # [N, 3] -> [1, N, 3]
    topological_edges = topological_edges.unsqueeze(0).cpu()  # [N, N] -> [1, N, N]
    tool_mask = tool_mask.unsqueeze(0).cpu()  # [N] -> [1, N]
    
    B, N, _ = positions.shape
    
    # Create mask for valid particles
    mask = torch.ones(B, N, dtype=torch.bool, device=positions.device)
    
    # Use construct_collision_edges to get edges
    Rr_collision, Rs_collision = construct_collision_edges(
        positions, adj_thresh, mask, tool_mask, 
        topk=topk, connect_tools_all=connect_tools_all, topological_edges=topological_edges
    )

    Rr_topo, Rs_topo, first_edge_lengths = construct_topological_edges(
        topological_edges, positions
    )
    
    # Create line sets for each edge type
    collision_lineset = create_lineset_from_Rr_Rs(Rr_collision, Rs_collision, colors[0], positions[0])
    topological_lineset = create_lineset_from_Rr_Rs(Rr_topo, Rs_topo, colors[1], positions[0])
    
    return [collision_lineset, topological_lineset]


def create_lineset_from_Rr_Rs(Rr, Rs, color, positions):
    """
    Create Open3D LineSet from sparse receiver/sender matrices
    
    Args:
        Rr: [B, n_rel, N] tensor - receiver matrix
        Rs: [B, n_rel, N] tensor - sender matrix  
        color: [r, g, b] color for edges
        positions: [N, 3] tensor - particle positions
        
    Returns:
        o3d.geometry.LineSet or None
    """
    # Convert sparse matrices to edge list
    # Find non-zero entries in Rr and Rs to get the actual edges
    rr_nonzero = Rr[0].nonzero()  # [n_edges, 2] where columns are [edge_idx, receiver_idx]
    rs_nonzero = Rs[0].nonzero()  # [n_edges, 2] where columns are [edge_idx, sender_idx]
            
    if len(rr_nonzero) > 0 and len(rs_nonzero) > 0:
        # Match edge indices to ensure we get correct sender-receiver pairs
        edge_indices = rr_nonzero[:, 0]  # Edge indices from receiver matrix
        receiver_indices = rr_nonzero[:, 1]  # Receiver particle indices
                
        # Find corresponding sender indices for the same edge indices
        rs_dict = {edge_idx.item(): sender_idx.item() for edge_idx, sender_idx in rs_nonzero}
        sender_indices = [rs_dict.get(edge_idx.item(), -1) for edge_idx in edge_indices]
                
        # Filter out any edges where sender index wasn't found
        valid_edges = [(s, r) for s, r in zip(sender_indices, receiver_indices.tolist()) if s != -1]
                
        if len(valid_edges) > 0:
            edges = np.array(valid_edges)
            positions = positions.cpu().numpy()
                    
            lineset = o3d.geometry.LineSet()
            lineset.points = o3d.utility.Vector3dVector(positions)
            lineset.lines = o3d.utility.Vector2iVector(edges)
            lineset.colors = o3d.utility.Vector3dVector(np.array([color] * len(edges)))
            return lineset
    
    return None


# Gaussian Splatting helper (composition-based backend)
class GaussianSplatRenderer:
    def __init__(self, intrinsics, w2c, width, height, gs_ply_path, white_bg=True, alpha_cutoff=100/255):
        self.white_bg = white_bg
        self.alpha_cutoff = alpha_cutoff

        self.gaussians = GaussianModel(sh_degree=3)
        self.gaussians.load_ply(gs_ply_path)
        self.gaussians = remove_gaussians_with_low_opacity(self.gaussians, 0.1)
        self.gaussians.isotropic = True

        # Camera/view setup (CUDA path like trainer_warp)
        K = torch.tensor(intrinsics, dtype=torch.float32, device="cuda")
        fx, fy = K[0, 0], K[1, 1]
        FoVx = focal2fov(fx, width)
        FoVy = focal2fov(fy, height)
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]
        self.view = Camera(
            (width, height),
            colmap_id="0000",
            R=R,
            T=T,
            FoVx=FoVx,
            FoVy=FoVy,
            depth_params=None,
            image=None,
            invdepthmap=None,
            image_name="0000",
            uid="0000",
            data_device="cuda",
            train_test_exp=None,
            is_test_dataset=None,
            is_test_view=None,
            K=K,
            normal=None,
            depth=None,
            occ_mask=None,
        )
        bg = [1, 1, 1] if white_bg else [0, 0, 0]
        self.background = torch.tensor(bg, dtype=torch.float32, device="cuda")

        # Dynamic motion binding state
        self.current_pos = self.gaussians.get_xyz
        self.current_rot = self.gaussians.get_rotation
        self.device = self.current_pos.device
        self.relations = None
        self.weights_indices = None
        self.K = 4

    @torch.no_grad()
    def bind_to_particles(self, prev_particle_pos):
        # Initialize GS motion binding using previous particle positions
        prev_particle_pos = prev_particle_pos.to(self.device)
        n_particles = prev_particle_pos.shape[0]
        K = min(self.K, n_particles-1)
        self.relations = get_topk_indices(prev_particle_pos, K=K)
        # Compute KNN indices between bones and gaussian centers
        _, self.weights_indices = knn_weights_sparse(prev_particle_pos, self.current_pos, K=K)

    @torch.no_grad()
    def update_from_particles(self, prev_particle_pos, cur_particle_pos):
        prev_particle_pos = prev_particle_pos.to(self.device)
        cur_particle_pos = cur_particle_pos.to(self.device)
        weights = calc_weights_vals_from_indices(prev_particle_pos, self.current_pos, self.weights_indices)
        new_pos, new_rot, _ = interpolate_motions_speedup(
            bones=prev_particle_pos,
            motions=cur_particle_pos - prev_particle_pos,
            relations=self.relations,
            weights=weights,
            weights_indices=self.weights_indices,
            xyz=self.current_pos,
            quat=self.current_rot,
        )
        # Persist updates
        self.current_pos = new_pos
        self.current_rot = new_rot
        self.gaussians._xyz = new_pos
        self.gaussians._rotation = new_rot

    @torch.no_grad()
    def render_rgb_and_mask(self):
        # Returns numpy uint8 RGB image and alpha channel in [0,1]
        results = render_gaussian(self.view, self.gaussians, None, self.background)
        image = results["render"].permute(1, 2, 0).detach().clamp(0, 1)  # H, W, 4 (RGBA)
        if self.white_bg:
            mask = torch.logical_and((image[..., :3] != 1.0).any(dim=-1), image[..., 3] > self.alpha_cutoff)
        else:
            mask = torch.logical_and((image[..., :3] != 0.0).any(dim=-1), image[..., 3] > self.alpha_cutoff)
        image[..., 3].masked_fill_(~mask, 0.0)
        alpha = image[..., 3].cpu().numpy()
        rgb = (image[..., :3] * 255).to(torch.uint8).cpu().numpy()
        return rgb, alpha


def gen_goal_shape(name, h, w, font_name='helvetica_thin'):
    """
    Generate goal shape from font.
    
    Args:
        name: str - character/shape name
        h, w: int - target height and width
        font_name: str - font name
        
    Returns:
        tuple: (goal_dist, goal_img) - distance transform and image
    """
    root_dir = f'env/target_shapes/{font_name}'
    shape_path = os.path.join(root_dir, 'helvetica_' + name + '.npy')
    goal = np.load(shape_path)
    goal = cv2.resize(goal, (w, h), interpolation=cv2.INTER_AREA)
    goal = (goal <= 0.5).astype(np.uint8)
    goal_dist = np.minimum(cv2.distanceTransform(1-goal, cv2.DIST_L2, 5), 1e4)
    goal_img = (goal * 255)[..., None].repeat(3, axis=-1).astype(np.uint8)
    return goal_dist, goal_img


# ============================================================================
# DEPRECATED
# ============================================================================

def create_edges_for_points(positions, distance_threshold):
    """
    Create connectivity edges between nearby particles for visualization.
        
    Args:
        positions: [n_points, 3] - particle positions
        distance_threshold: float - maximum distance for connections
            
    Returns:
        edges: [n_edges, 2] - indices of connected particle pairs
    """
    edges = []
    n_points = positions.shape[0]
        
    for i in range(n_points):
        for j in range(i + 1, n_points):
            distance = np.linalg.norm(positions[i] - positions[j])
            if distance <= distance_threshold:
                edges.append([i, j])
            
    return np.array(edges) if edges else np.empty((0, 2), dtype=int)


def drawRotatedRect(img, s, e, width=1):
    """
    Draw a rotated rectangle on image with color gradient.
    
    Args:
        img: (h, w, 3) numpy array - input image
        s: (x, y) tuple - start point
        e: (x, y) tuple - end point
        width: int - rectangle width
        
    Returns:
        numpy array - image with drawn rectangle
    """
    l = int(np.sqrt((s[0] - e[0]) ** 2 + (s[1] - e[1]) ** 2) + 1)
    theta = np.arctan2(e[1] - s[1], e[0] - s[0])
    theta_ortho = theta + np.pi / 2
    for i in range(l):
        color = (255, int(255 * i / l), 0)
        x = int(s[0] + (e[0] - s[0]) * i / l)
        y = int(s[1] + (e[1] - s[1]) * i / l)
        img = cv2.line(img.copy(), (int(x - 0.5 * width * np.cos(theta_ortho)), int(y - 0.5 * width * np.sin(theta_ortho))), 
                    (int(x + 0.5 * width * np.cos(theta_ortho)), int(y + 0.5 * width * np.sin(theta_ortho))), color, 1)
    return img


def drawPushing(img, s, e, width):
    """
    Draw pushing action visualization on image.
    
    Args:
        img: (h, w, 3) numpy array - input image
        s: (x, y) tuple - start point
        e: (x, y) tuple - end point
        width: float - line width
        
    Returns:
        numpy array - image with drawn pushing action
    """
    l = int(np.sqrt((s[0] - e[0]) ** 2 + (s[1] - e[1]) ** 2) + 1)
    theta = np.arctan2(e[1] - s[1], e[0] - s[0])
    theta_ortho = theta + np.pi / 2
    img = cv2.line(img.copy(), (int(s[0] - 0.5 * width * np.cos(theta_ortho)), int(s[1] - 0.5 * width * np.sin(theta_ortho))), 
                (int(s[0] + 0.5 * width * np.cos(theta_ortho)), int(s[1] + 0.5 * width * np.sin(theta_ortho))), (255,99,71), 5)
    img = cv2.line(img.copy(), (int(e[0] - 0.5 * width * np.cos(theta_ortho)), int(e[1] - 0.5 * width * np.sin(theta_ortho))),
                (int(e[0] + 0.5 * width * np.cos(theta_ortho)), int(e[1] + 0.5 * width * np.sin(theta_ortho))), (255,99,71), 5)
    img = cv2.arrowedLine(img.copy(), (int(s[0]), int(s[1])), (int(e[0]), int(e[1])), (255,99,71), 5)
    return img


def gt_rewards(mask, subgoal):
    """
    Calculate ground truth reward based on object mask and subgoal.
    
    Args:
        mask: (h, w) numpy array - object mask
        subgoal: (h, w) numpy array - subgoal region
        
    Returns:
        float - reward value
    """
    subgoal_mask = subgoal < 0.5
    obj_dist = cv2.distanceTransform(1 - mask.astype(np.uint8), cv2.DIST_L2, 5)
    return np.sum(mask * subgoal) / mask.sum() + np.sum(obj_dist * subgoal_mask) / subgoal_mask.sum()


def gt_rewards_norm_by_sum(mask, subgoal):
    """
    Calculate ground truth reward normalized by sum.
    
    Args:
        mask: (h, w) numpy array - object mask
        subgoal: (h, w) numpy array - subgoal region
        
    Returns:
        float - normalized reward value
    """
    subgoal_mask = subgoal < 0.5
    obj_dist = cv2.distanceTransform(1 - mask.astype(np.uint8), cv2.DIST_L2, 5)
    return np.sum(mask * subgoal) / subgoal.sum() + np.sum(obj_dist * subgoal_mask) / obj_dist.sum()


def gen_ch_goal(name, h, w):
    """
    Generate Chinese character goal shape.
    
    Args:
        name: str - character name
        h, w: int - target height and width
        
    Returns:
        tuple: (goal_dist, goal_img) - distance transform and image
    """
    root_dir = 'env/target_shapes/720_ch'
    shape_path = os.path.join(root_dir, name + '.npy')
    goal = np.load(shape_path)
    goal = cv2.resize(goal, (w, h), interpolation=cv2.INTER_AREA)
    goal = (goal <= 0.5).astype(np.uint8)
    goal_dist = cv2.distanceTransform(1-goal, cv2.DIST_L2, 5)
    goal_img = (goal * 255)[..., None].repeat(3, axis=-1).astype(np.uint8)
    return goal_dist, goal_img


def gen_subgoal(c_row, c_col, r, h=64, w=64):
    """
    Generate circular subgoal region.
    
    Args:
        c_row, c_col: int - center coordinates
        r: float - radius
        h, w: int - image dimensions
        
    Returns:
        tuple: (subgoal, mask) - distance transform and binary mask
    """
    mask = np.zeros((h, w))
    grid = np.mgrid[0:h, 0:w]
    grid[0] = grid[0] - c_row
    grid[1] = grid[1] - c_col
    dist = np.sqrt(np.sum(grid**2, axis=0))
    mask[dist < r] = 1
    subgoal = np.minimum(cv2.distanceTransform((1-mask).astype(np.uint8), cv2.DIST_L2, 5), 1e4)
    return subgoal, mask


# Color constants
dodger_blue_RGB = (30, 144, 255)
dodger_blue_BGR = (255, 144, 30)
tomato_RGB = (255, 99, 71)
tomato_BGR = (71, 99, 255)


def lighten_img(img, factor=1.2):
    # img: assuming an RGB image
    assert img.dtype == np.uint8
    assert img.shape[2] == 3
    cv2.imwrite('tmp_1.png', img)
    img = Image.open('tmp_1.png').convert("RGB")
    img_enhancer = ImageEnhance.Brightness(img)
    enhanced_output = img_enhancer.enhance(factor)
    enhanced_output.save("tmp_2.png")
    color_lighten_img = cv2.imread('tmp_2.png')
    os.system('rm tmp_1.png tmp_2.png')
    return color_lighten_img

class Visualizer:
    """
    Visualizer for particle trajectories with camera calibration.
    Handles 3D visualization of particle motion with proper camera setup.
    """
    
    def __init__(self, camera_calib_path, bg_img_path=None, downsample_rate=1, gs_path=None):
        """
        Initialize visualizer with camera calibration data.
        
        Args:
            camera_calib_path: str - path to camera calibration data directory
            bg_img_path: str - path to background image
            bg_img_path: str, optional - path to background image
        """
        # Load camera to world transforms
        with open(os.path.join(camera_calib_path, "calibrate.pkl"), "rb") as f:
            self.c2ws = pickle.load(f)
        self.w2cs = [np.linalg.inv(c2w) for c2w in self.c2ws]

        with open(os.path.join(camera_calib_path, "metadata.json"), "r") as f:
            data = json.load(f)
        self.intrinsics = np.array(data["intrinsics"])
        self.WH = data["WH"]
        self.FPS = data["fps"] / downsample_rate
        
        print(f"Loaded camera calibration from: {camera_calib_path}")
        print(f"Resolution: {self.WH}, FPS: {self.FPS}")

        self.gs_renderer = None
        self._gs_prev_obj = None
        self.gs_path = gs_path
        
        self.pcds = None
        self.line_sets = None
        self.bg_image = None

        if bg_img_path is not None:
            self.bg_image = cv2.imread(bg_img_path)
            self.bg_image = cv2.cvtColor(self.bg_image, cv2.COLOR_BGR2RGB)
    
    def create_gaussian(self, gs_path, object_pos, vis_cam_idx=0, white_bg=True, alpha_cutoff=100/255):
        """Enable Gaussian Splatting background rendering for this visualizer."""
        # Pick intrinsics for the selected camera
        if hasattr(self.intrinsics, "shape") and len(self.intrinsics.shape) > 1:
            intr = self.intrinsics[vis_cam_idx]
        else:
            intr = self.intrinsics
        self.gs_renderer = GaussianSplatRenderer(
            intrinsics=intr,
            w2c=self.w2cs[vis_cam_idx],
            width=self.WH[0],
            height=self.WH[1],
            gs_ply_path=gs_path,
            white_bg=white_bg,
            alpha_cutoff=alpha_cutoff,
        )
        self.gs_renderer.bind_to_particles(object_pos)
        self._gs_prev_obj = object_pos.detach().clone()

    def update_gaussian(self, object_pos):
        assert self.gs_renderer is not None, "Gaussian renderer not created"
        self.gs_renderer.update_from_particles(self._gs_prev_obj, object_pos)
        self._gs_prev_obj = object_pos.detach().clone()
    
    def create_edges(self, object_pos, robot_pos, topo_edges, colli_adj, colli_topk):
        self.topo_edges = topo_edges
        self.colli_adj = colli_adj
        self.colli_topk = colli_topk
        self.line_sets = []

        # Create combined state and tool mask for edge computation
        combined_pos = torch.cat([object_pos, robot_pos], dim=0)
        tool_mask = torch.cat([
            torch.zeros(len(object_pos), dtype=torch.bool, device=object_pos.device),
            torch.ones(len(robot_pos), dtype=torch.bool, device=robot_pos.device)
        ], dim=0)
        
        self.line_sets = visualize_edges(
            combined_pos, self.topo_edges, tool_mask,
            adj_thresh=self.colli_adj, topk=self.colli_topk, connect_tools_all=False,
            colors=[[1.0, 0.6, 0.2], [0.3, 0.6, 0.3]]  # light orange, light green
        )
        for line_set in self.line_sets:
            if line_set is not None:
                self.vis.add_geometry(line_set)

    def update_edges(self, object_pos, robot_pos):
        # Remove old edges
        assert self.line_sets is not None, "Edges not enabled"
        for line_set in self.line_sets:
            if line_set is not None:
                self.vis.remove_geometry(line_set, reset_bounding_box=False)
        
        # Create new edges with updated positions
        combined_pos = torch.cat([object_pos, robot_pos], dim=0).cpu()
        tool_mask = torch.cat([
            torch.zeros(len(object_pos), dtype=torch.bool),
            torch.ones(len(robot_pos), dtype=torch.bool)
        ], dim=0)
        
        self.line_sets = visualize_edges(
            combined_pos, self.topo_edges, tool_mask,
            adj_thresh=self.colli_adj, topk=self.colli_topk, connect_tools_all=False,
            colors=[[1.0, 0.6, 0.2], [0.3, 0.6, 0.3]]  # light orange, light green
        )
        for line_set in self.line_sets:
            if line_set is not None:
                self.vis.add_geometry(line_set, reset_bounding_box=False)

    def create_pcd(self, pcds, colors):
        """
        Create o3d point clouds

        Args: 
            pcds: dict of {name: [n_points, 3]} 
            colors: dict of {name: [3]}
        """
        self.pcds = {}
        for name, pcd in pcds.items():
            assert name in colors, f"Color for {name} not found"
            if isinstance(pcd, torch.Tensor):
                pcd = pcd.cpu().numpy()
            self.pcds[name] = o3d.geometry.PointCloud()
            self.pcds[name].points = o3d.utility.Vector3dVector(pcd)
            self.pcds[name].paint_uniform_color(colors[name])
            self.vis.add_geometry(self.pcds[name])

    def update_pcd(self, pcds):
        """
        Update point clouds

        Args:
            pcds: dict of {name: [n_points, 3]} 
        """
        assert self.pcds is not None, "Point clouds not created"
        for name, pcd in pcds.items():
            if name not in self.pcds:
                continue
            if isinstance(pcd, torch.Tensor):
                pcd = pcd.cpu().numpy()
            self.pcds[name].points = o3d.utility.Vector3dVector(pcd)
            self.vis.update_geometry(self.pcds[name])
        
    def set_cam_params(self):
        # Set up camera parameters
        view_control = self.vis.get_view_control()
        camera_params = o3d.camera.PinholeCameraParameters()
        intrinsic_parameter = o3d.camera.PinholeCameraIntrinsic(
            self.WH[0], self.WH[1], self.intrinsics[0] if len(self.intrinsics.shape) > 1 else self.intrinsics
        )
        camera_params.intrinsic = intrinsic_parameter
        camera_params.extrinsic = self.w2cs[0]
        view_control.convert_from_pinhole_camera_parameters(
            camera_params, allow_arbitrary=True
        )

    def init_o3d_visualizer(self, pcds, colors, topo_edges=None, colli_adj=None, colli_topk=None):
        assert "object" in pcds and "robot" in pcds, "pcds must contain object and robot"
        
        self.vis = o3d.visualization.Visualizer()
        self.vis.create_window(width=self.WH[0], height=self.WH[1], visible=False)

        self.create_pcd(pcds, colors)
        if topo_edges is not None:
            object_pos = pcds["object"]
            robot_pos = pcds["robot"]
            self.create_edges(object_pos, robot_pos, topo_edges, colli_adj, colli_topk)
        if self.gs_path is not None:
            self.create_gaussian(self.gs_path, pcds["object"])

        self.set_cam_params()

    def render_frame(self, pcds):
        if self.pcds is not None:
            self.update_pcd(pcds)
        if self.line_sets is not None:
            object_pos = pcds["object"]
            robot_pos = pcds["robot"]
            self.update_edges(object_pos, robot_pos)
        
        # Render the scene
        self.vis.poll_events()
        self.vis.update_renderer()

        o3d_rgb = np.asarray(self.vis.capture_screen_float_buffer(do_render=True))
        o3d_rgb = (o3d_rgb * 255).astype(np.uint8)
        o3d_nonwhite = ~np.all(o3d_rgb == [255, 255, 255], axis=-1)

        # Base frame: start from background if available, else white
        if self.bg_image is not None:
            frame = self.bg_image.copy()
        else:
            frame = np.full_like(o3d_rgb, 255, dtype=np.uint8)

        # Blend Gaussian Splatting over base frame (alpha blending like trainer_warp)
        if self.gs_renderer is not None:
            self.update_gaussian(pcds["object"])  # move GS with object
            gs_rgb, gs_alpha = self.gs_renderer.render_rgb_and_mask()  # rgb uint8, alpha float [0,1]
            frame_f = gs_alpha[..., None] * gs_rgb.astype(np.float32) + (1.0 - gs_alpha[..., None]) * frame.astype(np.float32)
            frame = frame_f.astype(np.uint8)

        # Overlay O3D on top of GS/background (non-white pixels)
        frame[o3d_nonwhite] = o3d_rgb[o3d_nonwhite]

        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

class VideoVisualizer(Visualizer):
    def visualize_object_motion(self, predicted_states, tool_mask, actual_objects, save_path, target=None, topo_edges=None, colli_adj=0.12, colli_topk=10):
        """
        Create 3D visualization comparing predicted vs actual object motion.
        Renders particles as colored point clouds with connectivity edges.
        
        Args:
            predicted_states: [timesteps, n_particles, 3] tensor - predicted trajectory (objects + robots)
            tool_mask: [n_particles] tensor - boolean mask (False=object, True=robot)
            actual_objects: [timesteps, n_obj, 3] tensor - ground truth object trajectory  
            save_path: str - output video file path
            target: [N, 3] tensor - target point cloud for MPC
        Returns:
            save_path: str - path where video was saved
        """        
        # Video parameters
        fourcc = cv2.VideoWriter_fourcc(*'avc1')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        out = cv2.VideoWriter(save_path, fourcc, self.FPS, (self.WH[0], self.WH[1]))
        
        # Move tensors to CPU
        predicted_states = predicted_states.detach().cpu()
        tool_mask = tool_mask.detach().cpu()
        
        # Split predicted states into objects and robots using tool_mask
        pred_objects = predicted_states[:, ~tool_mask, :]
        pred_robots = predicted_states[:, tool_mask, :]

        pcds = {
            "object": pred_objects[0],
            "robot": pred_robots[0],
        }
        if actual_objects is not None:
            pcds["actual"] = actual_objects[0]
        if target is not None:
            pcds["target"] = target
        colors = {
            "object": [0.0, 1.0, 0.0],
            "robot": [1.0, 0.0, 0.0],
            "actual": [0.0, 0.0, 1.0],
            "target": [1.0, 0.6, 0.2],
        }

        self.init_o3d_visualizer(
            pcds=pcds,
            colors=colors,
            topo_edges=topo_edges,
            colli_adj=colli_adj, 
            colli_topk=colli_topk,
        )

        n_frames = len(predicted_states)
        print(f"Rendering {n_frames} frames...")

        # Render each frame
        for frame_idx in range(n_frames):
            # Update particle positions
            pcds = {
                "object": pred_objects[frame_idx],
                "robot": pred_robots[frame_idx],
            }
            if actual_objects is not None:
                pcds["actual"] = actual_objects[frame_idx]
            # target doesn't need to be updated
           
            out.write(self.render_frame(pcds))
                        
            if frame_idx % 10 == 0:
                print(f"  Rendered frame {frame_idx}/{n_frames}")
        
        # Cleanup
        out.release()
        self.vis.destroy_window()
        
        print(f"Video saved to: {save_path}")
        return save_path

class InteractiveVisualizer(Visualizer):
    def init_control_ui(self):
        """Initialize UI control elements for interactive mode"""
        self.arrow_size = 30
        
        # Load arrow assets from PhysTwin
        self.arrow_empty_orig = cv2.imread("data/arrow_empty.png", cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0, 3]]
        self.arrow_1_orig = cv2.imread("data/arrow_1.png", cv2.IMREAD_UNCHANGED)[:, :, [2, 1, 0, 3]]
        
        width, height = self.WH
        spacing = self.arrow_size + 5
        self.bottom_margin = 25
        bottom_y = height - self.bottom_margin
        top_y = height - self.bottom_margin - spacing
        self.edge_buffer = self.bottom_margin
        set1_margin_x = self.edge_buffer
        
        # Control UI positions (simplified for one control set)
        self.arrow_positions = {
            "q": (set1_margin_x + spacing * 3, top_y),    # Up
            "w": (set1_margin_x + spacing, top_y),       # Forward  
            "a": (set1_margin_x, bottom_y),              # Left
            "s": (set1_margin_x + spacing, bottom_y),    # Backward
            "d": (set1_margin_x + spacing * 2, bottom_y), # Right
            "e": (set1_margin_x + spacing * 3, bottom_y), # Down
        }
        
        # Pre-compute rotated arrows
        self.interm_size = 512
        self.rotations = {
            "w": cv2.getRotationMatrix2D((self.interm_size // 2, self.interm_size // 2), 0, 1),
            "a": cv2.getRotationMatrix2D((self.interm_size // 2, self.interm_size // 2), 90, 1),
            "s": cv2.getRotationMatrix2D((self.interm_size // 2, self.interm_size // 2), 180, 1),
            "d": cv2.getRotationMatrix2D((self.interm_size // 2, self.interm_size // 2), 270, 1),
            "q": cv2.getRotationMatrix2D((self.interm_size // 2, self.interm_size // 2), 0, 1),
            "e": cv2.getRotationMatrix2D((self.interm_size // 2, self.interm_size // 2), 180, 1),
        }
        
        self.arrow_rotated_filled = {}
        self.arrow_rotated_empty = {}
        for key in self.arrow_positions:
            self.arrow_rotated_filled[key] = cv2.resize(
                self._rotate_arrow(
                    cv2.resize(self.arrow_1_orig, (self.interm_size, self.interm_size), interpolation=cv2.INTER_AREA),
                    key,
                ),
                (self.arrow_size, self.arrow_size), interpolation=cv2.INTER_AREA,
            )
            self.arrow_rotated_empty[key] = cv2.resize(
                self._rotate_arrow(
                    cv2.resize(self.arrow_empty_orig, (self.interm_size, self.interm_size), interpolation=cv2.INTER_AREA),
                    key,
                ),
                (self.arrow_size, self.arrow_size), interpolation=cv2.INTER_AREA,
            )

    def _rotate_arrow(self, arrow, key):
        """Rotate arrow image for UI display"""
        rotation_matrix = self.rotations[key]
        rotated = cv2.warpAffine(
            arrow, rotation_matrix, (self.interm_size, self.interm_size),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_TRANSPARENT,
        )
        return rotated

    def _overlay_arrow(self, background, position, key, filled=True):
        """Overlay arrow on background image"""
        x, y = position
        
        if filled:
            rotated_arrow = self.arrow_rotated_filled[key].copy()
        else:
            rotated_arrow = self.arrow_rotated_empty[key].copy()
        
        h, w = rotated_arrow.shape[:2]
        
        roi_x = max(0, x - w // 2)
        roi_y = max(0, y - h // 2)
        roi_w = min(w, background.shape[1] - roi_x)
        roi_h = min(h, background.shape[0] - roi_y)
        
        arrow_x = max(0, w // 2 - x)
        arrow_y = max(0, h // 2 - y)
        
        roi = background[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w]
        arrow_roi = rotated_arrow[arrow_y : arrow_y + roi_h, arrow_x : arrow_x + roi_w]
        
        alpha = arrow_roi[:, :, 3] / 255.0
        
        for c in range(3):
            roi[:, :, c] = roi[:, :, c] * (1 - alpha) + arrow_roi[:, :, c] * alpha
        
        background[roi_y : roi_y + roi_h, roi_x : roi_x + roi_w] = roi
        return background

    def update_frame_with_ui(self, frame, pressed_keys):
        """Add UI overlay to frame based on pressed keys from external input handler"""
        result = frame.copy()
        
        # Add transparent overlay for controls
        trans_width = 160
        trans_height = 120
        overlay = result.copy()
        
        bottom_left_pt1 = (0, self.WH[1] - trans_height)
        bottom_left_pt2 = (trans_width, self.WH[1])
        cv2.rectangle(overlay, bottom_left_pt1, bottom_left_pt2, (255, 255, 255), -1)
        
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, result, 1 - alpha, 0, result)
        
        # Draw control arrows
        for key, pos in self.arrow_positions.items():
            if key in pressed_keys:
                result = self._overlay_arrow(result, pos, key, filled=True)
            else:
                result = self._overlay_arrow(result, pos, key, filled=False)
        
        # Add control labels
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        control_x = self.edge_buffer
        text_y = self.WH[1] - self.arrow_size * 2 - self.bottom_margin - 10
        cv2.putText(result, "Robot Control", (control_x, text_y), font, font_scale, (0, 0, 0), thickness)
        
        return result