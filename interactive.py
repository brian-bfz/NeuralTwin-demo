import argparse
import numpy as np
import torch
import cv2
from pynput import keyboard
import time
from scipy.spatial.transform import Rotation as R
import h5py
import os

from utils import load_yaml, fps_rad_tensor, construct_edges_from_tensor, InteractiveVisualizer
from model.lightning import GNNLightning
from model.rollout import Rollout

from paths import DATA_DIFFERENT_TYPES, GAUSSIAN_OUTPUT_DIR, DATA_BG_IMG, get_model_paths

class PickStartPose:
    def __init__(self, input_file, device):
        assert os.path.exists(input_file), f"{input_file} does not exist. Please run generate_start_poses first."
        self.device = device
        
        with h5py.File(input_file, 'r') as f:
            self.object_states = f['object'][:]
            self.n_poses = self.object_states.shape[0]
            self.robot_states = f['robot'][:]
            if 'finger' in f:
                self.init_fingers = f['finger'][:]
            else:
                self.init_fingers = np.zeros(self.n_poses)
            self.first_states = torch.from_numpy(f['first_states'][:]).to(self.device)
    
    def __call__(self, idx=None):
        if idx is None:
            from random import randint
            idx = randint(0, self.n_poses - 1)
        object_state = torch.from_numpy(self.object_states[idx]).to(self.device)
        robot_state = torch.from_numpy(self.robot_states[idx]).to(self.device)
        init_finger = self.init_fingers[idx]
        return object_state, robot_state, self.first_states, init_finger

class GNNPlayground: 
    def __init__(self, model_path, config, device, case_name, motion, data_base_path=str(DATA_DIFFERENT_TYPES)):
        self.model_path = model_path
        self.config = config  
        self.device = device
        self.case_name = case_name
        self.motion = motion
        self.data_base_path = data_base_path
        self.n_ctrl_parts = 1  # Only support 1 control part as requested
        self.inv_ctrl = 1.0  # No inverse control by default
        self.pressed_keys = set()
        self.robot_center = None  # Will be calculated from robot particles
        self.downsample_rate = config['dataset']['downsample_rate']
        
        # Initialize visualizer
        camera_calib_path = f"{self.data_base_path}/{self.case_name}"
        gaussian_path = f"{str(GAUSSIAN_OUTPUT_DIR)}/{self.case_name}/init=hybrid_iso=True_ldepth=0.001_lnormal=0.0_laniso_0.0_lseg=1.0/point_cloud/iteration_10000/point_cloud.ply"
        self.visualizer = InteractiveVisualizer(camera_calib_path, bg_img_path=str(DATA_BG_IMG), downsample_rate=self.downsample_rate, gs_path=gaussian_path)
        
        self.load_model()

    def load_model(self):
        module = GNNLightning.load_from_checkpoint(self.model_path, config=self.config, visualize=False)
        self.model = module.model
        self.model.to(self.device)
        self.model.eval()

    def downsample_and_topo_edges(self, object_state, robot_state, first_states):
        fps_radius = self.config['dataset']['fps_radius']

        object_indices = fps_rad_tensor(object_state, fps_radius)
        robot_indices = fps_rad_tensor(robot_state, fps_radius)
        shifted_robot_indices = robot_indices + object_state.shape[0]
        combined_indices = torch.cat([object_indices, shifted_robot_indices], dim=0)

        n_object = object_indices.shape[0]
        n_robot = robot_indices.shape[0]
        particle_num = n_object + n_robot

        topo_edges = None
        if self.config['edges']['topological']['enabled']:
            topo_edges = torch.zeros(particle_num, particle_num, device=self.device) # [particles, particles]
            adj_thresh = self.config['edges']['topological']['adj_thresh']
            topk = self.config['edges']['topological']['topk']
            object_edges = construct_edges_from_tensor(object_state[object_indices], adj_thresh, topk)
            topo_edges[:n_object, :n_object] = object_edges
        
        return object_state[object_indices], robot_state[robot_indices], topo_edges, first_states[combined_indices]

    def init_model(self, object_state, robot_state, topo_edges, first_states):
        n_history = self.config['train']['n_history']
        n_object = object_state.shape[0]
        n_robot = robot_state.shape[0]
        particle_num = n_object + n_robot

        states = torch.cat([object_state, robot_state], dim=0) # [particles, 3]
        states_delta = torch.zeros(n_history - 1, particle_num, 3, device=self.device) # [n_history - 1, particles, 3]

        self.robot_mask = torch.cat([torch.zeros(n_object, dtype=torch.bool, device=self.device), torch.ones(n_robot, dtype=torch.bool, device=self.device)], dim=0)
        attrs = torch.zeros(particle_num, 2, device=self.device)
        attrs[self.robot_mask, 0] = 1.0   # 0=object, 1=robot
        attrs[self.robot_mask, 1] = (self.motion == 'lift') # 0=push, 1=lift for robot, 0 for object
        attrs = attrs.float()

        self.rollout = Rollout(
            self.model,
            self.config,
            states.unsqueeze(0),
            states_delta.unsqueeze(0),
            attrs.unsqueeze(0),
            torch.tensor([particle_num], device=self.device),
            topo_edges.unsqueeze(0) if topo_edges is not None else None,
            first_states.unsqueeze(0) if first_states is not None else None,
        )

    def on_press(self, key):
        try:
            self.pressed_keys.add(key.char)
        except AttributeError:
            pass

    def on_release(self, key):
        try:
            self.pressed_keys.remove(key.char)
        except (KeyError, AttributeError):
            try:
                self.pressed_keys.remove(str(key))
            except KeyError:
                pass

    def get_target_change(self):
        target_change = np.zeros((self.n_ctrl_parts, 3))
        for key in self.pressed_keys:
            if key in self.key_mappings:
                if key == "q" or key == "w" or key == "e" or key == "a" or key == "s" or key == "d":
                    idx, change = self.key_mappings[key]
                    target_change[idx] += change * self.downsample_rate
        return target_change

    def get_rot_change(self):
        for key in self.pressed_keys:
            if key in self.key_mappings:
                if key == "z" or key == "x" or key == "c" or key == "v":
                    return np.array(self.key_mappings[key]) * self.downsample_rate
        return np.zeros(3)
    
    def init_control_ui(self):
        print("GNN Interactive Controls:")
        print("- Translation: WASD (XY movement), QE (Z movement)")  
        print("- Rotation: ZXCV (rotation around different axes)")
        print("- 6: Save current state snapshot")
        print("- ESC: Exit")
        
        # Set up key mappings - consistent with PhysTwin trainer_warp.py
        self.key_mappings = {
            # Translational controls
            "w": (0, np.array([0.005, 0, 0]) * self.inv_ctrl),
            "s": (0, np.array([-0.005, 0, 0]) * self.inv_ctrl),
            "a": (0, np.array([0, -0.005, 0]) * self.inv_ctrl),
            "d": (0, np.array([0, 0.005, 0]) * self.inv_ctrl),
            "e": (0, np.array([0, 0, 0.005])),
            "q": (0, np.array([0, 0, -0.005])),
            # Rotational controls
            "z": [0, 0, 2.0 / 180 * np.pi],
            "x": [0, 0, -2.0 / 180 * np.pi],
            "c": [2.0 / 180 * np.pi, 0, 0],
            "v": [-2.0 / 180 * np.pi, 0, 0],
        }

        self.visualizer.init_control_ui()


    def apply_robot_transformation(self, current_state, translation, rotation):
        """
        Apply translation and rotation directly to robot particles.
        Rotation is applied around the robot's center.
        
        Args:
            current_state: [particles, 3] current positions 
            translation: [3] translation vector
            rotation: [3] rotation angles (rx, ry, rz) in radians
            
        Returns:
            delta: [particles, 3] change in position
        """
        new_state = current_state.clone()
        
        # Apply translation to robot particles
        new_state[self.robot_mask] += torch.tensor(translation, device=self.device, dtype=torch.float32)
        
        # Apply rotation around robot center if rotation is non-zero
        if np.any(rotation != 0):
            robot_positions = new_state[self.robot_mask]
            robot_center = robot_positions.mean(dim=0)
            
            # Convert rotation angles to rotation matrix
            rotation_matrix = torch.tensor(
                R.from_euler('xyz', rotation).as_matrix(), 
                device=self.device, 
                dtype=torch.float32
            )
            
            # Translate to origin, rotate, translate back
            centered_positions = robot_positions - robot_center
            rotated_positions = torch.matmul(centered_positions, rotation_matrix.T)
            new_state[self.robot_mask] = rotated_positions + robot_center
        
        return new_state
    
    def update_gnn_state(self, delta):
        """Update the GNN rollout with new robot state"""
        next_delta = delta.unsqueeze(0)
        
        # Update rollout with new state
        with torch.no_grad():
            predicted_state = self.rollout.forward(next_delta).squeeze(0)
            
        return predicted_state
    
    def interact(self, object_state, robot_state, first_states, virtual_key=False):
        """Main interactive loop for GNN playground"""
        # Initialize model and UI
        object_state_gnn, robot_state_gnn, topo_edges, first_states_gnn = self.downsample_and_topo_edges(object_state, robot_state, first_states)
        self.init_model(object_state_gnn, robot_state_gnn, topo_edges, first_states_gnn)
        pcds = { # Use first states to initialize point clouds and gaussians
            "object": first_states_gnn[:object_state_gnn.shape[0]],
            "robot": first_states_gnn[object_state_gnn.shape[0]:],
        }
        colors = {
            "object": [0.0, 1.0, 0.0],
            "robot": [1.0, 0.0, 0.0],
        }
        if self.config['edges']['topological']['enabled']:
            self.visualizer.init_o3d_visualizer(pcds, colors, topo_edges, self.config['edges']['collision']['adj_thresh'], self.config['edges']['collision']['topk'])
        else:
            self.visualizer.init_o3d_visualizer(pcds, colors)
        self.init_control_ui()
        
        # Initialize virtual key tracking if enabled
        if virtual_key:
            self.virtual_keys = {}  # Dictionary to track virtual keys with timestamps
            self.virtual_key_duration = 0.03  # Virtual key press duration in seconds
        
        # Set up keyboard listener
        listener = keyboard.Listener(on_press=self.on_press, on_release=self.on_release)
        listener.start()
        
        # Initialize current state
        current_state = self.rollout.first_states.squeeze(0).clone()
        
        # Main interaction loop
        print("Starting interactive session...")
        print("Press ESC to exit")
        
        while True:
            # Get user input
            translation_change = self.get_target_change()[0]  # Only one control part
            rotation_change = self.get_rot_change()
            
            # Apply transformations if there are any changes
            if np.any(translation_change != 0) or np.any(rotation_change != 0):
                # Apply robot transformation
                new_state = self.apply_robot_transformation(
                    current_state, translation_change, rotation_change
                )
                
                # Get GNN prediction
                predicted_state = self.update_gnn_state(new_state - current_state)
                
                # Update current state to new state
                current_state = new_state
                
                print(f"Applied translation: {translation_change}, rotation: {rotation_change}")
            else:
                # No changes, just predict from current state
                predicted_state = self.update_gnn_state(torch.zeros(current_state.shape[0], 3, device=self.device))
            
            # Render interactive frame with UI overlay
            pcds = {
                "object": predicted_state[~self.robot_mask],
                "robot": predicted_state[self.robot_mask],
            }
            frame = self.visualizer.render_frame(pcds)
            frame = self.visualizer.update_frame_with_ui(frame, self.pressed_keys)
            cv2.imshow("GNN Interactive Playground", frame)
            
            # Handle save snapshot
            if "6" in self.pressed_keys:
                self.pressed_keys.discard("6")
                timestamp = int(time.time())
                object_pos = current_state[~self.robot_mask].cpu().numpy()
                robot_pos = current_state[self.robot_mask].cpu().numpy()
                
                np.save(f"gnn_snapshot_object_{timestamp}.npy", object_pos)
                np.save(f"gnn_snapshot_robot_{timestamp}.npy", robot_pos)
                print(f"Saved snapshot: gnn_snapshot_*_{timestamp}.npy")
            
            # Handle OpenCV window events
            key = cv2.waitKey(1) & 0xFF
            
            if virtual_key:
                # Handle virtual keyboard input through OpenCV window
                if key != -1:
                    key_char = chr(key & 0xFF).lower()
                    if key_char in self.key_mappings:
                        # Store virtual key with timestamp - refresh timestamp if already pressed
                        self.virtual_keys[key_char] = time.time()
                        self.pressed_keys.add(key_char)
                    elif key == 27:  # ESC key to exit
                        break
                
                # Process all keyboard inputs (both physical and virtual)
                # For virtual keys, check if they're still active based on timestamp
                current_time = time.time()
                keys_to_remove = []
                for k, press_time in self.virtual_keys.items():
                    if current_time - press_time > self.virtual_key_duration:
                        keys_to_remove.append(k)
                
                # Remove expired virtual keys
                for k in keys_to_remove:
                    if k in self.pressed_keys:
                        self.pressed_keys.discard(k)
                    if k in self.virtual_keys:
                        del self.virtual_keys[k]
            else:
                # Normal mode - just handle ESC
                if key == 27:  # ESC key
                    break
            
            # Small delay to control frame rate 
            time.sleep(1.0 / self.visualizer.FPS)
                
        listener.stop()
        cv2.destroyAllWindows()



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model name (e.g., 2025-05-31-21-01-09-427982)")
    parser.add_argument("--config", type=str, required=None, help="Path to model config file. Default: the config file in the model directory")
    parser.add_argument("--case_name", type=str, default=None, help="Case name. Default: the case name in the config file")
    parser.add_argument("--motion", type=str, choices=["push", "lift"], required=True, help="Push or lift motion")
    parser.add_argument("--start_pose", type=int, default=None, help="Start pose index. Default: random")
    parser.add_argument("--virtual_key", action="store_true", help="Use virtual key input")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_paths = get_model_paths(args.model)
    model_path = str(model_paths['best_ckpt'])
    
    if args.config is None:
        config_path = str(model_paths['config'])
    else:
        config_path = args.config
    config = load_yaml(config_path)
    
    if args.case_name is None:
        case_name = config["data_generation"]["case_name"]
    else:
        case_name = args.case_name

    motion = args.motion
    if motion == "push":
        pose_file = f"data/{case_name}/push_poses.h5"
    elif motion == "lift":
        pose_file = f"data/{case_name}/lift_poses.h5"
    else:
        raise ValueError(f"Invalid motion: {motion}")
    
    gnn_playground = GNNPlayground(model_path, config, device, case_name, motion)
    pose_picker = PickStartPose(pose_file, device)
    object_state, robot_state, first_states, _ = pose_picker(args.start_pose)
    gnn_playground.interact(object_state, robot_state, first_states, args.virtual_key)
