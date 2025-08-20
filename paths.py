"""
Central path configuration for PhysTwin package.
All paths are calculated relative to the PhysTwin package location.
"""
import glob
import os

# Data directories
DATA_DIFFERENT_TYPES = "data/different_types"
DATA_BG_IMG = "data/bg.png"
GAUSSIAN_OUTPUT_DIR = "data/gaussian_output"

def get_model_paths(model_name):
    """
    Get paths for a specific GNN model.
    
    Args:
        model_name: str - name of the model
        
    Returns:
        dict - dictionary containing model-related paths
    """
    model_dir = "data/" + model_name
    checkpoint_files = glob.glob(os.path.join(model_dir, 'checkpoints', 'gnn_dyn-epoch=*.ckpt'))
    if checkpoint_files:
        best = min(checkpoint_files, key=lambda x: float(x.split('=')[-1].split('.')[0]))

    return {
        'model_dir': model_dir,
        'config': model_dir + "/config.yaml",
        'best_ckpt': best,
    } 