import torch
import torch.nn as nn

class TransformerDynamics(nn.Module):
    """
    Main Transformer module for particle dynamics prediction with history-aware processing.
    Handles multiple history timesteps and different particle types (objects vs robots/tools).
    """
    
    def __init__(self, config):
        """
        Args:
            config: dict - configuration containing model hyperparameters
        """
        super(TransformerDynamics, self).__init__()

        self.config = config
        self.latent_size = config['model']['latent_size']
        self.n_layers = config['model']['n_layers']
        self.n_heads = config['model']['n_heads']
        self.ff_size = config['model']['ff_size']
        self.dropout_rate = config['train']['dropout_rate']
        
        # History length for temporal modeling
        self.n_history = config['train']['n_history']
        
        # Particle embedding: displacement (3 * n_history) + attributes (2) + coordinates (3)
        self.particle_embedding = nn.Linear(3 * self.n_history + 5, self.latent_size)

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.latent_size,
            nhead=self.n_heads,
            dim_feedforward=self.ff_size,
            dropout=self.dropout_rate,
            activation='relu',
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=self.n_layers)
        
        # Output projection
        self.output_projection = nn.Linear(self.latent_size, 3)

    def predict_one_step(self, a_cur, s_cur, s_delta, mask):
        """
        Forward pass of the Transformer dynamics model.
        
        Args:
            a_cur: (B, particle_num, 2) - current particle attributes
            s_cur: (B, particle_num, 3) - current positions  
            s_delta: (B, particle_num, 3) - particle displacements over history
            
        Returns:
            (B, particle_num, 3) - predicted next particle positions
        """
        B, N, _ = a_cur.size()

        # Convert from data format (B x time x particles) to model format (B x particles x time)
        s_delta = s_delta.transpose(1, 2)  # B x particle_num x n_history x 3

        s_delta_flat = s_delta.reshape(B, N, -1)  # B x particle_num x (3 * n_history)
        s_center = s_cur.mean(dim=1, keepdim=True) # B x 1 x 3
        s_center[:,:,2] = 0. # keep z-coordinate because height is important
        s_cur_centered = s_cur - s_center
        input_features = torch.cat([s_delta_flat, a_cur, s_cur_centered], 2)  # B x particle_num x (3*n_history + 5)

        embedded = self.particle_embedding(input_features)  # B x particle_num x latent_size
        transformed = self.transformer(embedded, src_key_padding_mask=~mask)  # B x particle_num x latent_size
        velocity_pred = self.output_projection(transformed)  # B x particle_num x 3

        s_pred = velocity_pred + s_cur
        s_pred[~mask] = 0. # Mask out predictions for padded particles

        return s_pred