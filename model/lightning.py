import torch
import lightning as pl
import torch.nn.functional as F
from . import TransformerDynamics, Rollout
from torch.optim.lr_scheduler import ReduceLROnPlateau, StepLR, OneCycleLR

def calculate_total_steps(config):
    """
    Count the number of iterations across the entire training run. Used by OneCycleLR. 
    
    Args: 
        config: dict - configuration containing model hyperparameters

    Returns: 
        int - number of iterations
    """
    from math import ceil
    n_timestep = config['dataset']['n_timestep'] + 1
    n_episode = config['dataset']['n_episode']
    n_history = config['train']['n_history']
    n_rollout = config['train']['n_rollout']
    downsample = config['dataset']['downsample_rate']
    train_split = config['train']['train_valid_ratio']
    batch_size = config['dataset']['batch_size']['train']
    step_per_episode = n_timestep - (n_history + n_rollout - 1) * downsample

    dg_enabled = config['data_generation']['enabled']
    if dg_enabled:
        dg_start_epoch = config['data_generation']['start_epoch']
        dg_per_epoch = config['data_generation']['every_n_epoch']
        dg_n_episodes = config['data_generation']['n_episodes']
    else:
        dg_start_epoch = 1e8
        dg_per_epoch = 1
        dg_n_episodes = 0

    total_steps = 0
    for epoch in range(config['train']['n_epoch']):
        total_steps += ceil(n_episode * step_per_episode * train_split / batch_size)
        if epoch >= dg_start_epoch and (epoch - dg_start_epoch) % dg_per_epoch == 0:
            n_episode += dg_n_episodes
    
    return total_steps

class GNNLightning(pl.LightningModule):
    def __init__(self, config, lr=None, visualize=False, mc_dropout=False):
        """
        Args: 
            config: dict - configuration containing model hyperparameters
            lr: float - optional variable used by find_lr.py. 
            visualize: bool - whether test_step should return the full output for visualization. 
            mc_dropout: bool - whether to use dropout in predict_step
        """
        super().__init__()
        self.config = config

        if config['model']['type'] == 'Transformer':
            self.model = TransformerDynamics(config)
        elif config['model']['type'] == 'GNN':
            raise NotImplementedError("GNN model is not implemented in this demo")
            self.model = GNNDynamics(config)
        elif config['model']['type'] == 'old_GNN':
            raise NotImplementedError("GNN model is not implemented in this demo")
            self.model = PropNetDiffDenModel(config)
        else:
            raise ValueError(f"Invalid model type: {config['model']['type']}")
        
        self.save_hyperparameters(ignore=['config'])
        
        # Training parameters
        self.n_rollout = config['train']['n_rollout']
        self.n_history = config['train']['n_history']
        self.optimizer_config = config['train']['optimizer']
        self.scheduler_config = config['train']['scheduler']

        if lr is not None:  # for lr tuning
            self.optimizer_config['lr'] = lr
            self.lr = self.optimizer_config['lr'] 
        
        # Testing and prediction parameters
        self.visualize = visualize
        self.mc_dropout = mc_dropout    

    def _train_val_shared(self, batch):
        """
        Shared logic between training and validation
        Rollout for n_rollout steps
        Return just the loss
        """
        states, states_delta, attrs, particle_nums, topo_edges, first_states = batch

        B, T, N, _ = states.size()
        assert T == self.n_rollout + self.n_history

        loss = 0.0
        
        # Initialize rollout with first n_history frames
        rollout = Rollout(
            self.model, 
            self.config, 
            states[:, self.n_history - 1, :, :],      # [B, N, 3]
            states_delta[:, :self.n_history - 1, :, :], # [B, n_history - 1, N, 3]
            attrs[:, self.n_history - 1, :, :],       # [B, N, 2]
            particle_nums,           # [B]
            topo_edges,
            first_states,
        )

        for idx_step in range(self.n_rollout):
            # Get next frame data
            next_delta = states_delta[:, idx_step + self.n_history - 1, :, :]  # [B, N, 3]
            
            # Ground truth next state 
            s_nxt = states[:, self.n_history + idx_step, :, :]  # B x N x 3

            # Predict next state using rollout
            s_pred = rollout.forward(next_delta)  # [B, N, 3]

            # Warning: Doesn't account for padded particles. Will weigh samples with more particles more. 
            loss += F.mse_loss(s_pred, s_nxt)
            # In the future, when training on different objects, use this: 
            # loss += compute_mse_sum(s_pred, s_nxt, particle_nums)

        # Normalize loss by rollout steps
        return loss / self.n_rollout

    def training_step(self, batch, batch_idx):
        loss = self._train_val_shared(batch)
        
        # Log training loss
        # Scale the loss up to make logging nicer
        self.log('train_loss', torch.sqrt(loss) * 100, prog_bar=True)
        
        # Return the original loss, as magnitude affects learning rate
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self._train_val_shared(batch)
        
        # Log validation loss
        self.log('val_loss', torch.sqrt(loss) * 100, sync_dist=True, prog_bar=True)

        return loss
    
    def _test_predict_shared(self, batch):
        """
        Shared logic between test and prediction
        Rollout to the end of episode
        Return the predicted particle positions
        """
        states, states_delta, attrs, particle_nums, topo_edges, first_states = batch
        n_rollout = states_delta.shape[1] - self.n_history + 1
        
        # Initialize rollout with first n_history frames
        rollout = Rollout(
            self.model, 
            self.config, 
            states[:, self.n_history - 1, :, :],      # [B, N, 3]
            states_delta[:, :self.n_history - 1, :, :], # [B, n_history - 1, N, 3]
            attrs[:, self.n_history - 1, :, :],       # [B, N, 2]
            particle_nums,           # [B]
            topo_edges,
            first_states,
        )

        predicted_states = []
        for idx_step in range(n_rollout):
            # Get next frame data
            next_delta = states_delta[:, idx_step + self.n_history - 1, :, :]  # [B, N, 3]
            
            # Predict next state using rollout
            s_pred = rollout.forward(next_delta)  # [B, N, 3]
            predicted_states.append(s_pred)

        predicted_states = torch.stack(predicted_states, dim=1)  # [B, n_rollout, N, 3]
        return predicted_states

    def test_step(self, batch, batch_idx):
        predicted_states = self._test_predict_shared(batch)
        ground_truth = batch[0]
        step_loss = F.mse_loss(predicted_states, ground_truth[:, self.n_history:, :, :], reduction='none').mean(dim=(2, 3)) # [B, n_rollout]
        step_loss = torch.clamp(step_loss, max=0.1) # clamp loss to 10cm per step
        loss = step_loss.mean(dim=1) # [B]
        loss = torch.sqrt(loss) * 100 # already normalized by n_rollout
        if self.visualize:
            return {
                'loss': loss,
                'predicted_states': predicted_states,
                'states': ground_truth[:, self.n_history:, :, :],
                'attrs': batch[2],
                'particle_nums': batch[3],
                'topo_edges': batch[4],
                'first_states': batch[5],
            }
        else:
            return {
                'loss': loss,
            }

    def predict_step(self, batch, batch_idx):
        if self.mc_dropout:
            self.model.train()
        return self._test_predict_shared(batch)

    def configure_optimizers(self):
        if self.optimizer_config['type'] == 'AdamW':
            optimizer = torch.optim.AdamW(
                self.parameters(), 
                lr=self.optimizer_config['lr'],
                betas=(self.optimizer_config['beta1'], self.optimizer_config['beta2']),
                weight_decay=self.optimizer_config['weight_decay']
            )
        elif self.optimizer_config['type'] == 'Adam':
            optimizer = torch.optim.Adam(
                self.parameters(), 
                lr=self.optimizer_config['lr'],
                betas=(self.optimizer_config['beta1'], self.optimizer_config['beta2']),
            )
        else:
            raise ValueError(f"Invalid optimizer type: {self.optimizer_config['type']}")
        
        if self.scheduler_config.get('enabled', False):
            scheduler_config = {k: v for k, v in self.scheduler_config.items() 
                              if k not in ['enabled', 'type']}
            
            if self.scheduler_config['type'] == 'ReduceLROnPlateau':
                scheduler = ReduceLROnPlateau(optimizer, **scheduler_config)
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'monitor': 'val_loss',
                        'interval': 'epoch',
                        'frequency': 1
                    }
                }
            elif self.scheduler_config['type'] == 'StepLR':
                scheduler = StepLR(optimizer, **scheduler_config)
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'step',
                        'frequency': 1
                    }
                }
            elif self.scheduler_config['type'] == 'OneCycleLR':
                scheduler_config['total_steps'] = calculate_total_steps(self.config)
                scheduler = OneCycleLR(optimizer, **scheduler_config)
                return {
                    'optimizer': optimizer,
                    'lr_scheduler': {
                        'scheduler': scheduler,
                        'interval': 'step',
                        'frequency': 1
                    }
                }
            else:
                raise ValueError(f"Invalid scheduler type: {self.scheduler_config['type']}")
        
        return optimizer