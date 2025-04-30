from typing import List
import numpy as np
from torch import nn
from torch.nn import functional as F
import torch


def build_mlp(layers_dims: List[int]):
    layers = []
    for i in range(len(layers_dims) - 2):
        layers.append(nn.Linear(layers_dims[i], layers_dims[i + 1]))
        layers.append(nn.BatchNorm1d(layers_dims[i + 1]))
        layers.append(nn.ReLU(True))
    layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
    return nn.Sequential(*layers)


class MockModel(torch.nn.Module):
    """
    Does nothing. Just for testing.
    """

    def __init__(self, device="cuda", output_dim=256):
        super().__init__()
        self.device = device
        self.repr_dim = output_dim

    def forward(self, states, actions):
        """
        Args:
            During training:
                states: [B, T, Ch, H, W]
            During inference:
                states: [B, 1, Ch, H, W]
            actions: [B, T-1, 2]

        Output:
            predictions: [B, T, D]
        """
        B, T, _ = actions.shape

        return torch.randn((B, T + 1, self.repr_dim)).to(self.device)


class Prober(torch.nn.Module):
    def __init__(
        self,
        embedding: int,
        arch: str,
        output_shape: List[int],
    ):
        super().__init__()
        self.output_dim = np.prod(output_shape)
        self.output_shape = output_shape
        self.arch = arch

        arch_list = list(map(int, arch.split("-"))) if arch != "" else []
        f = [embedding] + arch_list + [self.output_dim]
        layers = []
        for i in range(len(f) - 2):
            layers.append(torch.nn.Linear(f[i], f[i + 1]))
            layers.append(torch.nn.ReLU(True))
        layers.append(torch.nn.Linear(f[-2], f[-1]))
        self.prober = torch.nn.Sequential(*layers)

    def forward(self, e):
        output = self.prober(e)
        return output


# Main Model implementation
class WallEncoder(nn.Module):
    """
    Encodes the environment/wall into an embedding vector.
    """
    def __init__(self, 
                input_shape=(1, 65, 65), 
                latent_dim=128, 
                conv_stride=2):
        super().__init__()
        
        self.conv_stride = conv_stride
        in_channels, h, w = input_shape
        
        # Calculate feature map dimensions after convolutions
        for _ in range(2):
            h = (h - 1) // conv_stride + 1
            w = (w - 1) // conv_stride + 1
            
        # Convolutional layers
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=conv_stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=conv_stride, padding=1),
            nn.ReLU(),
        )
        
        # Final projection layer
        self.projection = nn.Linear(h * w * 64, latent_dim)
        self.flatten = nn.Flatten()
        
    def forward(self, x):
        x = x.squeeze(1)
        features = self.conv_net(x)
        flattened = self.flatten(features)
        embedding = self.projection(flattened)
        return embedding


class AgentEncoder(nn.Module):
    """
    Encodes agent observations into embeddings.
    """
    def __init__(self, 
                input_shape=(1, 65, 65), 
                latent_dim=128, 
                conv_stride=2):
        super().__init__()
        
        self.conv_stride = conv_stride
        self.latent_dim = latent_dim
        in_channels, h, w = input_shape
        
        # Calculate feature dimensions
        for _ in range(2):
            h = (h - 1) // conv_stride + 1
            w = (w - 1) // conv_stride + 1
            
        # Convolutional network
        self.conv_net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=conv_stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=conv_stride, padding=1),
            nn.ReLU(),
        )
        
        self.projection = nn.Linear(h * w * 64, latent_dim)
        self.flatten = nn.Flatten(start_dim=1)
        
    def forward(self, x):
        batch_size, timesteps, channels, h, w = x.shape
        
        # Reshape to process all timesteps at once
        reshaped_x = x.reshape(batch_size * timesteps, channels, h, w)
        
        features = self.conv_net(reshaped_x)
        flattened = self.flatten(features)
        embeddings = self.projection(flattened)
        
        # Restore batch and time dimensions
        return embeddings.reshape(batch_size, timesteps, -1)


class CombinedEncoder(nn.Module):
    """
    Combines agent observations with environment context.
    """
    def __init__(self, 
                env_dim=128, 
                output_dim=128):
        super().__init__()
        
        self.agent_encoder = AgentEncoder(latent_dim=output_dim)
        self.fusion_layer = nn.Linear(
            self.agent_encoder.latent_dim + env_dim, 
            output_dim
        )
    
    def forward(self, observations, env_context):
        # Encode agent observations
        agent_features = self.agent_encoder(observations)
        
        batch_size, timesteps, _ = agent_features.shape
        
        # Expand environment context to match timesteps
        expanded_env = env_context.unsqueeze(1).expand(-1, timesteps, -1)
        
        # Concatenate features
        combined = torch.cat([agent_features, expanded_env], dim=2)
        
        # Reshape for linear layer
        reshaped = combined.reshape(batch_size * timesteps, -1)
        
        # Apply fusion layer
        fused = self.fusion_layer(reshaped)
        
        # Restore shape
        return fused.reshape(batch_size, timesteps, -1)


class StatePredictor(nn.Module):
    """
    Predicts next state representation from current state and action.
    """
    def __init__(self, input_dim=128, output_dim=128):
        super().__init__()
        
        action_encoding_size = 16
        combined_size = action_encoding_size + input_dim
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(2, action_encoding_size),
            nn.ReLU()
        )
        
        # Prediction network
        self.predictor = nn.Sequential(
            nn.Linear(combined_size, output_dim * 4),
            nn.ReLU(),
            nn.Linear(output_dim * 4, output_dim)
        )
        
    def forward(self, state_embedding, action):
        # Encode action
        action_embedding = self.action_encoder(action)
        
        # Combine state and action
        combined = torch.cat([state_embedding, action_embedding], dim=1)
        
        # Predict next state
        prediction = self.predictor(combined)
        return prediction


def extract_nondiagonal(matrix):
    """
    Extract off-diagonal elements from a square matrix for VICReg covariance calculation.
    
    Args:
        matrix (Tensor): Square matrix [n, n]
        
    Returns:
        Tensor: Off-diagonal elements [(n-1)*n]
    """
    n, m = matrix.shape
    assert n == m
    flat = matrix.flatten()[:-1]
    return flat.view(n - 1, n + 1)[:, 1:].flatten()


class MainModel(nn.Module):
    """
    Joint Embedding Predictive Architecture (JEPA) world model
    that predicts state representations in a self-supervised manner.
    """
    def __init__(self, 
                embedding_size=128, 
                is_training=False):
        super().__init__()
        
        self.embedding_size = embedding_size
        self.is_training = is_training
        
        # Encoders
        self.environment_encoder = WallEncoder(latent_dim=embedding_size)
        self.state_encoder = CombinedEncoder(env_dim=embedding_size, output_dim=embedding_size)
        
        # Predictor
        self.next_state_predictor = StatePredictor(input_dim=embedding_size, output_dim=embedding_size)
    
    def forward(self, states, actions):
        """
        Process states and predict future state representations.
        
        Args:
            states: [batch_size, timesteps, channels, height, width]
            actions: [batch_size, timesteps-1, 2]
            
        Returns:
            Tuple containing environment context, predicted states,
            and target encoded states (if in training mode)
        """
        batch_size, _, _ = actions.shape
        
        # Split agent and wall channels
        agent_channel = states[:, :, 0:1, :, :]
        wall_channel = states[:, :, 1:2, :, :]
        
        # Encode environment (wall)
        env_embedding = self.environment_encoder(wall_channel[:, :1])
        
        # Encode initial state
        initial_state = self.state_encoder(agent_channel[:, :1], env_embedding)
        
        # Pre-compute target encodings for training
        target_encodings = None
        if self.is_training:
            target_encodings = self.state_encoder(agent_channel[:, 1:], env_embedding)
        
        # Autoregressive prediction of future states
        predictions = [initial_state[:, 0]]
        for t in range(actions.shape[1]):
            next_state = self.next_state_predictor(
                predictions[-1],
                actions[:, t]
            )
            predictions.append(next_state)
        
        # Stack predictions along time dimension
        predictions = torch.stack(predictions, dim=1)
        
        return env_embedding, predictions, target_encodings
        
    def loss(self, predictions, targets, env_context):
        """
        Compute VICReg-based loss function with invariance, variance and covariance terms.
        """
        # Exclude first prediction (initial state)
        predictions = predictions[:, 1:]
        
        batch_size, timesteps = predictions.shape[0], predictions.shape[1]
        
        # Invariance loss (MSE between predictions and targets)
        mse_loss = F.mse_loss(predictions, targets)
        
        # Reshape for variance and covariance computation
        flat_preds = predictions.reshape(batch_size * timesteps, -1)
        flat_targets = targets.reshape(batch_size * timesteps, -1)
        
        # Center the representations
        flat_preds = flat_preds - flat_preds.mean(dim=0)
        flat_targets = flat_targets - flat_targets.mean(dim=0)
        centered_env = env_context - env_context.mean(dim=0)
        
        # Compute standard deviations
        pred_std = torch.sqrt(flat_preds.var(dim=0) + 1e-4)
        target_std = torch.sqrt(flat_targets.var(dim=0) + 1e-4)
        env_std = torch.sqrt(centered_env.var(dim=0) + 1e-4)
        
        # Variance loss
        variance_loss = (
            torch.mean(F.relu(1 - pred_std)) / 2 +
            torch.mean(F.relu(1 - target_std)) / 2
        )
        
        # Environment variance loss
        env_variance_loss = torch.mean(F.relu(1 - env_std)) / 2
        
        # Covariance loss
        pred_cov = (flat_preds.T @ flat_preds) / (batch_size * timesteps - 1)
        target_cov = (flat_targets.T @ flat_targets) / (batch_size * timesteps - 1)
        
        cov_loss = (
            extract_nondiagonal(pred_cov).pow(2).sum() / self.embedding_size +
            extract_nondiagonal(target_cov).pow(2).sum() / self.embedding_size
        )
        
        return mse_loss, variance_loss, cov_loss, env_variance_loss
