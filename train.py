import numpy as np
import os
import torch
import torchvision
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm

from dataset import TrajectoryDataset
from models import MainModel  

def setup_compute_device():
    """Determine whether to use GPU or CPU for computation."""
    if torch.cuda.is_available():
        compute_device = torch.device("cuda")
        print(f"Training on GPU: {torch.cuda.get_device_name(0)}")
    else:
        compute_device = torch.device("cpu")
        print("No GPU available, training on CPU")
    return compute_device


def checkpoint_model(model_state, current_epoch, output_dir="model_checkpoints", model_prefix="jepa_model"):
    """Store model checkpoint to disk."""
    # Create directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filename with epoch number
    checkpoint_path = os.path.join(output_dir, f"{model_prefix}_epoch_{current_epoch}.pth")
    
    # Save model state dictionary
    torch.save(model_state, checkpoint_path)
    print(f"Checkpoint saved: {checkpoint_path}")


def execute_training_loop(
    world_model, 
    data_iterator, 
    training_epochs=50, 
    lr=1e-4, 
    compute_device=None
):
    """Execute the main training procedure for the JEPA world model."""
    # Initialize optimizer
    optimizer = optim.Adam(world_model.parameters(), lr=lr)
    
    # Loss coefficients for VICReg components
    invariance_coef = 25.0  # MSE loss coefficient
    variance_coef = 25.0    # Standard deviation loss coefficient
    covariance_coef = 1.0   # Covariance loss coefficient
    
    # Training loop
    for epoch in tqdm(range(training_epochs), desc='Training Progress'):
        # Set model to training mode
        world_model.train()
        
        # Initialize loss tracking variables
        total_epoch_loss = 0.0
        invariance_loss_sum = 0.0
        variance_loss_sum = 0.0
        covariance_loss_sum = 0.0
        env_variance_loss_sum = 0.0
        
        # Process batches
        for observations, actions in data_iterator:
            # Move data to device
            observations = observations.to(compute_device)
            actions = actions.to(compute_device)
            
            # Forward pass
            env_context, state_predictions, target_states = world_model(observations, actions)
            
            # Calculate loss components
            invariance_loss, variance_loss, covariance_loss, env_variance_loss = world_model.loss(
                state_predictions, target_states, env_context
            )
            
            # Compute weighted total loss
            combined_loss = (
                invariance_loss * invariance_coef + 
                variance_loss * variance_coef + 
                covariance_loss * covariance_coef +
                env_variance_loss * variance_coef
            )
            
            # Backpropagation
            optimizer.zero_grad()
            combined_loss.backward()
            optimizer.step()
            
            # Accumulate losses for logging
            total_epoch_loss += combined_loss.item()
            invariance_loss_sum += invariance_loss.item()
            variance_loss_sum += variance_loss.item()
            covariance_loss_sum += covariance_loss.item()
            env_variance_loss_sum += env_variance_loss.item()
        
        # Save checkpoint at each epoch
        checkpoint_model(world_model.state_dict(), epoch)
        
        # Calculate average losses
        batch_count = len(data_iterator)
        avg_invariance_loss = invariance_loss_sum / batch_count
        avg_variance_loss = variance_loss_sum / batch_count
        avg_covariance_loss = covariance_loss_sum / batch_count
        avg_env_variance_loss = env_variance_loss_sum / batch_count
        avg_total_loss = total_epoch_loss / batch_count
        
        # Log training progress
        print(f"Epoch {epoch+1}/{training_epochs} Summary:")
        print(f"  Invariance Loss:     {avg_invariance_loss:.10f}")
        print(f"  Variance Loss:       {avg_variance_loss:.10f}")
        print(f"  Covariance Loss:     {avg_covariance_loss:.10f}")
        print(f"  Env Variance Loss:   {avg_env_variance_loss:.10f}")
        print(f"  Combined Total Loss: {avg_total_loss:.10f}")
        print("-" * 50)
    
    return state_predictions, target_states


if __name__ == "__main__":
    print("Initializing main model training pipeline")
    
    # Setup computing device
    compute_device = setup_compute_device()
    
    # Configure dataset
    print("Loading trajectory dataset...")
    trajectory_dataset = TrajectoryDataset(
        data_dir="/scratch/DL25SP/train",
        states_filename="states.npy",
        actions_filename="actions.npy",
        s_transform=None,
        a_transform=None
    )
    
    # Create data loader
    batch_size = 64
    trajectory_loader = DataLoader(
        trajectory_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    print(f"Dataset loaded with {len(trajectory_dataset)} trajectories")
    
    # Initialize model
    print("Building world model...")
    world_model = MainModel(is_training=True)
    world_model.to(compute_device)
    
    # Configure training parameters
    training_epochs = 30
    learning_rate = 1e-4
    
    # Execute training loop
    print("Starting training procedure...")
    final_predictions, final_targets = execute_training_loop(
        world_model,
        trajectory_loader,
        training_epochs,
        learning_rate,
        compute_device
    )
    
    # Save final model version
    print("Training complete, saving final model")
    checkpoint_model(world_model.state_dict(), "final", output_dir="final_model")
    
    print("Training pipeline completed successfully")
