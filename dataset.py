from typing import NamedTuple, Optional
import torch
import numpy as np
from torch.utils.data import Dataset 


class AgentTrajectoryData(Dataset):
    """
    Dataset for loading and processing agent trajectories.
    
    Parameters:
        data_dir: Root directory containing the data files
        states_filename: Filename for the states array
        actions_filename: Filename for the actions array
        s_transform: Optional transformation applied to state data
        a_transform: Optional transformation applied to action data
        length: Optional parameter to limit dataset size
        
    Data format:
        - States: numpy array with shape (num_trajectories, sequence_length, 2, 65, 65)
        - Actions: numpy array with shape (num_trajectories, sequence_length, 2)
    """
    def __init__(self, 
                 data_dir, 
                 states_filename, 
                 actions_filename, 
                 s_transform=None, 
                 a_transform=None,
                 length=None):
        # Load data arrays from files
        state_path = f"{data_dir}/{states_filename}"
        action_path = f"{data_dir}/{actions_filename}"
        
        # Use memory mapping for efficient loading of large arrays
        self.observation_data = np.load(state_path, mmap_mode="r")
        self.motion_data = np.load(action_path)
        
        # Limit dataset size if specified
        if length is not None:
            self.observation_data = self.observation_data[:length]
            self.motion_data = self.motion_data[:length]
        
        # Store transform functions
        self.obs_transform = s_transform
        self.motion_transform = a_transform
    
    def __len__(self):
        """Return the number of trajectories in the dataset."""
        return self.observation_data.shape[0]
    
    def __getitem__(self, idx):
        """Retrieve a single trajectory pair (observations, actions)."""
        # Get the raw data for the requested index
        obs = self.observation_data[idx]
        motion = self.motion_data[idx]
        
        # Apply transformations if specified
        if self.obs_transform:
            for frame_idx in range(obs.shape[0]):
                obs[frame_idx] = self.obs_transform(obs[frame_idx])
        
        if self.motion_transform:
            for action_idx in range(motion.shape[0]):
                motion[action_idx] = self.motion_transform(motion[action_idx])
        
        return obs, motion


class WallSample(NamedTuple):
    states: torch.Tensor
    locations: torch.Tensor
    actions: torch.Tensor


class WallDataset:
    def __init__(
        self,
        data_path,
        probing=False,
        device="cuda",
    ):
        self.device = device
        self.states = np.load(f"{data_path}/states.npy", mmap_mode="r")
        self.actions = np.load(f"{data_path}/actions.npy")

        if probing:
            self.locations = np.load(f"{data_path}/locations.npy")
        else:
            self.locations = None

    def __len__(self):
        return len(self.states)

    def __getitem__(self, i):
        states = torch.from_numpy(self.states[i]).float().to(self.device)
        actions = torch.from_numpy(self.actions[i]).float().to(self.device)

        if self.locations is not None:
            locations = torch.from_numpy(self.locations[i]).float().to(self.device)
        else:
            locations = torch.empty(0).to(self.device)

        return WallSample(states=states, locations=locations, actions=actions)


def create_wall_dataloader(
    data_path,
    probing=False,
    device="cuda",
    batch_size=64,
    train=True,
):
    ds = WallDataset(
        data_path=data_path,
        probing=probing,
        device=device,
    )

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size,
        shuffle=train,
        drop_last=True,
        pin_memory=False,
    )

    return loader
