import json

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from lerobot_data.tfds_builder import LeRobotV2DatasetTFDSBuilder

class LeRobotV2Dataset:
    """
    Dataset class for sampling episodes from LeRobot datasets.
    """
    
    def __init__(self, seed: int, dataset_type: str, repeat: bool = True):
        """Initialize the dataset.
        
        Args:
            seed: Random seed for reproducibility
            dataset_type: Either 'pretrain' or 'finetune'
            repeat: Whether to repeat the dataset infinitely
        """
        # Load dataset names and weights from config
        dataset_names_cfg = f'configs/{dataset_type}_datasets.json'
        sample_weights_cfg = f'configs/{dataset_type}_sample_weights.json'
        
        with open(dataset_names_cfg, 'r') as f:
            self.dataset_names = json.load(f)
        with open(sample_weights_cfg, 'r') as f:
            sample_weights = json.load(f)
            
        # Set random seed
        tf.random.set_seed(seed)
        np.random.seed(seed)
        
        # Initialize datasets
        self.name2dataset = {}
        self.sample_weights = []
        
        for dataset_name in self.dataset_names:
            # Create, initialize, and prepare the TFDS builder
            builder = LeRobotV2DatasetTFDSBuilder(dataset_name)
            print("___________")
            print(f"builder: {builder}")
            print("___________")

            builder.download_and_prepare()
            dataset = builder.as_dataset(split='train', shuffle_files=True)

            print("___________")
            print(f"dataset: {dataset}")
            print("___________")

            if repeat:
                dataset = dataset.repeat()
                
            self.name2dataset[dataset_name] = iter(dataset)
            self.sample_weights.append(sample_weights[dataset_name])
            
        # Normalize weights
        self.sample_weights = np.array(self.sample_weights)
        self.sample_weights /= np.sum(self.sample_weights)
    
    def __iter__(self):
        """Iterate over episodes."""
        while True:
            # Sample dataset according to weights
            dataset_name = np.random.choice(
                self.dataset_names, 
                p=self.sample_weights
            )
            
            # Get next episode from the selected dataset
            episode = next(self.name2dataset[dataset_name])
            
            # Convert episode to steps (this needs to be implemented
            # based on actual data format)
            episode_steps = self._episode_to_steps(episode)
            
            yield episode_steps
            
    def _episode_to_steps(self, episode):
        """Convert an episode to a list of steps.
        
        This needs to be implemented based on the actual data format
        of LeRobot episodes.
        """
        # Placeholder - implement based on actual format
        return episode['steps']

if __name__ == "__main__":
    dataset = LeRobotV2Dataset(0, 'finetune')
    try:
        for episode in dataset:
            print("First step of episode:")
            print(episode[0])
            print("\nShape information:")
            # Print shapes of key tensors if available
            for key, value in episode[0].items():
                if hasattr(value, 'shape'):
                    print(f"{key}: {value.shape}")
            break
    except Exception as e:
        print(f"Error during dataset testing: {e}")
