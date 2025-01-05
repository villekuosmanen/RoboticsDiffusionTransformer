import json
import random
import yaml

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
# from lerobot_data.tfds_builder import LeRobotV2DatasetTFDSBuilder

# Read the config
with open('configs/base.yaml', 'r') as file:
    config = yaml.safe_load(file)
# Load some constants from the config
IMG_HISTORY_SIZE = config['common']['img_history_size']
if IMG_HISTORY_SIZE < 1:
    raise ValueError("Config `img_history_size` must be at least 1.")
ACTION_CHUNK_SIZE = config['common']['action_chunk_size']
if ACTION_CHUNK_SIZE < 1:
    raise ValueError("Config `action_chunk_size` must be at least 1.")
EPSD_LEN_THRESH_LOW = config['dataset']['epsd_len_thresh_low']
EPSD_LEN_THRESH_HIGH = config['dataset']['epsd_len_thresh_high']
# print(f"IMG_HISTORY_SIZE: {IMG_HISTORY_SIZE}")


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
        # Prevent TF from using GPU - the producer will load data in RAM only
        tf.config.set_visible_devices([], 'GPU')
        self.max_steps = EPSD_LEN_THRESH_HIGH

        # Load dataset names and weights from config
        dataset_names_cfg = f'configs/{dataset_type}_datasets.json'
        sample_weights_cfg = f'configs/{dataset_type}_sample_weights.json'
        
        with open(dataset_names_cfg, 'r') as f:
            self.dataset_names = json.load(f)
        with open(sample_weights_cfg, 'r') as f:
            sample_weights = json.load(f)
            
        # Set random seed
        np.random.seed(seed)
        
        # Initialize datasets and counters
        self.datasets = {}
        self.episode_counters = {}
        self.repeat = repeat
        self.sample_weights = []

        for dataset_name in self.dataset_names:
            self.datasets[dataset_name] = LeRobotDataset(dataset_name)
            self.episode_counters[dataset_name] = 0
            self.sample_weights.append(sample_weights[dataset_name])

        # Normalize weights
        self.sample_weights = np.array(self.sample_weights, dtype=np.float32)
        self.sample_weights /= np.sum(self.sample_weights)

        # # Initialize datasets
        # self.name2dataset = {}
        # self.sample_weights = []
        
        # for dataset_name in self.dataset_names:
        #     # Load dataset and preprocess all episodes
        #     lerobot_dataset = LeRobotDataset(dataset_name)

        #     # collect episodes by feature
        #     collected_features = {}
        #     first_episode = self._collect_episode(lerobot_dataset, 0)
        #     first_processed = self._preprocess_episode(first_episode, dataset_name)
        #     for key in first_processed.keys():
        #         collected_features[key] = []
            
        #     # Pre-process all episodes and convert to the format we need
        #     for episode_idx in range(lerobot_dataset.num_episodes):
        #         episode = self._collect_episode(lerobot_dataset, episode_idx)
                
        #         processed = self._preprocess_episode(episode, dataset_name)
        #         for key in processed.keys():
        #             collected_features[key].append(processed[key])

        #     print("processed_episodes collected")
            
        #     # Convert to tensor format
        #     # Each processed episode should now be a dict of tensors
        #     dataset = tf.data.Dataset.from_tensor_slices(collected_features)
        #     print("dataset created")

        #     if repeat:
        #         dataset = dataset.repeat()
                
        #     self.name2dataset[dataset_name] = iter(dataset)
        #     self.sample_weights.append(sample_weights[dataset_name])
            
        # # Normalize weights
        # self.sample_weights = np.array(self.sample_weights, dtype=np.float32)
        # self.sample_weights /= np.sum(self.sample_weights)

    def sample_episode_frames(self, from_idx, to_idx):
        episode_length = to_idx - from_idx
        
        if episode_length <= self.max_steps:
            # If episode is shorter than max_steps, return full episode
            return (from_idx, to_idx)
        
        # First determine if a centered window is possible at the chosen point
        def get_valid_window(center_idx):
            half_window = self.max_steps // 2
            start = center_idx - half_window
            end = start + self.max_steps  # Use full window length to handle odd max_steps
            
            # If window would go beyond bounds, adjust it
            if start < from_idx:
                # If too close to start, anchor at start
                return (from_idx, from_idx + self.max_steps)
            elif end > to_idx:
                # If too close to end, anchor at end
                return (to_idx - self.max_steps, to_idx)
            else:
                # Center window is valid
                return (start, end)
        
        # Pick a random center point across full range
        center_idx = random.randint(from_idx, to_idx)
        window = get_valid_window(center_idx)
        
        return window

    def get_episode(self, dataset_name):
        """Get next episode from a dataset."""
        dataset = self.datasets[dataset_name]
        counter = self.episode_counters[dataset_name]
        
        # Reset counter if needed
        if counter >= dataset.num_episodes:
            if not self.repeat:
                raise StopIteration
            counter = 0
            
        # Get episode boundaries
        from_idx = dataset.episode_data_index["from"][counter].item()
        to_idx = dataset.episode_data_index["to"][counter].item()
        if to_idx - from_idx < EPSD_LEN_THRESH_LOW:
            # return nothing if too small of an episode
            self.episode_counters[dataset_name] = counter + 1
            return None
        (from_idx, to_idx) = self.sample_episode_frames(from_idx, to_idx)
        
        # Get episode frames from parquet
        print(f"from_idx: {from_idx}, to_idx: {to_idx}")
        frames = dataset.hf_dataset[from_idx:to_idx]
        # print(dataset.features)
        # print(dataset.hf_features)

        # Get video data for each camera
        camera_keys = [k for k in dataset.features.keys() if k.startswith('observation.images.')]
        timestamps = [frame.item() for frame in frames["timestamp"]]     
        query_timestamps = {
            camera_key: timestamps for camera_key in camera_keys
        }
        video_frames = dataset._query_videos(query_timestamps, counter)
        
        # Create episode data combining parquet and video data
        episode_data = {}
        for key in frames.keys():
            if key not in camera_keys:  # Skip camera keys as we handle them separately
                stacked = torch.stack(frames[key])
                episode_data[key] = tf.convert_to_tensor(stacked.numpy(), dtype=tf.float32)
        for camera_key in camera_keys:
            episode_data[camera_key] = tf.convert_to_tensor(video_frames[camera_key].numpy(), dtype=tf.float32)
        episode_data['language_instruction'] = dataset.meta.episodes[counter]['tasks'][0]
        
        # Update counter
        self.episode_counters[dataset_name] = counter + 1
        
        return episode_data

    # def _collect_episode(self, lerobot_dataset, episode_idx):
    #     """Convert a full episode from LeRobotDataset to tensor format."""
    #     # Get episode boundaries
    #     from_idx = lerobot_dataset.episode_data_index["from"][episode_idx].item()
    #     to_idx = lerobot_dataset.episode_data_index["to"][episode_idx].item()
        
    #     # Get all frames for this episode
    #     print(f"from_idx: {from_idx}, to_idx: {to_idx}")
    #     frames = lerobot_dataset.hf_dataset[from_idx:to_idx]
        
    #     # Convert to tensors - frames is now a datasets.Dataset object
    #     # which already contains all our data in a batch
    #     episode_data = {}

    #     for key in frames.keys():
    #         # Stack PyTorch tensors then convert to TF
    #         stacked = torch.stack(frames[key])
    #         episode_data[key] = tf.convert_to_tensor(stacked.numpy(), dtype=tf.float32) 
    #         episode_data['language_instruction'] = lerobot_dataset.meta.episodes[episode_idx]['tasks'][0]
    #     return episode_data

    def _preprocess_episode(self, episode, dataset_name):
        """Convert raw episode to tensor format with all necessary preprocessing."""

        # Get all states and actions from episode
        states = tf.convert_to_tensor(episode['observation.state'].numpy(), dtype=tf.float32)
        actions = tf.convert_to_tensor(episode['action'].numpy(), dtype=tf.float32)

        # Handle state history (past states window)
        first_state = tf.expand_dims(states[0], axis=0)
        first_state = tf.repeat(first_state, ACTION_CHUNK_SIZE-1, axis=0)
        padded_states = tf.concat([first_state, states], axis=0)
        state_indices = tf.range(ACTION_CHUNK_SIZE, tf.shape(states)[0] + ACTION_CHUNK_SIZE)
        past_states = tf.map_fn(
            lambda i: padded_states[i - ACTION_CHUNK_SIZE:i],
            state_indices,
            dtype=tf.float32  # Explicitly specify TF dtype
        )
        
        # Handle future actions window
        last_action = tf.expand_dims(actions[-1], axis=0)
        last_action = tf.repeat(last_action, ACTION_CHUNK_SIZE, axis=0)
        padded_actions = tf.concat([actions, last_action], axis=0)
        action_indices = tf.range(0, tf.shape(actions)[0])
        future_actions = tf.map_fn(
            lambda i: padded_actions[i:i + ACTION_CHUNK_SIZE],
            action_indices,
            dtype=tf.float32
        )
        
        # Calculate stats
        state_std = tf.math.reduce_std(states, axis=0, keepdims=True)
        state_mean = tf.math.reduce_mean(states, axis=0, keepdims=True)
        state_norm = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(states), axis=0, keepdims=True))
        
        # Repeat stats to match sequence length
        state_std = tf.repeat(state_std, tf.shape(states)[0], axis=0)
        state_mean = tf.repeat(state_mean, tf.shape(states)[0], axis=0)
        state_norm = tf.repeat(state_norm, tf.shape(states)[0], axis=0)
        
        # Handle camera frames
        camera_frames = {}
        camera_masks = {}
        camera_keys = [k for k in episode.keys() if k.startswith('observation.images.')]

        for idx, camera_key in enumerate(camera_keys):
            frames = tf.convert_to_tensor(episode[camera_key].numpy(), dtype=tf.float32)
            
            # Create frame history window
            first_frame = tf.expand_dims(frames[0], axis=0)
            first_frame = tf.repeat(first_frame, IMG_HISTORY_SIZE-1, axis=0)
            padded_frames = tf.concat([first_frame, frames], axis=0)
            frame_indices = tf.range(IMG_HISTORY_SIZE, tf.shape(frames)[0] + IMG_HISTORY_SIZE)
            
            past_frames = tf.map_fn(
                lambda i: padded_frames[i - IMG_HISTORY_SIZE:i],
                frame_indices,
                dtype=tf.float32
            )
            
            # Create time masks for frames
            frames_time_mask = tf.ones([tf.shape(frames)[0]], dtype=tf.bool)
            padded_time_mask = tf.pad(
                frames_time_mask, 
                [[IMG_HISTORY_SIZE-1, 0]], 
                "CONSTANT", 
                constant_values=False
            )
            past_frames_time_mask = tf.map_fn(
                lambda i: padded_time_mask[i - IMG_HISTORY_SIZE:i],
                frame_indices,
                dtype=tf.bool
            )
            
            camera_frames[f'past_frames_{idx}'] = past_frames
            camera_masks[f'past_frames_{idx}_time_mask'] = past_frames_time_mask

        num_steps = tf.shape(states)[0]
        print(f"num_steps: {num_steps}")

        # Get the dataset name and instruction
        instruction = episode.get('language_instruction', None)
        if instruction is not None:
            if isinstance(instruction, torch.Tensor):
                instruction = instruction.numpy()
        else:
            instruction = ''

        # Convert these to tensors that can be sliced
        steps_data = {
            'step_id': tf.range(num_steps),
            'dataset_name': tf.constant(dataset_name, dtype=tf.string),
            'language_instruction': tf.constant(instruction, dtype=tf.string),
            'state_chunk': past_states,
            'action_chunk': future_actions,
            'state_vec_mask': tf.ones_like(states, dtype=tf.bool),
            'state_std': state_std,
            'state_mean': state_mean,
            'state_norm': state_norm
        }
        
        # Add camera frames and masks
        for cam_key in camera_frames:
            steps_data[cam_key] = camera_frames[cam_key]
            mask_key = f"{cam_key}_time_mask"
            steps_data[mask_key] = camera_masks[mask_key]
        
        return steps_data
    
    def __iter__(self):
        """Iterate over episodes."""
        while True:
            dataset_name = np.random.choice(
                self.dataset_names, 
                p=self.sample_weights
            )
            
            episode_frames = self.get_episode(dataset_name)
            if episode_frames == None:
                continue
            
            # Process frames into required format
            processed = self._preprocess_episode(episode_frames, dataset_name)
            processed['json_content'] = {
                'dataset_name': processed['dataset_name'],
                'instruction': processed['language_instruction'],
            }
            del(episode_frames)

            yield processed            

if __name__ == "__main__":
    dataset = LeRobotV2Dataset(0, 'finetune')
    i = 0
    for episode in dataset:
        print("step in an episode")
        if i == 0:
            print("\nShape information:")
            # Print shapes of key tensors if available
            for key, value in episode.items():
                if hasattr(value, 'shape'):
                    print(f"{key}: {value.shape}")
        del(episode)
        i += 1
        if i == 10:
            break
