import json
import random
import yaml

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch

from state_vec import STATE_VEC_IDX_MAPPING, STATE_VEC_LEN
OFFLOAD_DIR = "data/lerobot/lang_embeddings/"
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

# Read the image keys of each dataset
with open('configs/dataset_img_keys.json', 'r') as file:
    IMAGE_KEYS = json.load(file)

AGILEX_STATE_INDICES = [
    STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING["left_gripper_open"]
]

AGILEX_STATE_INDICES_BIMANUAL = [
    STATE_VEC_IDX_MAPPING[f"left_arm_joint_{i}_pos"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING["left_gripper_open"]
] + [
    STATE_VEC_IDX_MAPPING[f"right_arm_joint_{i}_pos"] for i in range(6)
] + [
    STATE_VEC_IDX_MAPPING[f"right_gripper_open"]
]

def _format_joint_to_state(joints):
        """
        Format the joint proprioception into the unified action vector.

        Args:
            joints (torch.Tensor): The joint proprioception to be formatted. 
                qpos ([B, N, 14]).

        Returns:
            state (torch.Tensor): The formatted vector for RDT ([B, N, 128]). 
        """
        # Rescale the gripper to the range of [0, 1]
        if len(joints[0]) == 7:
            joints = joints * torch.tensor(
                [[1, 1, 1, 1, 1, 1, 20]],
                device=joints.device, dtype=joints.dtype
            )
            state_indices = AGILEX_STATE_INDICES
        elif len(joints[0]) == 14:
            joints = joints * torch.tensor(
                [[1, 1, 1, 1, 1, 1, 20, 1, 1, 1, 1, 1, 1, 20]],
                device=joints.device, dtype=joints.dtype
            )
            state_indices = AGILEX_STATE_INDICES_BIMANUAL
        else:
            raise ValueError(f"does not support datasets with joints of size {len(joints[0][0])}")
        
        B, _ = joints.shape
        state = torch.zeros(
            (B, STATE_VEC_LEN), 
            device=joints.device, dtype=joints.dtype
        )
        # Fill into the unified state vector
        state[:, state_indices] = joints
        # Assemble the mask indicating each dimension's availability 
        state_elem_mask = torch.zeros(
            (B, STATE_VEC_LEN),
            device=joints.device, dtype=joints.dtype
        )
        state_elem_mask[:, state_indices] = 1
        return state, state_elem_mask

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

        # Print dataset stats
        total_frames = 0
        fps = 50
        for dataset in self.datasets.values():
            total_frames += dataset.meta.total_frames
        print(f"total frames: {total_frames}")
        hours_of_data = total_frames / fps
        import time
        print(f"Amount of robot data included: {time.strftime('%H:%M:%S', time.gmtime(hours_of_data))}")


        # Normalize weights
        self.sample_weights = np.array(self.sample_weights, dtype=np.float32)
        self.sample_weights /= np.sum(self.sample_weights)

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
                episode_data[key] = stacked
        for camera_key in camera_keys:
            episode_data[camera_key] = video_frames[camera_key]
        episode_data['language_instruction'] = dataset.meta.episodes[counter]['tasks'][0]
        
        # Update counter
        self.episode_counters[dataset_name] = counter + 1
        
        return episode_data
    
    def get_preprocessed_states(self, dataset_name):
        dataset = self.datasets[dataset_name]
        preprocessed_states = []
        for i in range(dataset.num_episodes):
            from_idx = dataset.episode_data_index["from"][i].item()
            to_idx = dataset.episode_data_index["to"][i].item()
            
            episode_data = {}
            try:
                frames = dataset.hf_dataset[from_idx:to_idx]
            except IndexError:
                # in case of incorrectly cleaned data
                continue
            for key in frames.keys():
                stacked = torch.stack(frames[key])
                episode_data[key] = stacked
            
            # pre-process
            states_raw, _ = _format_joint_to_state(episode_data['observation.state'])
            preprocessed_states.append(tf.convert_to_tensor(states_raw.numpy(), dtype=tf.float32))

        return preprocessed_states

    def preprocess_episode(self, episode, dataset_name):
        """Convert raw episode to tensor format with all necessary preprocessing."""
        states_raw, state_masks_raw = _format_joint_to_state(episode['observation.state'])
        actions_raw, actions_mask = _format_joint_to_state(episode['action'])

        # Get all states and actions from episode
        states = tf.convert_to_tensor(states_raw.numpy(), dtype=tf.float32)
        state_masks = tf.convert_to_tensor(state_masks_raw.numpy(), dtype=tf.float32)
        actions = tf.convert_to_tensor(actions_raw.numpy(), dtype=tf.float32)

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

        image_meta_for_dataset = IMAGE_KEYS[dataset_name]
        for idx in range(4):
            image_key = image_meta_for_dataset['image_keys'][idx]
            image_mask = image_meta_for_dataset['image_mask'][idx]
            if image_mask == 1:
                frames = tf.convert_to_tensor(episode[image_key].numpy(), dtype=tf.float32)
            else:
                frames = tf.TensorArray(dtype=tf.float32, size=tf.shape(states)[0] - 1, dynamic_size=True)
                frames = frames.write(
                    tf.shape(states)[0] - 1,
                    tf.zeros([0, 0, 0], dtype=tf.float32),
                ).stack()
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

        # Get the dataset name and instruction
        instruction = episode.get('language_instruction', None)
        if instruction is not None:
            if isinstance(instruction, torch.Tensor):
                instruction = instruction.numpy()
        else:
            instruction = ''

        steps = []
        for i in range(num_steps):
            step_dict = {
                'step_id': tf.constant(i, dtype=tf.uint8),
                'state_chunk': past_states[i],
                'action_chunk': future_actions[i],
                'state_vec_mask': state_masks[i],
                'state_std': state_std[i],
                'state_mean': state_mean[i],
                'state_norm': state_norm[i],
                'json_content': {
                    'dataset_name': dataset_name,
                    'instruction': instruction,
                },
            }
            
            # Add camera frames and masks for this timestep
            for cam_key in camera_frames:
                step_dict[cam_key] = camera_frames[cam_key][i]
                mask_key = f"{cam_key}_time_mask"
                step_dict[mask_key] = camera_masks[mask_key][i]
            
            steps.append(step_dict)

        return steps
    
    def __iter__(self):
        """Iterate over episodes."""
        while True:
            dataset_name = np.random.choice(
                self.dataset_names, 
                p=self.sample_weights
            )
            
            episode = self.get_episode(dataset_name)
            if episode == None:
                continue

            yield self.preprocess_episode(episode, dataset_name)            

if __name__ == "__main__":
    dataset = LeRobotV2Dataset(0, 'finetune')
    i = 0
    for episode in dataset:
        print("step in an episode")
        if i == 0:
            print(f"Shape information (len={len(episode)}):")
            # Print shapes of key tensors if available
            for key, value in episode[0].items():
                if hasattr(value, 'shape'):
                    print(f"{key}: {value.shape}")
                else:
                    print(value)
        del(episode)
        i += 1
        if i == 10:
            break
