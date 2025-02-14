import json
import random
import yaml

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import torch

from state_vec import AGILEX_STATE_INDICES, AGILEX_STATE_INDICES_BIMANUAL, STATE_VEC_LEN
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

def _format_joint_to_state(joints, is_agilex: bool):
        """
        Format the joint proprioception into the unified action vector.

        Args:
            joints (torch.Tensor): The joint proprioception to be formatted. 
                qpos ([B, N, 14]).

        Returns:
            state (torch.Tensor): The formatted vector for RDT ([B, N, 128]). 
        """
        # in AgileX datasets, gripper size is measured in centimeters, while the default for default ARX5 SDK is in meters.
        gripper_multiplier = 0.20 if is_agilex else 20

        # Rescale the gripper to the range of [0, 1]
        if len(joints[0]) == 7:
            joints = joints * torch.tensor(
                [[1, 1, 1, 1, 1, 1, gripper_multiplier]],
                device=joints.device, dtype=joints.dtype
            )
            state_indices = AGILEX_STATE_INDICES
        elif len(joints[0]) == 14:
            joints = joints * torch.tensor(
                [[1, 1, 1, 1, 1, 1, gripper_multiplier, 1, 1, 1, 1, 1, 1, gripper_multiplier]],
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

def get_past_states(states, episode_start, episode_end):
    state_indices = tf.range(episode_start, episode_end)
    
    def get_history_slice(i):
        # Get available history
        history = states[max(0, i - ACTION_CHUNK_SIZE):i]
        # Get the actual size we got
        actual_size = tf.shape(history)[0]
        
        # If we need padding
        if actual_size < ACTION_CHUNK_SIZE:
            # Get the first state (either from history or current position if history is empty)
            first_state = tf.cond(
                tf.greater(actual_size, 0),
                lambda: history[0],
                lambda: states[i]
            )
            first_state = tf.expand_dims(first_state, 0)
            
            # Create padding
            padding_size = ACTION_CHUNK_SIZE - actual_size
            padding = tf.repeat(first_state, padding_size, axis=0)
            
            # Combine padding with history
            return tf.concat([padding, history], axis=0)
        
        return history
    
    past_states = tf.map_fn(
        get_history_slice,
        state_indices,
        dtype=tf.float32
    )
    
    return past_states

def get_future_actions(actions, episode_start, episode_end):
    action_indices = tf.range(episode_start, episode_end)
    
    def get_future_slice(i):
        # Get available future actions
        future = actions[i:i + ACTION_CHUNK_SIZE]
        # Get the actual size we got
        actual_size = tf.shape(future)[0]
        
        # If we need padding
        if actual_size < ACTION_CHUNK_SIZE:
            # Get the last action (either from future or current position if future is empty)
            last_action = tf.cond(
                tf.greater(actual_size, 0),
                lambda: future[-1],
                lambda: actions[i]
            )
            last_action = tf.expand_dims(last_action, 0)
            
            # Create padding
            padding_size = ACTION_CHUNK_SIZE - actual_size
            padding = tf.repeat(last_action, padding_size, axis=0)
            
            # Combine future with padding
            return tf.concat([future, padding], axis=0)
        
        return future
    
    future_actions = tf.map_fn(
        get_future_slice,
        action_indices,
        dtype=tf.float32
    )
    
    return future_actions

def get_past_frames(frames, episode_start, episode_end):
    frame_indices = tf.range(episode_start, episode_end)
    
    def get_history_slice(i):
        # Get available history
        history = frames[max(0, i - IMG_HISTORY_SIZE):i]
        # Get the actual size we got
        actual_size = tf.shape(history)[0]
        
        # If we need padding
        if actual_size < IMG_HISTORY_SIZE:
            # Get the first frame (either from history or current position if history is empty)
            first_frame = tf.cond(
                tf.greater(actual_size, 0),
                lambda: history[0],
                lambda: frames[i]
            )
            first_frame = tf.expand_dims(first_frame, 0)
            
            # Create padding
            padding_size = IMG_HISTORY_SIZE - actual_size
            padding = tf.repeat(first_frame, padding_size, axis=0)
            
            # Combine padding with history
            return tf.concat([padding, history], axis=0)
        
        return history
    
    past_frames = tf.map_fn(
        get_history_slice,
        frame_indices,
        dtype=tf.float32
    )
    
    return past_frames, frame_indices

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
        total_episodes = 0
        fps = 50
        for dataset in self.datasets.values():
            total_frames += dataset.meta.total_frames
            total_episodes += dataset.num_episodes
        print(f"total frames: {total_frames}")
        print(f"total number of trajectories: {total_episodes}")
        hours_of_data = total_frames / fps
        import time
        print(f"Amount of robot data included: {time.strftime('%H:%M:%S', time.gmtime(hours_of_data))}")


        # Normalize weights
        self.sample_weights = np.array(self.sample_weights, dtype=np.float32)
        self.sample_weights /= np.sum(self.sample_weights)

    def sample_episode_frames(self, from_idx, to_idx):
        episode_length = to_idx - from_idx
        MIN_HORIZON_SIZE = 10
        horizon_size = max(MIN_HORIZON_SIZE, ACTION_CHUNK_SIZE)
        
        # First ensure we have enough space for the window plus minimum horizons
        total_required_space = self.max_steps + (2 * horizon_size)
        if episode_length <= total_required_space:
            # If episode is too short to accommodate window + horizons,
            # just return the full episode
            return (from_idx, from_idx, to_idx, to_idx)
        
        # Calculate the valid range for the window start position,
        # ensuring space for horizons on both sides
        valid_start_min = from_idx + horizon_size
        valid_start_max = to_idx - (self.max_steps + horizon_size)
        
        # First determine if a centered window is possible at the chosen point
        def get_valid_window(center_idx):
            half_window = self.max_steps // 2
            start = center_idx - half_window
            end = start + self.max_steps
            
            # If window would go beyond valid bounds, adjust it
            if start < valid_start_min:
                # If too close to start, anchor at valid_start_min
                return (valid_start_min, valid_start_min + self.max_steps)
            elif end > (to_idx - horizon_size):
                # If too close to end, anchor at the last valid position
                return (valid_start_max, to_idx - horizon_size)
            else:
                # Center window is valid
                return (start, end)
        
        # Pick a random center point across valid range
        valid_center_min = valid_start_min + (self.max_steps // 2)
        valid_center_max = to_idx - (self.max_steps // 2) - horizon_size
        center_idx = random.randint(valid_center_min, valid_center_max)
        
        window_start, window_end = get_valid_window(center_idx)
        
        # Now calculate horizons - they'll always be exactly horizon_size
        # since we've ensured the space exists
        horizon_from = window_start - horizon_size
        horizon_to = window_end + horizon_size
        
        return (horizon_from, window_start, window_end, horizon_to)

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
            return None, None, None
        (horizon_from, from_idx, to_idx, horizon_to) = self.sample_episode_frames(from_idx, to_idx)
        
        # Get episode frames from parquet
        print(f"horizon_from: {horizon_from}, from_idx: {from_idx}, to_idx: {to_idx}, horizon_to: {horizon_to}")
        frames = dataset.hf_dataset[horizon_from:horizon_to]
        if (from_idx - horizon_from) < ACTION_CHUNK_SIZE:
            print("restricted horizon for states")
        if (horizon_to - to_idx) < ACTION_CHUNK_SIZE:
            print("restricted horizon for actions")

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
        
        return episode_data, (from_idx - horizon_from), (to_idx - horizon_from)
    
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
            states_raw, _ = _format_joint_to_state(episode_data['observation.state'], "agilex_" in dataset_name)
            preprocessed_states.append(tf.convert_to_tensor(states_raw.numpy(), dtype=tf.float32))

        return preprocessed_states

    def preprocess_episode(self, episode, episode_start, episode_end, dataset_name):
        """Convert raw episode to tensor format with all necessary preprocessing."""
        states_raw, state_masks_raw = _format_joint_to_state(episode['observation.state'], "agilex_" in dataset_name)
        actions_raw, actions_mask = _format_joint_to_state(episode['action'], "agilex_" in dataset_name)

        # Get all states and actions from episode
        states = tf.convert_to_tensor(states_raw.numpy(), dtype=tf.float32)
        state_masks = tf.convert_to_tensor(state_masks_raw.numpy(), dtype=tf.float32)
        actions = tf.convert_to_tensor(actions_raw.numpy(), dtype=tf.float32)

        past_states = get_past_states(states, episode_start, episode_end)
        future_actions = get_future_actions(actions, episode_start, episode_end)

        # Calculate stats only on the episode portion
        episode_states = states[episode_start:episode_end]
        state_std = tf.math.reduce_std(episode_states, axis=0, keepdims=True)
        state_mean = tf.math.reduce_mean(episode_states, axis=0, keepdims=True)
        state_norm = tf.math.sqrt(tf.math.reduce_mean(tf.math.square(episode_states), axis=0, keepdims=True))
        
        # Repeat stats to match episode length (not full sequence length)
        episode_length = episode_end - episode_start
        state_std = tf.repeat(state_std, episode_length, axis=0)
        state_mean = tf.repeat(state_mean, episode_length, axis=0)
        state_norm = tf.repeat(state_norm, episode_length, axis=0)
        
        # Handle camera frames
        camera_frames = {}
        camera_masks = {}

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

            # Get frame history for each timestep in the episode
            past_frames, frame_indices = get_past_frames(frames, episode_start, episode_end)
            
            # Create masks indicating which frames are real vs padded
            def create_chunk_mask(i):
                # Calculate how many real frames we have for this chunk
                real_frames = tf.minimum(i - max(0, i - IMG_HISTORY_SIZE), IMG_HISTORY_SIZE)
                # Create mask with False for padded frames, True for real frames
                mask = tf.concat([
                    tf.zeros([IMG_HISTORY_SIZE - real_frames], dtype=tf.bool),
                    tf.ones([real_frames], dtype=tf.bool)
                ], axis=0)
                return mask

            past_frames_time_mask = tf.map_fn(
                create_chunk_mask,
                frame_indices,
                dtype=tf.bool
            )
            camera_frames[f'past_frames_{idx}'] = past_frames
            camera_masks[f'past_frames_{idx}_time_mask'] = past_frames_time_mask

        # Get the dataset name and instruction
        instruction = episode.get('language_instruction', None)
        if instruction is not None:
            if isinstance(instruction, torch.Tensor):
                instruction = instruction.numpy()
        else:
            instruction = ''

        steps = []
        # Iterate only over the episode portion
        for step_idx in range(episode_length):
            abs_idx = step_idx + episode_start  # Convert to absolute index for accessing full data
            
            step_dict = {
                'step_id': tf.constant(step_idx, dtype=tf.uint8),  # Use relative index for step_id
                'state_chunk': past_states[step_idx],              # These are already episode-only
                'action_chunk': future_actions[step_idx],          # These are already episode-only
                'state_vec_mask': state_masks[abs_idx],           # Use absolute index for full data
                'state_std': state_std[step_idx],                 # These are episode-only
                'state_mean': state_mean[step_idx],
                'state_norm': state_norm[step_idx],
                'json_content': {
                    'dataset_name': dataset_name,
                    'instruction': instruction,
                },
            }
            
            # Add camera frames and masks for this timestep
            for cam_key in camera_frames:
                step_dict[cam_key] = camera_frames[cam_key][step_idx]       # These are episode-only
                mask_key = f"{cam_key}_time_mask"
                step_dict[mask_key] = camera_masks[mask_key][step_idx]      # These are episode-only
            
            steps.append(step_dict)

        return steps

    def __iter__(self):
        """Iterate over episodes."""
        while True:
            dataset_name = np.random.choice(
                self.dataset_names, 
                p=self.sample_weights
            )
            
            episode, episode_start, episode_end = self.get_episode(dataset_name)
            if episode == None:
                continue

            yield self.preprocess_episode(episode, episode_start, episode_end, dataset_name)            

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
