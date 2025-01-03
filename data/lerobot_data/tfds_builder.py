from typing import List

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import tensorflow_datasets as tfds
import tensorflow as tf

DTYPE_MAP = {
    'float32': tf.float32,
    'int64': tf.int64,
    'bool': tf.bool,
    'string': tf.string
}

class LeRobotV2DatasetTFDSBuilder(tfds.core.GeneratorBasedBuilder):
    """
    TFDS Builder for LeRobot V2 datasets.
    """
    VERSION = tfds.core.Version('1.0.0')
    _dataset = None
    _metadata = None

    def __init__(self, repo_id: str, **kwargs):
        self.repo_id = repo_id
        super().__init__(**kwargs)

    def _info(self):
        """Define the dataset structure."""
        
        print("_info")
        # Lazy load metadata if not already loaded
        if not self._metadata:
            dataset = self._load_dataset()
            self._metadata = dataset.meta.info

        # Convert LeRobot features to TFDS features
        tfds_features = {}
        for key, feature in self._metadata['features'].items():
            if feature['dtype'] == 'video':
                # For video features, we'll get frames as image tensors
                tfds_features[key] = tfds.features.Image(
                    shape=feature['shape'],
                    encoding_format='png'
                )
            else:
                # For other features, map to appropriate TFDS feature type
                dtype_str = feature['dtype']
                if dtype_str == 'string':
                    tfds_features[key] = tfds.features.Text()
                else:
                    tf_dtype = DTYPE_MAP.get(dtype_str)
                    if tf_dtype is None:
                        raise ValueError(f"Unsupported dtype: {dtype_str}")
                    tfds_features[key] = tfds.features.Tensor(
                        shape=feature['shape'],
                        dtype=tf_dtype
                    )

        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.features.FeaturesDict(tfds_features),
            supervised_keys=None,
            homepage=f"https://huggingface.co/datasets/{self.repo_id}"
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        raise ValueError("lol")

        self._load_dataset()
        print("_split_generators")
        # Create split generators
        return {
            'train': self._generate_examples()  # Just use all data as train split
        }
    
    def _load_dataset(self):
        """Lazy load the LeRobot dataset."""
        print("_load_dataset")

        if self._dataset is None:
            self._dataset = LeRobotDataset(self.repo_id)
        return self._dataset
    
    def _generate_examples(self):
        """Generate examples from the LeRobot dataset."""        
        
        print("_generate_examples")
        # Iterate through all frames in the dataset
        for idx in range(len(self._dataset)):
            print("generate: {idx}")
            # dataset[idx] already returns a dict with all features
            # thanks to LeRobotDataset's __getitem__ implementation
            yield idx, self._dataset[idx]
