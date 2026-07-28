"""
Utilities for loading Bridge dataset with VLA latents (hidden_states).
Converts TFRecord format to QC-compatible dict format.

Reference: V-GPS octo/octo/data/dataset.py:375-432 for hidden_states decoding
"""
import os
import glob
import numpy as np

# TensorFlow is only imported when needed (may not be in qc env)
try:
    import tensorflow as tf
    HAS_TF = True
    # Prevent TF from using GPU (V-GPS train.py:65) - JAX will use GPU instead
    tf.config.set_visible_devices([], 'GPU')
except ImportError:
    HAS_TF = False


# ============================================================
# STREAMING DATASET - V-GPS style streaming with QC training params
# Code structure: V-GPS train.py:134-203
# Training params: QC FLAGS (discount, batch_size, action_horizon)
# ============================================================

class BridgeStreamingDataset:
    """
    Streaming dataset wrapper matching QC Dataset.sample_sequence() interface.

    Code structure from V-GPS train.py:201-203, 266
    Format conversion matches QC make_bridge_dataset and sample_sequence
    """

    def __init__(self, tf_dataset, action_horizon, discount):
        self.tf_dataset = tf_dataset
        self.action_horizon = action_horizon
        self.discount = discount
        self._iterator = None
        self._reset_iterator()

        # For compatibility with Dataset interface
        self.size = None  # Unknown for streaming
        self.terminal_locs = np.array([])
        self.initial_locs = np.array([])

    def _reset_iterator(self):
        """Reset iterator - from V-GPS train.py:201-203"""
        self._iterator = iter(self.tf_dataset.iterator(prefetch=0))

    def sample_sequence(self, batch_size, sequence_length, discount):
        """
        Returns QC-compatible batch. Args are ignored (pre-configured).

        From V-GPS train.py:266: batch = next(train_data_iter)
        """
        try:
            octo_batch = next(self._iterator)
        except StopIteration:
            self._reset_iterator()
            octo_batch = next(self._iterator)
        return self._convert_to_qc_format(octo_batch)

    def _convert_to_qc_format(self, octo_batch):
        """
        Convert octo batch to QC sample_sequence format.

        Octo batch structure (with window_size=1, action_horizon=N):
          observation['hidden_states']: (B, 1, hidden_dim)
          action: (B, 1, N, 7)
          td_mask: (B, 1, N) - per-step mask, 1=continue, 0=terminal
          reward: (B, 1, N) - per-step reward
          mc_return: (B,) - scalar MC return

        QC expects (B, N) arrays. We compute cumulative discounted rewards and masks/valid.
        """
        # # Debug: print octo batch shapes before processing
        # print(f"\n[DEBUG] === Octo batch shapes (before processing) ===")
        # print(f"  observation['hidden_states']: {octo_batch['observation']['hidden_states'].shape}")
        # print(f"  next_observation['hidden_states']: {octo_batch['next_observation']['hidden_states'].shape}")
        # print(f"  action: {octo_batch['action'].shape}")
        # print(f"  next_action: {octo_batch['next_action'].shape}")
        # print(f"  td_mask: {octo_batch['td_mask'].shape}")
        # print(f"  reward: {octo_batch['reward'].shape}")

        batch_size = octo_batch['action'].shape[0]
        action_horizon = self.action_horizon

        # Extract observations - with window_size=1, shape is (B, 1, hidden_dim)
        hs = octo_batch['observation']['hidden_states']
        next_hs = octo_batch['next_observation']['hidden_states']

        # Actions - with window_size=1, action_horizon=N: (B, 1, N, 7) -> (B, N, 7)
        actions = octo_batch['action'][:, 0, :, :]
        next_actions = octo_batch['next_action'][:, 0, :, :]

        # Observations for network input - single observation at current timestep
        observations = hs[:, 0, :]        # (B, 1, hidden_dim) -> (B, hidden_dim)
        next_observations = next_hs       # (B, 1, hidden_dim) - agent uses [..., -1, :]

        # # Debug: print QC batch shapes after processing
        # print(f"\n[DEBUG] === QC batch shapes (after processing) ===")
        # print(f"  observations: {observations.shape}")
        # print(f"  next_observations: {next_observations.shape}")
        # print(f"  actions: {actions.shape}")
        # print(f"  next_actions: {next_actions.shape}")
        # print(f"  action_horizon: {action_horizon}")

        # td_mask from octo is now (B, 1, N) after chunking - extract to (B, N)
        # td_mask[i] = 1 means step i is valid/continue, 0 means terminal
        # td_mask is monotonic [1,1,...,0,0] so no accumulate needed
        td_mask_chunk = octo_batch['td_mask'][:, 0, :]  # (B, N)
        td_mask_last_step = octo_batch['td_mask_last_step'][:, 0, :]  # (B, N)

        # masks: 1=bootstrap, 0=don't bootstrap (same as td_mask)
        masks = td_mask_chunk  # (B, N)

        # terminals: 1=terminal, 0=continue (inverse of td_mask)
        terminals = 1.0 - td_mask_chunk  # (B, N)

        # valid: exclude post-terminal steps from loss
        # valid[i] = 1 if step i-1 was NOT terminal (i.e., td_mask[i-1] = 1)
        valid = np.ones((batch_size, action_horizon), dtype=np.float32)
        valid[:, 1:] = td_mask_last_step[:, :-1]  # valid[i] = td_mask[i-1]

        # Rewards from octo is now (B, 1, N) after chunking - extract to (B, N)
        # Compute cumulative discounted rewards: r[0] + γ*r[1] + γ²*r[2] + ...
        reward_chunk = octo_batch['reward'][:, 0, :]  # (B, N)
        discount_powers = self.discount ** np.arange(action_horizon)  # [1, γ, γ², ..., γ^(N-1)]
        rewards = np.cumsum(reward_chunk * discount_powers, axis=-1)  # (B, N)

        return {
            'observations': observations.astype(np.float32),
            'actions': actions.astype(np.float32),
            'rewards': rewards.astype(np.float32),
            'masks': masks.astype(np.float32),
            'terminals': terminals.astype(np.float32),
            'valid': valid.astype(np.float32),
            'next_observations': next_observations.astype(np.float32),  # (B, 1, hidden_dim) - agent uses [..., -1, :]
            'next_actions': next_actions.astype(np.float32),
        }


def make_bridge_streaming_dataset(
    dataset_dir,
    split='train',
    batch_size=256,
    action_horizon=5,
    discount=0.99,
    shuffle_buffer_size=50000,
    num_final_repeat=3,
    skip_unlabeled=True,
):
    """
    Create streaming Bridge dataset using octo pipeline.

    Code structure: V-GPS train.py:134-136
    Training params: QC FLAGS (discount, batch_size, action_horizon)

    Args:
        dataset_dir: Path to Bridge dataset (e.g., .../bridge_dataset/1.0.0)
        split: 'train' or 'val'
        batch_size: From QC config['batch_size']
        action_horizon: From QC FLAGS.horizon_length (number of future actions to predict)
        discount: From QC FLAGS.discount
        shuffle_buffer_size: Shuffle buffer size
        num_final_repeat: Number of final steps to treat as success

    Returns:
        BridgeStreamingDataset instance with sample_sequence() interface
    """
    if not HAS_TF:
        raise ImportError("TensorFlow is required for streaming Bridge dataset")

    # Import octo modules (V-GPS train.py:23-24)
    try:
        from octo.data.dataset import make_interleaved_dataset
        from octo.data.oxe import make_oxe_dataset_kwargs_and_weights
        from octo.data.utils.data_utils import NormalizationType
    except ImportError:
        raise ImportError(
            "octo submodule not found. Please add it:\n"
            "  git submodule add https://github.com/mobile-pi/octo.git octo\n"
            "  pip install git+https://github.com/kvablack/dlimp@5edaa4691567873d495633f2708982b42edf1972"
        )

    print(f"[Bridge Streaming] Creating dataset from {dataset_dir}, split='{split}'")
    print(f"[Bridge Streaming] batch_size={batch_size}, action_horizon={action_horizon}, discount={discount}")

    # Get dataset kwargs (V-GPS train.py:125-130)
    # Use BOUNDS normalization to normalize actions to [-1, 1]
    dataset_kwargs, weights = make_oxe_dataset_kwargs_and_weights(
        data_mix="bridge",
        data_dir=dataset_dir,
        discount=discount,
        num_final_repeat=num_final_repeat,
        force_recompute_dataset_statistics=False,
        action_proprio_normalization_type=NormalizationType.BOUNDS,
    )

    # Create interleaved dataset (V-GPS train.py:134-136)
    dataset = make_interleaved_dataset(
        dataset_kwargs, weights,
        train=(split == 'train'),
        shuffle_buffer_size=shuffle_buffer_size,
        batch_size=batch_size,
        balance_weights=True,       # V-GPS data_config.py:122
        traj_transform_threads=48,  # V-GPS data_config.py:50
        traj_read_threads=48,       # V-GPS data_config.py:51
        traj_transform_kwargs=dict(
            window_size=1,  # single observation (no history)
            action_horizon=action_horizon,  # number of future actions to predict
            goal_relabeling_strategy=None,
            subsample_length=100,
            skip_unlabeled=skip_unlabeled,  # V-GPS: skip trajectories without reward labels
        ),
        frame_transform_kwargs=dict(
            resize_size={},
            image_augment_kwargs={},
        ),
    )

    print(f"[Bridge Streaming] Dataset created successfully")
    return BridgeStreamingDataset(dataset, action_horizon, discount)
