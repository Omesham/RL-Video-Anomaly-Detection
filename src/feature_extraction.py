import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import hiera  # Ensure the hiera package is installed
import pickle  # For saving the complete data dictionary
from collections import defaultdict

# ---------------------------
# Load Hiera model function
# ---------------------------
def load_model(model_name, device):
    checkpoint = "mae_k400_ft_k400"  # finetuned on Kinetics400
    if model_name == "hiera_base_16x224":
        hiera_model = hiera.hiera_base_16x224  # ~909MB
    elif model_name == "hiera_large_16x224":
        hiera_model = hiera.hiera_large_16x224  # ~2.72GB
    elif model_name == "hiera_huge_16x224":
        hiera_model = hiera.hiera_huge_16x224  # ~7.9GB
    else:
        raise ValueError("Unknown model name")
    model_backbone = hiera_model(pretrained=True, checkpoint=checkpoint).to(device)
    # Remove classification head to obtain the feature vector
    model_backbone.head = nn.Identity()
    model_backbone.eval()
    return model_backbone

# ---------------------------
# Sliding Window Dataset for PED2
# ---------------------------
class SlidingWindowDatasetPED2(Dataset):
    """
    This dataset groups frames from the PED2 dataset into rolling windows of 16 consecutive frames.
    For each window, the label for the center frame is determined as follows:
      - In 'train' mode, all frames are assumed to be normal (label 0).
      - In 'test' mode, the corresponding ground-truth mask is checked (from a sibling folder named "<sequence>_gt")
        and if any pixel is non-zero, the frame is marked as anomalous (label 1).
    
    Each sample returns:
      - A tensor of shape [1, 3, 16, 224, 224] ready for Hiera,
      - The corresponding label (0: normal, 1: anomalous),
      - The sequence identifier,
      - The filename of the center frame (for metadata).
    """
    def __init__(self, dataset_path, mode, window_size=16, resize_shape=(224, 224)):
        self.dataset_path = dataset_path
        self.mode = mode  # 'train' or 'test'
        self.window_size = window_size
        self.resize_shape = resize_shape  # (width, height) required for Hiera
        
        # Get list of frame paths from the dataset.
        self.frame_paths, self.sequence_ids = self.get_dataset_paths(dataset_path)
        
        # Group frames by sequence and create sliding windows.
        self.sequence_windows = self.create_sliding_windows()
        
    def is_image_file(self, filename):
        valid_extensions = {'.tif', '.tiff', '.bmp', '.jpg', '.jpeg', '.png'}
        _, ext = os.path.splitext(filename)
        return ext.lower() in valid_extensions

    def get_dataset_paths(self, dataset_path):
        """
        PED2 is organized into "Train" and "Test" folders,
        with each sequence in its own subfolder.
        Returns a list of frame file paths and corresponding sequence identifiers.
        """
        frame_paths = []
        sequence_ids = []
        data_folder = os.path.join(dataset_path, "Train" if self.mode == 'train' else "Test")
        for seq in sorted(os.listdir(data_folder)):
            if seq.endswith("_gt"):  # Skip ground truth mask folders
                continue
            seq_path = os.path.join(data_folder, seq)
            if not os.path.isdir(seq_path):
                continue
            # List image files (e.g., .jpg, .png, etc.)
            frames = sorted([f for f in os.listdir(seq_path) if self.is_image_file(f)])
            for f in frames:
                frame_paths.append(os.path.join(seq_path, f))
                sequence_ids.append(seq)
        return frame_paths, sequence_ids

    def create_sliding_windows(self):
        """
        Groups frames by sequence and creates a list of sliding windows.
        Each window is a dict with:
          - window_paths: list of 16 consecutive frame paths,
          - label: anomaly label for the center frame,
          - sequence: the sequence identifier,
          - center_frame: the center frame's path.
        """
        # Organize frame paths by sequence identifier
        seq_dict = defaultdict(list)
        for path, seq in zip(self.frame_paths, self.sequence_ids):
            seq_dict[seq].append(path)
            
        windows = []
        for seq, paths in seq_dict.items():
            # Sort frames by frame number (if filenames are numeric)
            try:
                paths.sort(key=lambda x: int(os.path.splitext(os.path.basename(x))[0]))
            except ValueError:
                paths.sort()
            n_frames = len(paths)
            if n_frames >= self.window_size:
                for i in range(n_frames - self.window_size + 1):
                    window_paths = paths[i:i + self.window_size]
                    center_frame = window_paths[self.window_size // 2]
                    # Determine label for the center frame:
                    label = 0  # default normal
                    if self.mode == 'test':
                        # Assume ground truth masks are in a sibling folder named "<sequence>_gt"
                        seq_folder = os.path.dirname(paths[0])
                        # Use the current sequence id (seq) for naming the gt folder.
                        gt_folder = os.path.join(os.path.dirname(seq_folder), seq + "_gt")
                        gt_mask_path = os.path.join(gt_folder, os.path.splitext(os.path.basename(center_frame))[0] + ".bmp")
                        if os.path.exists(gt_mask_path):
                            gt_mask = cv2.imread(gt_mask_path, cv2.IMREAD_GRAYSCALE)
                            if gt_mask is not None and np.any(gt_mask > 0):
                                label = 1
                    windows.append({
                        'window_paths': window_paths,
                        'label': label,
                        'sequence': seq,
                        'center_frame': center_frame
                    })
        return windows

    def __len__(self):
        return len(self.sequence_windows)

    def __getitem__(self, idx):
        window_info = self.sequence_windows[idx]
        window_paths = window_info['window_paths']
        label = window_info['label']
        frames = []
        for path in window_paths:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is None:
                raise ValueError(f"Error reading image: {path}")
            img = cv2.resize(img, self.resize_shape, interpolation=cv2.INTER_AREA)
            img_tensor = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1) / 255.0
            frames.append(img_tensor)
        frames = torch.stack(frames)  # shape: (window_size, 3, 224, 224)
        frames = frames.permute(1, 0, 2, 3)  # now: (3, 16, 224, 224)
        frames = frames.unsqueeze(0)  # add batch dim: (1, 3, 16, 224, 224)
        return frames, label, window_info['sequence'], window_info['center_frame']

# ---------------------------
# Feature Extraction and Metadata Saving Pipeline for PED2
# ---------------------------
def extract_and_save_features_PED2(dataset_path,
                                   model_name="hiera_large_16x224",
                                   device="cuda",
                                   batch_size=1,
                                   output_feature_file="features_PED2_test.npy",
                                   output_label_file="labels_PED2_test.npy",
                                   output_meta_file="meta_PED2_test.pkl"):
    """
    1. Loads the Hiera model.
    2. Creates a sliding window dataset from the PED2 dataset (window size: 16 frames).
    3. For each window, extracts a feature vector using the pre-trained Hiera model.
    4. Saves the features, labels, and metadata (including video IDs and per-video window counts) to disk.
    """
    device = torch.device(device if torch.cuda.is_available() else "cpu")
    model = load_model(model_name, device)
    # Use 'test' mode for evaluation.
    dataset = SlidingWindowDatasetPED2(dataset_path, mode='train', window_size=16, resize_shape=(224, 224))
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    features_list = []
    labels_list = []
    metadata_list = []  # list of tuples: (sequence, center_frame_filename)
    
    with torch.no_grad():
        for batch in dataloader:
            frames, labels, sequences, center_frames = batch
            frames = frames.to(device)
            B = frames.shape[0]
            for i in range(B):
                sample = frames[i]  # shape: (1, 3, 16, 224, 224)
                feature = model(sample)  # Expected shape: (1, d)
                features_list.append(feature.cpu().numpy())
            labels_list.extend(labels.numpy().tolist())
            for seq, center in zip(sequences, center_frames):
                metadata_list.append((seq, os.path.basename(center)))
    
    # Concatenate feature arrays along axis 0 (samples, d)
    features_array = np.concatenate(features_list, axis=0)
    labels_array = np.array(labels_list)
    
    # Save features and labels as separate .npy files
    np.save(output_feature_file, features_array)
    np.save(output_label_file, labels_array)
    print(f"Saved features to {output_feature_file} and labels to {output_label_file}")
    
    # --- Create and save metadata ---
    # Group metadata by sequence to get per-video window counts and an ordered list of video IDs.
    meta_dict = defaultdict(list)
    for seq, center in metadata_list:
        meta_dict[seq].append(center)
    
    video_ids = sorted(meta_dict.keys())
    # For each video, record the number of sliding windows (which serves as the test length for evaluation)
    test_lengths = np.array([len(meta_dict[vid]) for vid in video_ids])
    meta_data = {
        "video_ids": video_ids,           # e.g., ["Test001", "Test002", ...]
        "test_lengths": test_lengths,       # e.g., [120, 150, ...]
        "metadata_per_video": dict(meta_dict)  # e.g., {"Test001": ["009.tif", "010.tif", ...], ...}
    }
    with open(output_meta_file, "wb") as f:
        pickle.dump(meta_data, f)
    print(f"Saved metadata to {output_meta_file}")
    
    return features_array, labels_array, meta_data

if __name__ == '__main__':
    dataset_path = '/UCSDped2'
    features, labels, meta = extract_and_save_features_PED2(
        dataset_path,
        model_name="hiera_large_16x224",
        device="cuda",
        batch_size=1,
        output_feature_file="features_PED2_train.npy",
        output_label_file="labels_PED2_train.npy",
        output_meta_file="meta_PED2_train.pkl"
    )
