import os
import cv2
import numpy as np


def get_label(mask_path):
    mask = cv2.imread(mask_path, 0)
    return 1 if np.sum(mask) > 0 else 0


def load_sequence(frame_dir, mask_dir):
    frame_files = sorted(os.listdir(frame_dir))
    mask_files = sorted(os.listdir(mask_dir))

    features = []
    labels = []

    for f, m in zip(frame_files, mask_files):
        if not f.endswith('.tif'):
            continue
        if not m.endswith('.bmp'):
            continue
        frame_path = os.path.join(frame_dir, f)
        mask_path = os.path.join(mask_dir, m)

        img = cv2.imread(frame_path, 0)
       
        img = cv2.resize(img, (64, 64))
        img = img.flatten() / 255.0

        label = get_label(mask_path)

        features.append(img)
        labels.append(label)

    return np.array(features), np.array(labels)