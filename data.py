from pathlib import Path
from typing import Tuple
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

def preprocess_roi(gray: np.ndarray) -> np.ndarray:
    """
    Manuscript-aligned preprocessing:
      - grayscale
      - CLAHE (clipLimit=2.0, tileGridSize=8x8)
      - Gaussian smoothing
      - resize to 224x224
    """
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    gray = gray.astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    x = clahe.apply(gray)
    x = cv2.GaussianBlur(x, (3, 3), 0)
    x = cv2.resize(x, (224, 224), interpolation=cv2.INTER_AREA)
    return x

class ROIDataset(Dataset):
    """
    CSV format:
      path,label,patient_id,split
    label: 0 benign/background, 1 malignant

    The manuscript requires patient-level separation before ROI generation.
    This loader assumes the CSV already respects that rule.
    """
    def __init__(self, csv_path: str, split: str):
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row["path"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise FileNotFoundError(row["path"])
        img = preprocess_roi(img)
        t = torch.from_numpy(img).float().unsqueeze(0) / 255.0
        y = torch.tensor(float(row["label"]), dtype=torch.float32)
        return t, y
