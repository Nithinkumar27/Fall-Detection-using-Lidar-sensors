import os
import glob
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch

class CustomDataset(Dataset):
    def __init__(self, root_dir, num_points=400, test_size=0.2, val_size=0.3, random_seed=42):
        self.num_points = num_points
        self.test_size = test_size
        self.val_size = val_size
        self.random_seed = random_seed
        self.class_map = {}
        self.data, self.labels = self.parse_dataset(root_dir)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        point_cloud = self.data[idx]
        label = self.labels[idx]
        return point_cloud, label

    def parse_dataset(self, root_dir):
        train_points = []
        train_labels = []
        folders = glob.glob(os.path.join(root_dir, '*'))

        for i, folder in enumerate(folders):
            class_name = os.path.basename(folder)
            self.class_map[i] = class_name

            class_files = glob.glob(os.path.join(folder, '*.csv'))

            for file in class_files:
                df = pd.read_csv(file)

                if len(df) >= self.num_points:
                    sampled_data = df.sample(self.num_points).values
                    train_points.append(sampled_data)
                    train_labels.append(i)

        return np.array(train_points), np.array(train_labels)

    def split_data(self):
        train_data, test_data, train_labels, test_labels = train_test_split(
            self.data, self.labels, test_size=self.test_size, random_state=self.random_seed
        )
        train_data, val_data, train_labels, val_labels = train_test_split(
            train_data, train_labels, test_size=self.val_size, random_state=self.random_seed
        )
        return train_data, val_data, test_data, train_labels, val_labels, test_labels

if __name__ == '__main__':
    # Modify this path to your dataset directory
    dataset_path = '/content/pytorch/data/anotation1/data_real'
    num_points = 400

    custom_dataset = CustomDataset(dataset_path, num_points=num_points)
    train_data, val_data, test_data, train_labels, val_labels, test_labels = custom_dataset.split_data()

    train_dataset = list(zip(train_data, train_labels))
    val_dataset = list(zip(val_data, val_labels))
    test_dataset = list(zip(test_data, test_labels))

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
