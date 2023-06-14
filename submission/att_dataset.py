import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision.datasets.folder import torchvision


class AttDataset(Dataset):
    def __init__(self, csv_file, attribute_columns, image_dir="", transform=None, multiply=1, attribute_encoders=None, loader=default_loader):
        if isinstance(csv_file, pd.DataFrame):
            self.df = csv_file
        else:
            self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        self.multiply = multiply
        self.attribute_columns = attribute_columns
        self.attribute_encoders = attribute_encoders or self.create_attribute_encoders()
        self.attribute_decoders = self.create_attribute_decoders(
            self.attribute_encoders)
        self.loader = loader

    def create_attribute_encoders(self):
        attribute_encoders = {}
        for col in self.attribute_columns:
            attribute_encoders[col] = {
                value: idx for idx, value in enumerate(sorted(self.df[col].unique()))}
        return attribute_encoders

    def create_attribute_decoders(self, attribute_encoders):
        # Create reverse mapping for each dictionary
        attribute_decoders = {}
        for col, encoding in attribute_encoders.items():
            attribute_decoders[col] = {v: k for k, v in encoding.items()}
        return attribute_decoders

    def __len__(self):
        return len(self.df) * self.multiply

    def __getitem__(self, idx):
        idx = idx % len(self.df)
        if torch.is_tensor(idx):
            idx = idx.tolist()
        image_path = os.path.join(self.image_dir, self.df.loc[idx, 'path'])
        image = self.loader(image_path)
        attributes = [self.attribute_encoders[col][self.df.loc[idx, col]]
                      for col in self.attribute_columns]
        sample = {'image': image, 'attributes': torch.tensor(
            attributes, dtype=torch.long), 'img_path': image_path}
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample
