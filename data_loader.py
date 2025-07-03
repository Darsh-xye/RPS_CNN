import tensorflow_datasets as tfds
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx].permute(1, 2, 0).numpy()
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label

def load_data():
    ds_train = tfds.load(name='rock_paper_scissors', split='train')
    ds_test = tfds.load(name='rock_paper_scissors', split='test')

    train_images = torch.tensor(np.array([example['image'] for example in ds_train]))
    train_labels = torch.tensor(np.array([example['label'] for example in ds_train]))
    test_images = torch.tensor(np.array([example['image'] for example in ds_test]))
    test_labels = torch.tensor(np.array([example['label'] for example in ds_test]))

    train_images = train_images.float() / 255.0
    test_images = test_images.float() / 255.0
    train_images = train_images.permute(0, 3, 1, 2)
    test_images = test_images.permute(0, 3, 1, 2)

    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(300, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((300, 300)),
        transforms.ToTensor()
    ])

    return train_images, train_labels, test_images, test_labels, train_transform, test_transform
