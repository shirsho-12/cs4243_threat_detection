from torch.utils.data import Dataset, DataLoader
import cv2
from scripts.utils import split_data
import torchvision.transforms as T
import numpy as np

train_transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(10),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

val_transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transforms = T.Compose([
    T.ToPILImage(),
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

transforms = {
    'train': train_transforms,
    'val': val_transforms,
    'test': test_transforms
}


class ThreatDataset(Dataset):
    def __init__(self, data, loader_type='train', transforms=None, color_space='rgb'):
        self.folder_names = ['carrying', 'threat', 'normal']
        self.data = data
        self.color_space = color_space
        self.transform = transforms

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = self.data[idx]
        label = self.folder_names.index(data.parent.name)
        image = cv2.imread(str(data))
        if self.color_space == 'rgb':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        elif self.color_space == 'gray':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Need to broadcast the gray image to 3 channels
            image = np.dstack((image, image, image))
        elif self.color_space == 'hsv':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        elif self.color_space == 'lab':
            image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        if self.transform:
            image = self.transform(image)
        return image, label


def get_loaders(transforms=transforms, color_space='rgb', batch_size=32, num_workers=4, pin_memory=True):

    train_data, val_data, test_data = split_data('data')
    train_dataset = ThreatDataset(
        train_data, transforms=transforms['train'], color_space=color_space)
    val_dataset = ThreatDataset(
        val_data, transforms=transforms['val'], color_space=color_space)
    test_dataset = ThreatDataset(
        test_data, transforms=transforms['test'], color_space=color_space)

    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size,
                            num_workers=num_workers, pin_memory=pin_memory, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size,
                             num_workers=num_workers, pin_memory=pin_memory, shuffle=True)

    return train_loader, val_loader, test_loader
