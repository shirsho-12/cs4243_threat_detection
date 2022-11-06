from torchvision.models import resnet18
from torch.optim.lr_scheduler import StepLR
# from torch import nn
import torch.optim as optim
import torch.nn as nn
from scripts.trainer import Trainer
from scripts.dataloader import get_loaders

if __name__ == '__main__':
    for color_space in ['gray', 'rgb', 'hsv', 'lab']:
        train_loader, val_loader, test_loader = get_loaders(
            color_space=color_space)

        model = resnet18(weights=None)
        model.fc = nn.Linear(512, 3)

        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()
        scheduler = StepLR(optimizer, step_size=10, gamma=0.1)

        trainer = Trainer(model, optimizer, criterion, scheduler, 'cuda')
        trainer.train(train_loader, val_loader, epochs=50,
                      name=f'resnet18_{color_space}')
        trainer.test(test_loader, name=f'vanilla_resnet18_{color_space}')
