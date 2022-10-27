from torchvision.models import resnet18
from torch.optim.lr_scheduler import StepLR
# from torch import nn
import torch.optim as optim
import torch.nn as nn
from scripts.trainer import Trainer
from scripts.dataloader import get_loaders

train_loader, val_loader, test_loader = get_loaders()

model = resnet18(pretrained=False)
model.fc = nn.Linear(512, 3)

optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()
scheduler = StepLR(optimizer, step_size=5, gamma=0.1)

trainer = Trainer(model, optimizer, criterion, scheduler, 'cuda')
trainer.train(train_loader, val_loader, epochs=50)
