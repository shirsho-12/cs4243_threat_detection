import random
import cv2
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torchvision.models import resnet18

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter()


def metrics(labels, preds):
    roc = roc_auc_score(labels, preds, multi_class='ovr')
    # acc = accuracy_score(labels, preds)
    return roc


class TripletDataset(Dataset):
    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.classes = ['carrying', 'threat', 'normal']
        self.carry_data = [
            x for x in dataset if x.parent.name == self.classes[0]]
        random.shuffle(self.carry_data)
        self.threat_data = [
            x for x in dataset if x.parent.name == self.classes[1]]
        random.shuffle(self.threat_data)
        self.normal_data = [
            x for x in dataset if x.parent.name == self.classes[2]]
        random.shuffle(self.normal_data)
        self.data_dict = {0: self.carry_data,
                          1: self.threat_data, 2: self.normal_data}
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        anchor = self.dataset[idx]

        anchor_label = self.classes.index(anchor.parent.name)
        # print(anchor_label, len(self.data_dict[anchor_label]))
        anchor_image = cv2.imread(str(anchor))

        positive_sample = random.choice(self.data_dict[anchor_label])
        positive_image = cv2.imread(str(positive_sample))
        # positive_label = anchor_label

        negative_label = random.choice(
            [x for x in self.data_dict.keys() if x != anchor_label])
        negative_sample = random.choice(self.data_dict[negative_label])
        negative_image = cv2.imread(str(negative_sample))

        anchor_image = cv2.cvtColor(anchor_image, cv2.COLOR_BGR2RGB)
        if self.transform:
            anchor_image = self.transform(anchor_image)
            negative_image = self.transform(negative_image)
            positive_image = self.transform(positive_image)

        # positive_label, negative_label)
        return (anchor_image, positive_image, negative_image, anchor_label)


class TripletLossTrainer:
    def __init__(self, model, optimizer, scheduler, device='cuda'):
        # The trainer uses a one-hot distribution for the labels, so we need to use the CrossEntropyLoss
        # instead of the NLLLoss
        # Using FCC layer as the last layer, we can try to use basic loss functions like MSE or L1

        self.model = model
        self.optimizer = optimizer
        self.criterion = nn.TripletMarginLoss(margin=1.0, p=2)
        self.scheduler = scheduler
        self.best_acc = 1/3
        self.train_acc_arr = []
        self.val_acc_arr = []
        self.train_losses = []
        self.val_losses = []
        self.test_acc = 0
        self.test_loss = 0
        if (device == 'cuda') and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')
        self.writer = writer

    def get_loss(self, anchor, positive, negative):
        anchor = anchor.to(self.device)
        positive = positive.to(self.device)
        negative = negative.to(self.device)
        anchor_embedding = self.model(anchor)
        positive_embedding = self.model(positive)
        negative_embedding = self.model(negative)

        loss = self.criterion(
            anchor_embedding, positive_embedding, negative_embedding)

        # d_p = torch.sum((anchor_embedding - positive_embedding)**2, dim=1)
        # d_n = torch.sum((anchor_embedding - negative_embedding)**2, dim=1)
        # loss = torch.sum(torch.clamp(d_p - d_n + 0.2, min=0.0))
        return loss

    def train(self, train_loader, val_loader, epochs=10, name='model'):
        self.model.to(self.device)
        total = 0
        correct = 0
        total_loss = 0
        for epoch in range(epochs):
            print(f"EPOCH {epoch}")
            self.model.train()
            tq = tqdm(enumerate(train_loader))
            for i, (anchor, positive, negative, y) in tq:
                # x = x.to(self.device)
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                y_label = y
                y = F.one_hot(y, num_classes=3).to(self.device).float()
                total += y.size(0)

                self.optimizer.zero_grad()
                y_pred = self.model(anchor)
                loss = self.get_loss(anchor, positive, negative)

                loss.backward()
                self.optimizer.step()
                total_loss += loss

                # Calculate Accuracy - Only for softmax/logit distributions
                _, predicted = torch.max(y_pred.data, 1)
                correct += (predicted.cpu() == y_label).sum().item()
                tq.set_postfix(loss=loss.item(), acc=correct/total)
                if i % 100 == 0:
                    print(f'Epoch: {epoch}, Loss: {loss.item()}')

                # return y, y_pred
                roc = metrics(y_label.detach().numpy(), F.softmax(
                    y_pred, dim=1).detach().numpy())
                writer.add_scalar("Loss/train", loss, epoch)
                writer.add_scalar("ROC/train", roc, epoch)
            writer.add_scalar("Accuracy/train", correct/total, epoch)
            self.train_acc_arr.append(correct/total)
            self.train_losses.append(total_loss)
            self.validate(val_loader, epoch, f'{name}_{epoch}')
            self.scheduler.step()
            writer.add_hparams({'hparam/lr': self.scheduler.get_last_lr()[0]})
            print(f'Epoch: {epoch}, Accuracy: {correct/total}')
        writer.flush()

    def validate(self, val_loader, epoch, name):
        self.model.eval()
        total = 0
        correct = 0
        total_loss = 0
        with torch.no_grad():
            tq = tqdm(enumerate(val_loader))
            for i, (anchor, positive, negative, y) in tq:
                # x = x.to(self.device)
                anchor = anchor.to(self.device)
                positive = positive.to(self.device)
                negative = negative.to(self.device)
                y_label = y
                y = F.one_hot(y, num_classes=3).to(self.device).float()
                total += y.size(0)
                loss = self.get_loss(anchor, positive, negative)

                total += y.size(0)
                y_pred = self.model(anchor)
                _, predicted = torch.max(y_pred.data, 1)
                # print(predicted)

                total_loss += loss
                correct += (predicted.cpu() == y_label).sum().item()
                if i % 100 == 0:
                    print(f'Validation Loss: {loss.item()}')
                roc = metrics(y_label.detach().numpy(), F.softmax(
                    y_pred, dim=1).detach().numpy())
                writer.add_scalar("Loss/val", loss, epoch)
                writer.add_scalar("ROC/val", roc, epoch)

            writer.add_scalar("Accuracy/val", correct/total, epoch)
            print(f'Validation Accuracy: {correct/total}')
            self.val_acc_arr.append(correct/total)
            self.val_losses.append(total_loss)
            if correct/total > self.best_acc:
                self.best_acc = correct/total
                print('Saving model...')
                self.save_model(name)

    def test(self, test_loader, name='triplet_model'):
        self.model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            for i, (x, y) in tqdm(enumerate(test_loader)):
                x = x.to(self.device)
                y_label = y
                y = F.one_hot(y, num_classes=3).to(self.device).float()
                total += y.size(0)
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)

                _, predicted = torch.max(y_pred.data, 1)
                correct += (predicted.detach().numpy() == y_label).sum().item()
                if i % 100 == 0:
                    print(f'Test Loss: {loss.item()}')
        print(f'Accuracy: {100 * correct / total}')
        self.test_acc = correct/total
        self.test_loss = loss
        self.save_all(name=name)

    def save_model(self, path):
        torch.save(self.model.state_dict(), f'models/{path}.pth')

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path))

    def save_all(self, name):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'train_acc_arr': self.train_acc_arr,
            'val_acc_arr': self.val_acc_arr,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'test_acc': self.test_acc,
            'test_loss': self.test_loss
        }, f'{name}_full.pth')


def split_data(data_dir, train_size=0.8, val_size=0.1):
    random.seed(1234)
    data = Path(data_dir).glob('*/*')
    data = [x for x in data if x.is_file() and x.suffix != '.zip']
    random.shuffle(data)
    train_size = int(len(data) * train_size)
    val_size = int(len(data) * val_size)
    train_data = data[:train_size]
    val_data = data[train_size:train_size+val_size]
    test_data = data[train_size+val_size:]

    return train_data, val_data, test_data


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


def runner(num_epochs=50, batch_size=32, lr=0.001, momentum=0.9, weight_decay=0.0005, step_size=10, gamma=0.1):
    train_data, val_data, test_data = split_data('data')
    print(len(train_data), len(val_data), len(test_data))
    train_dataset = TripletDataset(train_data, transform=transforms['train'])
    val_dataset = TripletDataset(val_data, transform=transforms['val'])
    test_dataset = TripletDataset(test_data, transform=transforms['test'])

    train_loader = DataLoader(train_dataset, batch_size=batch_size)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    model = resnet18(pretrained=False)
    model.fc = nn.Linear(512, 3)
    optimizer = torch.optim.Adam(model.parameters(
    ), lr=lr, weight_decay=weight_decay, amsgrad=True, eps=1e-8, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=step_size, gamma=gamma)

    trainer = TripletLossTrainer(model, optimizer, scheduler)

    trainer.train(train_loader, val_loader,
                  epochs=num_epochs, name='triplet_model')
    trainer.test(test_loader)

    return trainer

# metrics(y.detach().numpy(), y_pred.detach().numpy())
