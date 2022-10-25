import torch
import torch.nn.functional as F

from tqdm import tqdm

# For NLLLoss, we need to use the logit distribution
# https://discuss.pytorch.org/t/what-is-the-difference-between-nllloss-and-crossentropyloss/15553
# The CrossEntropyLoss combines the LogSoftmax and NLLLoss in one single class
# https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
# https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html

class Trainer:
    def __init__(self, model, optimizer, criterion, scheduler, device):
        # The trainer uses a one-hot distribution for the labels, so we need to use the CrossEntropyLoss
        # instead of the NLLLoss
        # Using FCC layer as the last layer, we can try to use basic loss functions like MSE or L1
        
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.best_acc = 50
        if (device == 'cuda') and torch.cuda.is_available():
            self.device = torch.device('cuda')
        else:
            self.device = torch.device('cpu')

    def train(self, train_loader, val_loader, epochs=10):
        self.model.to(self.device)
        total = 0
        correct = 0
        for epoch in range(epochs):
            print(f"EPOCH {epoch}")
            self.model.train()
            tq = tqdm(enumerate(train_loader))
            for i, (x, y) in tq:
                x = x.to(self.device)
                y_label = y
                y = F.one_hot(y, num_classes=3).to(self.device).float()
                total += y.size(0)
                self.optimizer.zero_grad()
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)
                loss.backward()
                self.optimizer.step()
                
                # Calculate Accuracy - Only for softmax/logit distributions  
                _, predicted = torch.max(y_pred.data, 1)
                correct += (predicted == y_label).sum().item()
                tq.set_postfix(loss=loss.item(), acc=correct/total)
                if i % 100 == 0:
                    print(f'Epoch: {epoch}, Loss: {loss.item()}')
            self.validate(val_loader)
            print(f'Epoch: {epoch}, Accuracy: {correct/total}')

    def validate(self, val_loader):
        self.model.eval()
        total = 0
        correct = 0
        with torch.no_grad():
            tq = tqdm(enumerate(val_loader))
            for i, (x, y) in tq:
                x = x.to(self.device)
                y_label = y
                y = F.one_hot(y, num_classes=3).to(self.device).float()
                y_pred = self.model(x)
                loss = self.criterion(y_pred, y)

                total += y.size(0)
                _, predicted = torch.max(y_pred.data, 1)
                print(predicted)
                
                correct += (predicted == y_label).sum().item()
                if i % 100 == 0:
                    print(f'Validation Loss: {loss.item()}')
            print(f'Validation Accuracy: {correct/total}')
            if correct/total > self.best_acc:
                self.best_acc = correct/total
                print('Saving model...')
                torch.save(self.model.state_dict(), 'best_model.pth')
    
    def test(self, test_loader):
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
                correct += (predicted == y_label).sum().item()
                if i % 100 == 0:
                    print(f'Test Loss: {loss.item()}')
        print(f'Accuracy: {100 * correct / total}')
    
    def save_model(self, path):
        torch.save(self.model.state_dict(), path)      

    def load_model(self, path):
        self.model.load_state_dict(torch.load(path)) 