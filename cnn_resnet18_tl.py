
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
import torch.optim as optim
import torchvision.transforms as transforms
from matplotlib.legend_handler import HandlerLine2D
from torch.utils.data.dataset import Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

__author__ = 'Bar Katz'


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 50, 5)
        self.conv1_bn = nn.BatchNorm2d(50)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(50, 16, 5)
        self.conv2_bn = nn.BatchNorm2d(16)

        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc1_bn = nn.BatchNorm1d(120)
        self.fc2 = nn.Linear(120, 84)
        self.fc2_bn = nn.BatchNorm1d(84)
        self.fc3 = nn.Linear(84, 10)
        self.fc3_bn = nn.BatchNorm1d(10)

    def forward(self, x):
        x = self.pool(f.relu(self.conv1_bn(self.conv1(x))))
        x = self.pool(f.relu(self.conv2_bn(self.conv2(x))))
        x = x.view(-1, 16 * 5 * 5)
        x = f.relu(self.fc1_bn(self.fc1(x)))
        x = f.relu(self.fc2_bn(self.fc2(x)))
        x = self.fc3_bn(self.fc3(x))
        return f.log_softmax(x, dim=1)


def train(epoch, model, train_loader, optimizer):
    model.train()

    train_loss = 0
    correct_train = 0
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data, labels) in enumerate(train_loader):
        data = data.cuda()
        labels = labels.cuda()

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, labels)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        pred = output.data.max(1, keepdim=True)[1]
        correct_train += pred.eq(labels.data.view_as(pred)).cpu().sum()

        print('Train Epoch: {}\tStatus: {}'.format(epoch, (batch_idx * batch_size) / (len(train_loader) * batch_size)))

    train_loss /= len(train_loader)
    print('Train Epoch: {}\tAccuracy {}/{} ({:.0f}%)\tAverage loss: {:.6f}'.format(
        epoch, correct_train, len(train_loader) * batch_size,
                              100. * correct_train / (len(train_loader) * batch_size), train_loss))

    return train_loss


def validation(epoch, model, valid_loader):
    model.eval()

    valid_loss = 0
    correct_valid = 0
    criterion = nn.CrossEntropyLoss()
    for data, label in valid_loader:
        data = data.cuda()
        label = label.cuda()

        output = model(data)
        valid_loss += criterion(output, label).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct_valid += pred.eq(label.data.view_as(pred)).cpu().sum()

    valid_loss /= (len(valid_loader) * batch_size)
    print('Validation Epoch: {}\tAccuracy: {}/{} ({:.0f}%)\tAverage loss: {:.6f}'.format(
        epoch, correct_valid, (len(valid_loader) * batch_size),
        100. * correct_valid / (len(valid_loader) * batch_size), valid_loss))

    return valid_loss


def test(model, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    criterion = nn.CrossEntropyLoss()
    predictions = list()
    y_label = []
    y_pred = []

    for data, target in test_loader:
        data = data.cuda()
        target = target.cuda()

        output = model(data)
        test_loss += criterion(output, target).item()
        pred = output.data.max(1, keepdim=True)[1]
        pred_vec = pred.view(len(pred))
        for x in pred_vec:
            predictions.append(x.item())
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
        y_label.append(target.item())
        y_pred.append(pred.item())

    test_loss /= len(test_loader.dataset)
    print('\nTest set:\tAccuracy: {}/{} ({:.0f}%)\tAverage loss: {:.4f}'.format(
        correct, len(test_loader.dataset), 100. * correct / len(test_loader.dataset), test_loss))

    conf_mat = confusion_matrix(y_label, y_pred)
    print(conf_mat)

    return predictions


# consts
output_size = 10

# parameters
epochs = 1
learning_rate = 0.01
batch_size = 128
valid_split = 0.2

write_test_pred = False
draw_loss_graph = False


def get_data_loaders():
    tran = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.1307, 0.1307, 0.1307,), (0.3081, 0.3081, 0.3081,))])

    train_ds = datasets.CIFAR10('./train_data', train=True, download=True, transform=tran)
    test_ds = datasets.CIFAR10('./test_data', train=False, download=True, transform=tran)

    num_train = len(train_ds)
    indices = list(range(num_train))
    split = int(np.floor(valid_split * num_train))

    valid_idx = np.random.choice(indices, size=split, replace=False)
    train_idx = list(set(indices) - set(valid_idx))

    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, sampler=train_sampler, num_workers=1)

    valid_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, sampler=valid_sampler, num_workers=1)

    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size, shuffle=True, num_workers=1)

    return train_loader, valid_loader, test_loader


def train_model(model, train_loader, valid_loader, test_loader):
    optimizer = optim.SGD(model.fc.parameters(), lr=learning_rate, momentum=0.9)

    x = list()
    train_y = list()
    valid_y = list()
    for epoch in range(1, epochs + 1):
        train_loss = train(epoch, model, train_loader, optimizer)
        valid_loss = validation(epoch, model, valid_loader)

        x.append(epoch)
        train_y.append(train_loss)
        valid_y.append(valid_loss)
    predictions = test(model, test_loader)

    options(x, train_y, valid_y, predictions)


def options(x, train_y, valid_y, predictions):
    if write_test_pred:
        write_to_file(predictions)

    if draw_loss_graph:
        draw_loss(x, train_y, valid_y)


def write_to_file(predictions):
    with open("test.pred", "w") as file:
        for pred in predictions:
            file.write(str(pred) + '\n')
    file.close()


def draw_loss(x, train_y, valid_y):
    fig = plt.figure(0)
    fig.canvas.set_window_title('Train loss VS Validation loss')
    plt.axis([0, epochs + 1, 0, 2])
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)

    train_graph, = plt.plot(x, train_y, 'r--', label='Train loss')

    plt.plot(x, valid_y, 'b', label='Validation loss')

    plt.legend(handler_map={train_graph: HandlerLine2D(numpoints=3)})
    plt.show()


def main():
    # init_params()

    train_loader, valid_loader, test_loader = get_data_loaders()

    model = models.resnet18(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 10)

    # model = Net()

    model = model.cuda()

    train_model(model, train_loader, valid_loader, test_loader)


if __name__ == '__main__':
    main()
