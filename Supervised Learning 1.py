import torch
import torch.nn as nn   #torch
import torchvision
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tqdm import tqdm   #the training progress show
from torch.optim import Adam, Rprop #the learning rules for the weight optimzation
from torch.nn.functional import normalize
from torchvision import datasets
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda # lambda self defined 
from torch.utils.data import DataLoader


device="cuda:3"
print(device)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train.shape, x_test.shape

def edit_data(x, y, method="edit"):
    is_batch = x.ndim == 3
    if method == "edit":
        if is_batch:
            x[:, 0, :10] = 0.0
            for i in range(x.shape[0]):
                x[i, 0, y[i]] = 255
        else:
            x[0, :10] = 0.0
            x[0, y] = 255

def random_label(y):
    if type(y) != np.ndarray:
        label = list(range(10))
        del label[y]
        return np.random.choice(label)
    else:
        label = np.copy(y)
        for i in range(y.shape[0]):
            label[i] = random_label(y[i])
        return label

pos = np.copy(x_train)
neg = np.copy(x_train)
edit_data(pos, y_train)
edit_data(neg, random_label(y_train))


pos = (pos-33.31002426147461 )/78.56748962402344
neg = (neg-33.31002426147461 )/78.56748962402344
pos = pos.reshape(pos.shape[0], -1)
neg = neg.reshape(neg.shape[0], -1)
x_pos = torch.tensor(pos, dtype=torch.float)
x_neg = torch.tensor(neg, dtype=torch.float)

x_train = (x_train-33.31002426147461 )/78.56748962402344
x_train = x_train.reshape(x_train.shape[0], -1)
y_train = y_train.reshape(y_train.shape[0])
x = torch.tensor(x_train, dtype=torch.float)
y = torch.tensor(y_train, dtype=torch.float)

x_test = (x_test-33.31002426147461 )/78.56748962402344
x_test = x_test.reshape(x_test.shape[0], -1)
y_test = y_test.reshape(y_test.shape[0])
x_te = torch.tensor(x_test, dtype=torch.float)
y_te = torch.tensor(y_test, dtype=torch.float)


x_pos, x_neg, x, y = x_pos.cuda(device), x_neg.cuda(device), x.cuda(device), y.cuda(device)
x_te, y_te = x_te.cuda(device), y_te.cuda(device)

def overlay_y_on_x(x, y):
    x_ = x.clone()
    x_[:, :10] *= 0.0
    x_[range(x.shape[0]), y] = x.max()
    return x_
# 实现hinton 说的把label 利用one-hot的方式 加到像素上去 
# 先克隆x的值 得到784 的Tensor 然后清空前十个 再把最大的normlized 的pixel 放到正确的y 或者错误的y上 
# label对就是positive label错就是negative 

class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        
        #self.Sigmoid = torch.nn.Sigmoid()
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.06)
        self.threshold = 2.0 # 为什么是2 distubition
        self.num_epochs = 120
# 基础变量 至于为什么threshhold是2 大家的模板都是2 我没找出来为什么

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 0.01) #batch norm 
        return self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg):
        for i in tqdm(range(self.num_epochs)):
            for b in range (60):
             g_pos = self.forward(x_pos[b*1000: (b+1)*1000]).pow(2).mean(1)
             g_neg = self.forward(x_neg[b*1000: (b+1)*1000]).pow(2).mean(1)
             #g_pos = self.forward(x_pos).pow(2).mean(1)
             #g_neg = self.forward(x_neg).pow(2).mean(1)
             # The following loss pushes pos (neg) samples to
             # values larger (smaller) than the self.threshold.
             loss = torch.log(1 + torch.exp(torch.cat([-g_pos + self.threshold, g_neg - self.threshold]))).mean()
             self.opt.zero_grad()
             # this backward just compute the derivative and hence
             # is not considered backpropagation.
             loss.backward()
             self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach() #Returns a new Tensor, detached from the current graph.

class Net(torch.nn.Module):

    def __init__(self, dims):
        super().__init__()
        self.layers = []
        for d in range(len(dims) - 1):
            self.layers += [Layer(dims[d], dims[d + 1]).cuda(device)]

    def predict(self, x):
        goodness_per_label = []
        for label in range(10):
            h = overlay_y_on_x(x, label)
            goodness = []
            for layer in self.layers:
                h = layer(h)
                goodness += [h.pow(2).mean(1)]
            goodness_per_label += [sum(goodness).unsqueeze(1)]
        goodness_per_label = torch.cat(goodness_per_label, 1)
        return goodness_per_label.argmax(1)

    def train(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train(h_pos, h_neg)

# 在每个layer里面进行训练 实际训练模式就是Layer的性质

if __name__ == "__main__":
    torch.manual_seed(123)

    net = Net([784, 2000, 2000, 2000, 2000])
    net.train(x_pos, x_neg)

print('train score:', 100*net.predict(x[0: 1000]).eq(y[0:1000]).float().mean().item(),"%")
print('test score:', 100*net.predict(x_te[0: 2000]).eq(y_te[0: 2000]).float().mean().item(),"%")
print('test score:', 100*net.predict(x_te[2000: 4000]).eq(y_te[2000: 4000]).float().mean().item(),"%")
print('test score:', 100*net.predict(x_te[4000: 6000]).eq(y_te[4000: 6000]).float().mean().item(),"%")
print('test score:', 100*net.predict(x_te[6000: 8000]).eq(y_te[6000: 8000]).float().mean().item(),"%")
print('test score:', 100*net.predict(x_te[8000: 10000]).eq(y_te[8000: 10000]).float().mean().item(),"%")
print('final accuracy', (100*net.predict(x_te).eq(y_te).float().mean().item()+100*net.predict(x_te[2000: 4000]).eq(y_te[2000: 4000]).float().mean().item()+100*net.predict(x_te[4000: 6000]).eq(y_te[4000: 6000]).float().mean().item()+100*net.predict(x_te[6000: 8000]).eq(y_te[6000: 8000]).float().mean().item()+100*net.predict(x_te[8000: 10000]).eq(y_te[8000: 10000]).float().mean().item())/5, '%')   
