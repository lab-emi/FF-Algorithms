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



device="cuda:5"
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

pos = np.copy(x_train)
neg1 = np.copy(x_train)
neg2 = np.copy(x_train)
neg3 = np.copy(x_train)
neg4 = np.copy(x_train)
neg5 = np.copy(x_train)
neg6 = np.copy(x_train)
neg7 = np.copy(x_train)
neg8 = np.copy(x_train)
neg9 = np.copy(x_train)

edit_data(pos, y_train)
edit_data(neg1, (y_train + 1)%10)
edit_data(neg2, (y_train + 2)%10)
edit_data(neg3, (y_train + 3)%10)
edit_data(neg4, (y_train + 4)%10)
edit_data(neg5, (y_train + 5)%10)
edit_data(neg6, (y_train + 6)%10)
edit_data(neg7, (y_train + 7)%10)
edit_data(neg8, (y_train + 8)%10)
edit_data(neg9, (y_train + 9)%10)


pos = (pos-33.31002426147461 )/78.56748962402344
neg1 = (neg1-33.31002426147461 )/78.56748962402344
neg2 = (neg2-33.31002426147461 )/78.56748962402344
neg3 = (neg3-33.31002426147461 )/78.56748962402344
neg4 = (neg4-33.31002426147461 )/78.56748962402344
neg5 = (neg5-33.31002426147461 )/78.56748962402344
neg6 = (neg6-33.31002426147461 )/78.56748962402344
neg7 = (neg7-33.31002426147461 )/78.56748962402344
neg8 = (neg8-33.31002426147461 )/78.56748962402344
neg9 = (neg9-33.31002426147461 )/78.56748962402344

pos = pos.reshape(pos.shape[0], -1)
neg1 = neg1.reshape(neg1.shape[0], -1)
neg2 = neg2.reshape(neg2.shape[0], -1)
neg3 = neg3.reshape(neg3.shape[0], -1)
neg4 = neg4.reshape(neg4.shape[0], -1)
neg5 = neg5.reshape(neg5.shape[0], -1)
neg6 = neg6.reshape(neg6.shape[0], -1)
neg7 = neg7.reshape(neg7.shape[0], -1)
neg8 = neg8.reshape(neg8.shape[0], -1)
neg9 = neg9.reshape(neg9.shape[0], -1)

x_pos = torch.tensor(pos, dtype=torch.float)
x_neg1 = torch.tensor(neg1, dtype=torch.float)
x_neg2 = torch.tensor(neg2, dtype=torch.float)
x_neg3 = torch.tensor(neg3, dtype=torch.float)
x_neg4 = torch.tensor(neg4, dtype=torch.float)
x_neg5 = torch.tensor(neg5, dtype=torch.float)
x_neg6 = torch.tensor(neg6, dtype=torch.float)
x_neg7 = torch.tensor(neg7, dtype=torch.float)
x_neg8 = torch.tensor(neg8, dtype=torch.float)
x_neg9 = torch.tensor(neg9, dtype=torch.float)

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

x_pos, x_neg1, x_neg2, x_neg3, x_neg4, x_neg5, x_neg6, x_neg7, x_neg8, x_neg9  = x_pos.cuda(device), x_neg1.cuda(device), x_neg2.cuda(device), x_neg3.cuda(device), x_neg4.cuda(device), x_neg5.cuda(device), x_neg6.cuda(device), x_neg7.cuda(device), x_neg8.cuda(device), x_neg9.cuda(device)
x, y = x.cuda(device), y.cuda(device)
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
        self.num_epochs = 126
# 基础变量 至于为什么threshhold是2 大家的模板都是2 我没找出来为什么

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 0.02) #batch norm 
        return self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))

    def train(self, x_pos, x_neg1, x_neg2, x_neg3, x_neg4, x_neg5, x_neg6, x_neg7, x_neg8, x_neg9):
        for i in tqdm(range(self.num_epochs)):
            for b in range (200):
             g_pos = self.forward(x_pos[b*300: (b+1)*300]).pow(2).mean(1)
             g_neg1 = self.forward(x_neg1[b*300: (b+1)*300]).pow(2).mean(1)
             g_neg2 = self.forward(x_neg2[b*300: (b+1)*300]).pow(2).mean(1)
             g_neg3 = self.forward(x_neg3[b*300: (b+1)*300]).pow(2).mean(1)
             g_neg4 = self.forward(x_neg4[b*300: (b+1)*300]).pow(2).mean(1)
             g_neg5 = self.forward(x_neg5[b*300: (b+1)*300]).pow(2).mean(1)
             g_neg6 = self.forward(x_neg6[b*300: (b+1)*300]).pow(2).mean(1)
             g_neg7 = self.forward(x_neg7[b*300: (b+1)*300]).pow(2).mean(1)
             g_neg8 = self.forward(x_neg8[b*300: (b+1)*300]).pow(2).mean(1)
             g_neg9 = self.forward(x_neg9[b*300: (b+1)*300]).pow(2).mean(1)
             #g_pos = self.forward(x_pos).pow(2).mean(1)
             #g_neg = self.forward(x_neg).pow(2).mean(1)
             # The following loss pushes pos (neg) samples to
             # values larger (smaller) than the self.threshold.
             #loss = torch.log(1 + torch.exp(torch.cat([-g_pos + self.threshold, g_neg1 - self.threshold,  g_neg2 - self.threshold, g_neg3 - self.threshold, g_neg4 - self.threshold, g_neg5 - self.threshold, g_neg6 - self.threshold, g_neg7 - self.threshold, g_neg8 - self.threshold, g_neg9 - self.threshold]))).mean()
             loss = torch.log(1 + torch.exp(torch.cat([-g_pos + self.threshold, ((g_neg1 - self.threshold) + (g_neg2 - self.threshold) + (g_neg3 - self.threshold) + (g_neg4 - self.threshold) + (g_neg5 - self.threshold) + (g_neg6 - self.threshold) + (g_neg7 - self.threshold) + (g_neg8 - self.threshold) + (g_neg9 - self.threshold))/9]))).mean()
             self.opt.zero_grad()
             # this backward just compute the derivative and hence
             # is not considered backpropagation.
             loss.backward()
             self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg1).detach(), self.forward(x_neg2).detach(), self.forward(x_neg3).detach(), self.forward(x_neg4).detach(), self.forward(x_neg5).detach(), self.forward(x_neg6).detach(), self.forward(x_neg7).detach(), self.forward(x_neg8).detach(), self.forward(x_neg9).detach() #Returns a new Tensor, detached from the current graph.


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

    def train(self, x_pos, x_neg1,  x_neg2, x_neg3, x_neg4, x_neg5, x_neg6, x_neg7, x_neg8, x_neg9):
        h_pos, h_neg1, h_neg2, h_neg3, h_neg4, h_neg5, h_neg6, h_neg7, h_neg8, h_neg9 = x_pos, x_neg1, x_neg2, x_neg3, x_neg4, x_neg5, x_neg6, x_neg7, x_neg8, x_neg9
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg1, h_neg2, h_neg3, h_neg4, h_neg5, h_neg6, h_neg7, h_neg8, h_neg9 = layer.train(h_pos, h_neg1, h_neg2, h_neg3, h_neg4, h_neg5, h_neg6, h_neg7, h_neg8, h_neg9)

# 在每个layer里面进行训练 实际训练模式就是Layer的性质

if __name__ == "__main__":
    torch.manual_seed(123)

    net = Net([784, 2000, 2000, 2000, 2000])
    net.train(x_pos, x_neg1, x_neg2, x_neg3, x_neg4, x_neg5, x_neg6, x_neg7, x_neg8, x_neg9)

print('train score:', 100*net.predict(x[0:10000]).eq(y[0:10000]).float().mean().item(),"%")
#print('train score:', 100*net.predict(x[10000: 20000]).eq(y[10000: 20000]).float().mean().item(),"%")
#print('train score:', 100*net.predict(x[10000: 20000]).eq(y[10000: 20000]).float().mean().item(),"%")
#print('train score:', 100*net.predict(x[20000: 30000]).eq(y[20000: 30000]).float().mean().item(),"%")
#print('train score:', 100*net.predict(x[30000: 40000]).eq(y[30000: 40000]).float().mean().item(),"%")
#print('train score:', 100*net.predict(x[40000: 50000]).eq(y[40000: 50000]).float().mean().item(),"%")
#print('train score:', 100*net.predict(x[50000: 60000]).eq(y[50000: 60000]).float().mean().item(),"%")
#print('test score:', 100*net.predict(x_te[0: 2000]).eq(y_te[0: 2000]).float().mean().item(),"%")
print('test score:', 100*net.predict(x_te).eq(y_te).float().mean().item(),"%")
#print('test score:', 100*net.predict(x_te[4000: 6000]).eq(y_te[4000: 6000]).float().mean().item(),"%")
#print('test score:', 100*net.predict(x_te[6000: 8000]).eq(y_te[6000: 8000]).float().mean().item(),"%")
#print('test score:', 100*net.predict(x_te[8000: 10000]).eq(y_te[8000: 10000]).float().mean().item(),"%")
#print('final accuracy', (100*net.predict(x_te[0: 2000]).eq(y_te[0: 2000]).float().mean().item()+100*net.predict(x_te[2000: 4000]).eq(y_te[2000: 4000]).float().mean().item()+100*net.predict(x_te[4000: 6000]).eq(y_te[4000: 6000]).float().mean().item()+100*net.predict(x_te[6000: 8000]).eq(y_te[6000: 8000]).float().mean().item()+100*net.predict(x_te[8000: 10000]).eq(y_te[8000: 10000]).float().mean().item())/5, '%')   


