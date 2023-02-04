# %%
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
import torch.nn.functional as F
from torchvision.transforms import Compose, ToTensor, Normalize, Lambda # lambda self defined 


# %%
train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=100, shuffle=True)

test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.MNIST('./files/', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=100, shuffle=True)


# %%
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train.shape, x_test.shape

# %%
random_ex = np.random.rand(28,28)
random_ex = random_ex *0
mask_ex = random_ex
mask_ex[1, 13:15] = 255
mask_ex[1, 23:25] = 255
mask_ex[2, 10:26] = 255
mask_ex[3, 9:27] = 255
mask_ex[4, 8:27] = 255
mask_ex[5, 5:26] = 255
mask_ex[6, 4:26] = 255
mask_ex[7, 4:18] = 255
mask_ex[7, 23:26] = 255
mask_ex[8, 3:17] = 255
mask_ex[8, 24:26] = 255
mask_ex[9, 3:16] = 255
mask_ex[9, 24:26] = 255
mask_ex[10, 3:16] = 255
mask_ex[10, 25:27] = 255
mask_ex[11, 3:15] = 255
mask_ex[11, 25:27] = 255
mask_ex[12, 3:12] = 255
mask_ex[12, 24:27] = 255
mask_ex[13, 3:9] = 255
mask_ex[13, 24:27] = 255
mask_ex[14, 3:8] = 255
mask_ex[14, 24:27] = 255
mask_ex[15, 3:7] = 255
mask_ex[15, 24:27] = 255
mask_ex[16, 3:8] = 255
mask_ex[16, 24:27] = 255
mask_ex[17, 3:9] = 255
mask_ex[17, 18:22] = 255
mask_ex[17, 24:26] = 255
mask_ex[18, 4:10] = 255
mask_ex[18, 18:22] = 255
mask_ex[18, 23:26] = 255
mask_ex[19, 5:10] = 255
mask_ex[19, 23:26] = 255
mask_ex[20, 6:10] = 255
mask_ex[20, 23:27] = 255
mask_ex[21, 7:10] = 255
mask_ex[21, 23:27] = 255
mask_ex[22, 7:10] = 255
mask_ex[22, 23:27] = 255
mask_ex[23, 7:9] = 255
mask_ex[23, 23:27] = 255
mask_ex[24,11:13] = 255
mask_ex[24, 23:27] = 255
mask_ex[25, 24:27] = 255
mask_ex[26, 25] = 255
mask_ex = mask_ex/255
plt.imshow(mask_ex, cmap='gray')

# %%
random_ex2 = np.random.rand(28,28)
random_ex2[:, :28] = 255
mask_ex2 = random_ex2
mask_ex2[1, 13:15] = 0
mask_ex2[1, 23:25] = 0
mask_ex2[2, 10:26] = 0
mask_ex2[3, 9:27] = 0
mask_ex2[4, 8:27] = 0
mask_ex2[5, 5:26] = 0
mask_ex2[6, 4:26] = 0
mask_ex2[7, 4:18] = 0
mask_ex2[7, 23:26] = 0
mask_ex2[8, 3:17] = 0
mask_ex2[8, 24:26] = 0
mask_ex2[9, 3:16] = 0
mask_ex2[9, 24:26] = 0
mask_ex2[10, 3:16] = 0
mask_ex2[10, 25:27] = 0
mask_ex2[11, 3:15] = 0
mask_ex2[11, 25:27] = 0
mask_ex2[12, 3:12] = 0
mask_ex2[12, 24:27] = 0
mask_ex2[13, 3:9] = 0
mask_ex2[13, 24:27] = 0
mask_ex2[14, 3:8] = 0
mask_ex2[14, 24:27] = 0
mask_ex2[15, 3:7] = 0
mask_ex2[15, 24:27] = 0
mask_ex2[16, 3:8] = 0
mask_ex2[16, 24:27] = 0
mask_ex2[17, 3:9] = 0
mask_ex2[17, 18:22] = 0
mask_ex2[17, 24:26] = 0
mask_ex2[18, 4:10] = 0
mask_ex2[18, 18:22] = 0
mask_ex2[18, 23:26] = 0
mask_ex2[19, 5:10] = 0
mask_ex2[19, 23:26] = 0
mask_ex2[20, 6:10] = 0
mask_ex2[20, 23:27] = 0
mask_ex2[21, 7:10] = 0
mask_ex2[21, 23:27] = 0
mask_ex2[22, 7:10] = 0
mask_ex2[22, 23:27] = 0
mask_ex2[23, 7:9] = 0
mask_ex2[23, 23:27] = 0
mask_ex2[24,11:13] = 0
mask_ex2[24, 23:27] = 0
mask_ex2[25, 24:27] = 0
mask_ex2[26, 25] = 0
mask_ex2 = mask_ex2/255
plt.imshow(mask_ex2, cmap='gray')


# %%
a = x_train[15] * mask_ex + x_train[18] * mask_ex2
plt.imshow(a, cmap='gray')

# %%
x_train_rnd = np.zeros(shape=(60000,28,28))

for i in range (x_train.shape[0]):
    rnd = np.random.randint(x_train.shape[0])
    x_train_rnd[i] = x_train[rnd]
    

# %%
x_neg = x_train * mask_ex + x_train_rnd * mask_ex2   
plt.imshow(x_neg[1], cmap='gray')
x_pos = x_train

# %%
device = 'cuda:2'

# %%

x_pos = (x_pos-33.31002426147461 )/78.56748962402344
x_neg = (x_neg-33.31002426147461 )/78.56748962402344
pos = x_pos.reshape(x_pos.shape[0], -1)
neg = x_neg.reshape(x_neg.shape[0], -1)
x_pos = torch.tensor(pos, dtype=torch.float)
x_neg = torch.tensor(neg, dtype=torch.float)

x_test = (x_test-33.31002426147461 )/78.56748962402344
x_test = x_test.reshape(x_test.shape[0], -1)
y_test = y_test.reshape(y_test.shape[0])
x_te = torch.tensor(x_test, dtype=torch.float)
y_te = torch.tensor(y_test, dtype=torch.float)

x_train = (x_train-33.31002426147461 )/78.56748962402344
x_train = x_train.reshape(x_train.shape[0], -1)
y_train = y_train.reshape(y_train.shape[0])
x = torch.tensor(x_train, dtype=torch.float)
y = torch.tensor(y_train, dtype=torch.float)

x_pos, x_neg = x_pos.cuda(device), x_neg.cuda(device)
x_te, y_te = x_te.cuda(device), y_te.cuda(device)
x, y = x.cuda(device), y.cuda(device)

print(x.shape, y.shape, x.dtype)

# %%
class Layer(nn.Linear):
    def __init__(self, in_features, out_features,
                 bias=True, device=None, dtype=None):
        super().__init__(in_features, out_features, bias, device, dtype)
        self.relu = torch.nn.ReLU()
        self.opt = Adam(self.parameters(), lr=0.06)
        self.threshold = 2
        self.num_epochs = 100

    def forward(self, x):
        x_direction = x / (x.norm(2, 1, keepdim=True) + 0.02)
        normlized_activity = self.relu(torch.mm(x_direction, self.weight.T) + self.bias.unsqueeze(0))
        return normlized_activity

    def train_layer(self, x_pos, x_neg):
        for i in tqdm(range(self.num_epochs)):
          for b in range (60):
            g_pos = self.forward(x_pos[b*1000: (b+1)*1000]).pow(2).mean(1)
            g_neg = self.forward(x_neg[b*1000: (b+1)*1000]).pow(2).mean(1)
            #g_pos = self.forward(x_pos).pow(2).mean(1)
            #g_neg = self.forward(x_neg).pow(2).mean(1)
            # The following loss pushes pos (neg) samples to
            # values larger (smaller) than the self.threshold.
            loss = torch.log(1 + torch.exp(torch.cat([
                -g_pos + self.threshold,
                g_neg - self.threshold]))).mean()
            self.opt.zero_grad()
            # this backward just compute the derivative and hence
            # is not considered backpropagation.
            loss.backward()
            self.opt.step()
        return self.forward(x_pos).detach(), self.forward(x_neg).detach()

# %%

class FFNet(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.hlayer1 = Layer(784, 2000)
        self.hlayer2 = Layer(2000, 2000)
        self.hlayer3 = Layer(2000, 2000)
        self.hlayer4 = Layer(2000, 2000)
        self.layers = []
        self.layers = nn.Sequential(self.hlayer1.cuda(device), self.hlayer2.cuda(device), self.hlayer3.cuda(device), self.hlayer4.cuda(device))

    def train_ffnet(self, x_pos, x_neg):
        h_pos, h_neg = x_pos, x_neg
        for i, layer in enumerate(self.layers):
            print('training layer', i, '...')
            h_pos, h_neg = layer.train_layer(h_pos, h_neg)


# %%
(x_train2, y_train2), (x_test2, y_test2) = tf.keras.datasets.mnist.load_data()

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

pos2 = np.copy(x_train2)
neg2 = np.copy(x_train2)
edit_data(pos2, y_train2)
edit_data(neg2, random_label(y_train2))


pos2 = (pos2-33.31002426147461 )/78.56748962402344
neg2 = (neg2-33.31002426147461 )/78.56748962402344
pos2 = pos2.reshape(pos2.shape[0], -1)
neg2 = neg2.reshape(neg2.shape[0], -1)
x_pos2 = torch.tensor(pos2, dtype=torch.float)
x_neg2 = torch.tensor(neg2, dtype=torch.float)

x_pos2, x_neg2 = x_pos.cuda(device), x_neg.cuda(device)


# %%
net = FFNet()
net.train_ffnet(x_pos2, x_neg2)

def freeze(model):
    for param in model.parameters():
        param.requires_grad = False

# %%
net.hlayer1.weight
net.hlayer2.weight
net.hlayer3.weight
net.hlayer4.weight

# %%

h1 = net.hlayer1(x)
h2 = net.hlayer2(h1)
h3 = net.hlayer3(h2)
h4 = net.hlayer4(h3)

h1.shape, h2.shape, h3.shape, h4.shape


# %%
h5= torch.cat((h2,h3,h4), 1)
print(h5.shape, h5,  h5.size(0))
h6= h5.view(h5.size(0), -1)
print(h6.shape, h6)

# %%
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        
        self.hlayer1 = net.hlayer1
        self.hlayer2 = net.hlayer2
        self.hlayer3 = net.hlayer3
        self.hlayer4 = net.hlayer4
        freeze(self)     
        self.fc = nn.Linear(6000, 10).cuda(device)
        
    def forward(self, x):
        x = torch.flatten(x, 1)
        n1= self.hlayer1(x)
        n2= self.hlayer2(n1)
        n3= self.hlayer3(n2)
        n4= self.hlayer4(n3)
        n2 = n2 / (n2.norm(2, 1, keepdim=True) + 0.01)
        n3 = n3 / (n2.norm(2, 1, keepdim=True) + 0.01)
        n4 = n4 / (n2.norm(2, 1, keepdim=True) + 0.01)

        n5= torch.cat((n2,n3,n4), 1)
        n5 = n5.view(n5.size(0), -1)
        
        output = self.fc(n5)
        
        return output 
        
network = Net()
print(network.fc)


# %%
criterion = torch.nn.CrossEntropyLoss()
#optimizer = torch.optim.Adam(network.parameters(), lr = 0.01)
optimizer = torch.optim.Adam(filter(lambda p : p.requires_grad, network.parameters()), lr = 0.01)

# %%
output = network(x)
output.shape

# %%
n_epochs = 1
batch_size_train = 100
batch_size_test = 1000

# %%
train_losses = []
train_counter = []
test_losses = []
test_counter = [i*len(train_loader.dataset) for i in range(n_epochs + 1)]


# %%
def train(epoch):
  network.train()
  for batch_idx, (data, target) in enumerate(train_loader):
    data =data.cuda(device)
    target = target.cuda(device)
    optimizer.zero_grad()
    output = network(data)
    loss = criterion(output, target)
    loss.backward()
    if batch_idx % 10 == 0:
      print("batch_idx = ", batch_idx)
      print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        epoch, batch_idx * len(data), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item()))
      
      train_losses.append(loss.item())
      train_counter.append(
        (batch_idx*batch_size_train) + ((epoch-1)*len(train_loader.dataset)))
      torch.save(network.state_dict(), './results/model.pth')
      torch.save(optimizer.state_dict(), './results/optimizer.pth')
         

# %%
for i in range (n_epochs):
    train(i)

# %%
net.hlayer1.weight
net.hlayer2.weight
net.hlayer3.weight
net.hlayer4.weight

# %%
def test():
  network.eval()
  test_loss = 0
  correct = 0
  with torch.no_grad():
    for data, target in test_loader:
      data =data.cuda(device)
      target = target.cuda(device)
      #print(data.dtype, target.shape)
      output = network(data)
      test_loss += criterion(output, target).item()
      pred = output.data.max(1, keepdim=True)[1]
      correct += pred.eq(target.data.view_as(pred)).sum()
  test_loss /= len(test_loader.dataset)
  test_losses.append(test_loss)
  print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
    test_loss, correct, len(test_loader.dataset),
    100. * correct / len(test_loader.dataset)))


# %%
test()

