import sys
import time
sys.path.append('./')
import mindtorch as torch
from mindtorch import nn, optim
from mindspore.dataset import vision, transforms
from mindspore.dataset import MnistDataset


# ## 处理数据集
# Download data from open datasets
from download import download

url = "https://mindspore-website.obs.cn-north-4.myhuaweicloud.com/" \
      "notebook/datasets/MNIST_Data.zip"
path = download(url, "./", kind="zip", replace=False)


train_dataset = MnistDataset('MNIST_Data/train')
test_dataset = MnistDataset('MNIST_Data/test')

def datapipe(dataset, batch_size):
    image_transforms = [
        vision.Rescale(1.0 / 255.0, 0),
        vision.Normalize(mean=(0.1307,), std=(0.3081,)),
        vision.HWC2CHW()
    ]
    label_transform = transforms.TypeCast(torch.int64)

    dataset = dataset.map(image_transforms, 'image')
    dataset = dataset.map(label_transform, 'label')
    dataset = dataset.batch(batch_size)
    return dataset

# Map vision transforms and batch dataset
train_dataset = datapipe(train_dataset, 64)
test_dataset = datapipe(test_dataset, 64)


import torch
from torch import nn, optim
# ## 网络构建
# Define model
class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.dense_relu_sequential = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.dense_relu_sequential(x)
        return logits

model = Network().cuda()
print(model)

# ## 模型训练

# Instantiate loss function and optimizer
loss_fn = nn.CrossEntropyLoss().cuda()
optimizer = optim.SGD(model.parameters(), 1e-2)


def train(model, dataset):
    size = dataset.get_dataset_size()
    model.train()
    for batch, (data, label) in enumerate(dataset.create_tuple_iterator(output_numpy=True)):
        data = torch.tensor(data).cuda()
        label = torch.tensor(label).cuda()
        s = time.time()
        optimizer.zero_grad()
        logits = model(data)
        loss = loss_fn(logits, label)
        loss.backward()
        t = time.time()
        optimizer.step()
        if batch % 100 == 0:
            loss, current = loss.detach().cpu().numpy(), batch
            print(f"loss: {loss:>7f}  [{current:>3d}/{size:>3d}]")
            print(t - s)

# 除训练外，我们定义测试函数，用来评估模型的性能。

# In[12]:


def test(model, dataset, loss_fn):
    num_batches = dataset.get_dataset_size()
    model.eval()
    total, test_loss, correct = 0, 0, 0
    for data, label in dataset.create_tuple_iterator(output_numpy=True):
        data = torch.tensor(data).cuda()
        label = torch.tensor(label).cuda()
        pred = model(data)
        total += len(data)
        test_loss += loss_fn(pred, label)
        correct += (pred.argmax(1) == label).sum()
    test_loss /= num_batches
    correct = correct / total
    print(f"Test: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


# 训练过程需多次迭代数据集，一次完整的迭代称为一轮（epoch）。在每一轮，遍历训练集进行训练，结束后使用测试集进行预测。打印每一轮的loss值和预测准确率（Accuracy），可以看到loss在不断下降，Accuracy在不断提高。

# In[13]:

import cProfile

cProfile.run('train(model, train_dataset)')

epochs = 3
start = time.time()
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(model, train_dataset)
    test(model, test_dataset, loss_fn)
print("Done!")
end = time.time()
print('total: ', end - start)

