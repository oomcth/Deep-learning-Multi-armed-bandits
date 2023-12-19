import torch
from torch import unbind, cat, stack
from torch import argmax

from torch.utils.data import DataLoader

import pickle

from model import TransformerDecoderLayer
import matplotlib.pyplot as plt


# Globals
savedir = ""

# Loading Data
with open(savedir + "train_test_valid_data_dict.pkl", 'rb') as f:

    data = pickle.load(f)

    train_, test_, valid_ = data['train'], data['test'], data['valid']
    X_train = list(unbind(cat([train_[0], test_[0], valid_[0]])))
    y_train = list(unbind(cat([train_[1], test_[1], valid_[1]]).float()))


# preprocessing of X_train
X_train = stack(X_train)
positions = argmax(X_train[:, :, :4], dim=2)
values = X_train[:, :, 4]
X_train = stack([positions, values], dim=2).long()


# plt.imshow(features[0, :, :].detach().numpy(), cmap='viridis')
# plt.colorbar()
# plt.title("Tensor Plot")
# plt.show()

# Creating the batch
batch_size = 1
train_loader = DataLoader([[X_train[i], y_train[i]]
                           for i in range(len(X_train))],
                          batch_size=batch_size, shuffle=True)

# load trained model
model = TransformerDecoderLayer()
model = torch.load("model_TransfoDecoder.pth")
model.train()

for batch in train_loader:
    (x, y) = batch
    _, _ = model(x)
    print("feature :", model.features)
    print("x = ", x)
    print("y = ", y)
    plt.imshow(model.features.detach().numpy(), cmap='viridis')
    plt.colorbar()
    plt.title("Tensor Plot")
    plt.show()
