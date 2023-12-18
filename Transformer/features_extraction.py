import torch
from torch import unbind, cat, stack
from torch import argmax
from torch.nn import CrossEntropyLoss

from torch.utils.data import DataLoader
from torch.optim import Adam

import pickle

from model import TransformerDecoderLayer
from utils import train, extract, accuray

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


# Creating the batch
batch_size = 64
train_loader = DataLoader([[X_train[i], y_train[i]]
                           for i in range(len(X_train))],
                          batch_size=batch_size, shuffle=True)


# creating model
model = TransformerDecoderLayer()


# training hyperparameters
epochs = 50
lr = 0.002
optimizer = Adam(model.parameters(), lr=lr)
criterion = CrossEntropyLoss()


# pretraining (Pretraining is not optimal it helps with loss,
# but decreases accuracy.)
train(model, epochs, criterion, optimizer, train_loader, trainAll=True)
# training
train(model, epochs, criterion, optimizer, train_loader, trainAll=False)


# load trained model
model = TransformerDecoderLayer()
model = torch.load("model_TransfoDecoder.pth")


# feature extracting hyperparameters
epochs = 20
lr = 0.001
# parameters to optimize
to_optimize = (list(model.encoder.parameters()) +
               list(model.decoder.parameters()))
optimizer = Adam(to_optimize, lr=lr)

# feature extracting/training
extract(model, epochs, optimizer, train_loader)

# print accuracy
print(accuray(model, X_train, y_train))
