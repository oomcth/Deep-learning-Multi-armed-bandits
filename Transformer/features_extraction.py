import torch
from torch import Tensor, unbind, load, cat, stack
from torch import argmax, tensor, zeros, save, zeros_like
from torch.nn import Module, Embedding, Sequential
from torch.nn import Dropout, Linear, CrossEntropyLoss, L1Loss, MSELoss, ReLU
import torch.nn.functional as F
from torch.nn.modules.transformer import MultiheadAttention
from torch.nn.init import xavier_uniform_
from torch.utils.data import DataLoader
from torch.optim import Adam, SparseAdam
from itertools import chain

from numpy import mean
import matplotlib.pyplot as plt
from tqdm import tqdm

# Globals
savedir = "Desktop/"


# data preparation
X_train = list(unbind(cat([load(savedir + 'X_train.pt'),
                           load(savedir + 'X_test.pt'),
                           load(savedir + 'X_valid.pt')])))
X_train = stack(X_train)


y_train = list(unbind(cat([load(savedir + 'Y_train.pt'),
                           load(savedir + 'Y_test.pt'),
                           load(savedir + 'Y_valid.pt')])))
y_train = stack(y_train).unsqueeze(1)
y_train = cat([X_train[:, 1:, :-1], y_train[:X_train.size()[0]]], dim=1)

positions = argmax(X_train[:, :, :4], dim=2)
values = X_train[:, :, 4]
X_train = stack([positions, values], dim=2).long()

batch_size = 64
train_loader = DataLoader([[X_train[i], y_train[i]]
                           for i in range(len(X_train))],
                          batch_size=batch_size, shuffle=True)


class TransformerDecoderLayer(Module):

    def __init__(self, d_model: int = 4, dim_emb_choice: int = 10,
                 dim_emb_val: int = 4, dim_feedforward: int = 2,
                 num_features: int = 5, dropout: float = 0,
                 bias: bool = True, device=None, dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.dim_feedforward = dim_feedforward
        self.num_features = num_features
        self.d_model = d_model
        self.dim_emb = dim_emb_choice + dim_emb_val
        self.dim_emb_choice = dim_emb_choice
        self.pos_emb = Embedding(4, self.dim_emb)
        self.pos = [tensor(i) for i in range(4)]
        self.choice_emb = Embedding(4, dim_emb_choice)
        self.val_emb = Embedding(100, dim_emb_val)

        self.causal_attention = zeros((d_model, d_model),
                                      dtype=torch.float)
        for i in range(d_model):
            for j in range(i + 1, d_model):
                self.causal_attention[i, j] = float('-inf')

        # attention head
        self.self_attn = MultiheadAttention(self.dim_emb, 1, dropout=dropout,
                                            bias=bias, **factory_kwargs)

        # Implementation of Feedforward model
        self.linear1 = Linear(self.dim_emb, dim_feedforward*10, bias=bias,
                              **factory_kwargs)

        self.linear2 = Linear(dim_feedforward*10, self.dim_emb, bias=bias,
                              **factory_kwargs)

        self.linear3 = Linear(self.dim_emb, 4)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = F.relu

        for p in self.parameters():
            if p.dim() > 1:
                p = xavier_uniform_(p)

        self.extracting_features = False
        self.L1Loss = L1Loss()
        self.L2Loss = MSELoss()
        enc = [(Linear(i*10, (i+1)*10), ReLU())
               for i in range(self.dim_feedforward, num_features, 1)]
        enc = list(chain(*enc))
        dec = [(Linear(i*10, (i-1)*10), ReLU())
               for i in range(num_features, self.dim_feedforward, -1)]
        dec = list(chain(*dec))
        self.encoder = Sequential(*enc)
        self.decoder = Sequential(*dec)

    def forward(
        self,
        x: Tensor,
    ):
        pos = stack([self.pos_emb(position) for position in self.pos])
        pos = pos.unsqueeze(0).expand(x.size()[0], 4, self.dim_emb)
        choice = self.choice_emb(x[:, :, :-1]).squeeze(dim=2)
        val = self.val_emb(x[:, :, -1])

        x = cat([choice, val], dim=2)
        x = x + pos
        x = x + self._sa_block(x)
        ff_block, loss = self._ff_block(x)
        x = x + ff_block
        x = self.linear3(x)

        return x, loss

    # self-attention block
    def _sa_block(self, x: Tensor) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=self.causal_attention,
                           is_causal=True,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:

        activation = F.relu(self.linear1(x))
        x = self.linear2(self.dropout2(activation))

        if self.extracting_features:
            features = self.encoder(activation)
            out = self.decoder(features)

            loss = (self.L2Loss(out, activation) +
                    4 * self.L1Loss(features, zeros_like(features)))
            return self.dropout3(x), loss
        return self.dropout3(x), None


# creating model
model = TransformerDecoderLayer()

# training hyperparameters
epochs = 5
lr = 0.005
optimizer = Adam(model.parameters(), lr=lr)
criterion = CrossEntropyLoss()


def train() -> None:
    losses = []

    model.extracting_features = False
    print("training started")
    for epoch in tqdm(range(epochs)):
        running_loss = []
        for batch in train_loader:
            (x, y) = batch
            outputs, _ = model(x)
            loss = criterion(outputs, y)

            optimizer.zero_grad()
            loss.backward()
            running_loss.append(loss.item())
            optimizer.step()
        losses.append(mean(running_loss))
        running_loss = []
    save(model, "model_TransfoDecoder.pth")

    plt.plot(losses)
    plt.show()


train()

model = TransformerDecoderLayer()
model = torch.load("model_TransfoDecoder.pth")

# feature extracting hyperparameters
epochs = 5
lr = 0.005
to_optimize = list(model.encoder.parameters()) + list(model.decoder.parameters())
optimizer = Adam(to_optimize, lr=lr)


def extract() -> None:
    losses = []

    model.extracting_features = True
    print("extraction started")
    for epoch in tqdm(range(epochs)):
        running_loss = []
        for batch in train_loader:
            (x, _) = batch
            _, loss = model(x)

            optimizer.zero_grad()
            loss.backward()
            running_loss.append(loss.item())
            optimizer.step()
        losses.append(mean(running_loss))
        running_loss = []
    save(model, "model_TransfoDecoder.pth")

    plt.plot(losses)
    plt.show()


extract()
