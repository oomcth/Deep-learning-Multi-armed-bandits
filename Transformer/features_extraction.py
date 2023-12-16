import torch
from torch import Tensor
import torch.nn as nn
from torch.nn import Dropout, Linear
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torch.nn.modules.transformer import MultiheadAttention
from torch.nn.init import xavier_uniform_
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from numpy import mean
import matplotlib.pyplot as plt
import random
from tqdm import tqdm

# Globals
savedir = "Desktop/"


# data preparation
X_train = list(torch.unbind(torch.cat([torch.load(savedir + 'X_train.pt'),
                                       torch.load(savedir + 'X_test.pt'),
                                       torch.load(savedir + 'X_valid.pt')])))
X_train = torch.stack(X_train)


y_train = list(torch.unbind(torch.cat([torch.load(savedir + 'Y_train.pt'),
                                       torch.load(savedir + 'Y_test.pt'),
                                       torch.load(savedir + 'Y_valid.pt')])))
y_train = torch.stack(y_train).unsqueeze(1)
y_train = torch.cat([X_train[:, 1:, :-1], y_train[:X_train.size()[0]]], dim=1)

positions = torch.argmax(X_train[:, :, :4], dim=2)
values = X_train[:, :, 4]
X_train = torch.stack([positions, values], dim=2).long()


batch_size = 64
train_loader = DataLoader([[X_train[i], y_train[i]] for i in range(len(X_train))],
                          batch_size=batch_size, shuffle=True)


class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model: int = 5, dim_emb_choice: int = 10,
                 dim_emb_val: int = 4, dim_feedforward: int = 20,
                 dropout: float = 0, bias: bool = True, device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        self.extracting_features = False
        self.d_model = d_model
        self.dim_emb = dim_emb_choice + dim_emb_val
        self.dim_emb_choice = dim_emb_choice
        self.pos_emb = nn.Embedding(4, self.dim_emb)
        self.pos = [torch.tensor(i) for i in range(4)]
        self.choice_emb = nn.Embedding(4, dim_emb_choice)
        self.val_emb = nn.Embedding(100, dim_emb_val)

        self.causal_attention = torch.zeros((d_model, d_model),
                                            dtype=torch.float)
        for i in range(d_model):
            for j in range(i + 1, d_model):
                self.causal_attention[i, j] = float('-inf')

        # attention head
        self.self_attn = MultiheadAttention(self.dim_emb, 1, dropout=dropout,
                                            bias=bias, **factory_kwargs)

        # Implementation of Feedforward model
        self.linear1 = Linear(self.dim_emb, dim_feedforward, bias=bias,
                              **factory_kwargs)

        self.linear2 = Linear(dim_feedforward, self.dim_emb, bias=bias,
                              **factory_kwargs)

        self.linear3 = Linear(self.dim_emb, 4)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        self.activation = F.relu

        for p in self.parameters():
            if p.dim() > 1:
                p = xavier_uniform_(p)

    def forward(
        self,
        x: Tensor,
    ):
        pos = torch.stack([self.pos_emb(position) for position in self.pos])
        pos = pos.unsqueeze(0).expand(x.size()[0], 4, self.dim_emb)
        choice = self.choice_emb(x[:, :, :-1]).squeeze(dim=2)
        val = self.val_emb(x[:, :, -1])

        x = torch.cat([choice, val], dim=2)
        x = x + pos
        x = x + self._sa_block(x)
        x = x + self._ff_block(x)
        x = self.linear3(x)

        if self.extracting_features:
            Autoencoder_loss = []
            return x, Autoencoder_loss
        return x, None

    # self-attention block
    def _sa_block(self, x: Tensor) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=self.causal_attention,
                           is_causal=True,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout2(F.relu(self.linear1(x))))
        return self.dropout3(x)


# creating model
model = TransformerDecoderLayer()

# training hyperparameters
epochs = 50
lr = 0.005
optimizer = Adam(model.parameters(), lr=lr)
criterion = CrossEntropyLoss()


def train():
    losses = []

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
    torch.save(model, "model_TransfoDecoder.pth")

    plt.plot(losses)
    plt.show()


train()

model = TransformerDecoderLayer()
model = torch.load("model_TransfoDecoder.pth")
