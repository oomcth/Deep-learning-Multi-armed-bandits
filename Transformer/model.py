import torch
from torch import Tensor, cat, stack
from torch import tensor, zeros, zeros_like
from torch.nn import Module, Embedding, Sequential
from torch.nn import Dropout, Linear, L1Loss, MSELoss, ReLU
import torch.nn.functional as F
from torch.nn.modules.transformer import MultiheadAttention
from torch.nn.init import xavier_uniform_
from itertools import chain


class TransformerDecoderLayer(Module):

    def __init__(self, d_model: int = 4, dim_emb_choice: int = 10,
                 dim_emb_val: int = 4, dim_feedforward: int = 2,
                 num_features: int = 5, dropout: float = 0,
                 nb_head: int = 1, bias: bool = True, device=None,
                 dtype=None) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()

        """
        d_model = sequence length
        dim_emb_choice = number of dimensions in which the choice is embedded.
        dim_emb_val = number of dimensions in which the reward is embedded.
        dim_feedforward = number of neuron in the feedforward activation layer.
        num_features = the encoder will encode the feedforward activation layer
        into a vector of dim 4*num_features.
        dropout = % of chance of drop of a coordinate during training. It is
        set to 0 by default as we don't use a test set.
        nb_head = number of activation head
        bias = set if our module have a bias
        device = try to set cuda
        """

        # if true we train the transformer for a sequence [0, 1, 2, 3]
        # to give [1, 2, 3, 4]. If false we only care that 3 becomes 4.
        self.trainAll = False
        self.num_features = num_features
        self.dim_emb = dim_emb_choice + dim_emb_val
        self.pos_emb = Embedding(4, self.dim_emb)
        self.pos = [tensor(i) for i in range(4)]

        # Embedder for choice
        self.choice_emb = Embedding(4, dim_emb_choice, **factory_kwargs)
        # Embedder for reward value
        self.val_emb = Embedding(100, dim_emb_val, **factory_kwargs)

        # causal mask for multihead
        self.causal_attention = zeros((d_model, d_model),
                                      dtype=torch.float)
        for i in range(d_model):
            for j in range(i + 1, d_model):
                # set non causal values to -inf
                self.causal_attention[i, j] = float('-inf')

        # attention head
        self.self_attn = MultiheadAttention(self.dim_emb, num_heads=nb_head,
                                            dropout=dropout, bias=bias,
                                            **factory_kwargs)

        # Implementation of Feedforward model
        self.linear1 = Linear(self.dim_emb, dim_feedforward*4, bias=bias,
                              **factory_kwargs)

        self.linear2 = Linear(dim_feedforward*4, self.dim_emb, bias=bias,
                              **factory_kwargs)

        # Feedforward activation
        self.activation = F.relu

        # linear at the end of the transformer
        self.linear3 = Linear(self.dim_emb, 4, bias=bias, **factory_kwargs)

        # implement the dropouts
        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        # initialise weights
        for p in self.parameters():
            if p.dim() > 1:
                p = xavier_uniform_(p)

        # True if training the sparse autoencoder
        self.extracting_features = False
        # losses for the sparse autoencoder training
        self.L1Loss = L1Loss()
        self.L2Loss = MSELoss()

        # implement the encoder
        enc = [(Linear(i*4, (i+1)*4, **factory_kwargs),
                ReLU())
               for i in range(dim_feedforward, num_features, 1)]
        enc = list(chain(*enc))
        self.encoder = Sequential(*enc)

        # implement the decoder
        dec = [(Linear(i*4, (i-1)*4, **factory_kwargs),
                ReLU())
               for i in range(num_features, dim_feedforward, -1)]
        dec = list(chain(*dec))
        self.decoder = Sequential(*dec)

        # set everything to device
        self.to(device)

    # forward pass function
    # x mutch be batched
    def forward(self, x: Tensor):

        # embeding and positional embedding
        pos = stack([self.pos_emb(position) for position in self.pos])
        pos = pos.unsqueeze(0).expand(x.size()[0], 4, self.dim_emb)
        choice = self.choice_emb(x[:, :, :-1]).squeeze(dim=2)
        val = self.val_emb(x[:, :, -1])
        x = cat([choice, val], dim=2)
        x = x + pos

        # attention block
        x = x + self._sa_block(x)

        # feedforward block
        ff_block, loss = self._ff_block(x)

        x = x + ff_block

        # the output depends of trainAll
        if self.trainAll:
            x = self.linear3(x)[:, :, :]
        else:
            x = self.linear3(x)[:, -1, :]

        return x, loss

    # self-attention block
    def _sa_block(self, x: Tensor) -> Tensor:
        x = self.self_attn(x, x, x, attn_mask=self.causal_attention,
                           is_causal=True, need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:

        activation = self.activation(self.linear1(x))
        x = self.linear2(self.dropout2(activation))

        # if we are trainnig the autoencoder
        if self.extracting_features:
            self.features = self.encoder(activation[:, -1, :])
            out = self.decoder(self.features)

            # L2 loss for autoencoder effiscienscy and L1 for
            # enforcing sparsity

            # It's cobbled together, it's ugly, it works
            # If I just do L1 + L2, only L1 loss decreases
            if (self.L2Loss(out, activation[:, -1, :]) /
               self.L1Loss(self.features, zeros_like(self.features))) >= 10:
                loss = self.L2Loss(out, activation[:, -1, :])
            else:
                loss = (10 * self.L2Loss(out, activation[:, -1, :]) +
                        self.L1Loss(self.features, zeros_like(self.features)))
            return self.dropout3(x), loss
        return self.dropout3(x), None
