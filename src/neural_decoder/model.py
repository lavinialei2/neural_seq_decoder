import torch
from torch import nn

from .augmentations import GaussianSmoothing


class GRUDecoder(nn.Module):
    def __init__(
        self,
        neural_dim,
        n_classes,
        hidden_dim,
        layer_dim,
        nDays=24,
        dropout=0,
        device="cuda",
        strideLen=4,
        kernelLen=14,
        gaussianSmoothWidth=0,
        bidirectional=False,
        rnn_type="gru",
        post_ffn_layers=0,
        post_ffn_hidden=None,
        post_ffn_dropout=0.0,
    ):
        super(GRUDecoder, self).__init__()

        # Defining the number of layers and the nodes in each layer
        self.layer_dim = layer_dim
        self.hidden_dim = hidden_dim
        self.neural_dim = neural_dim
        self.n_classes = n_classes
        self.nDays = nDays
        self.device = device
        self.dropout = dropout
        self.strideLen = strideLen
        self.kernelLen = kernelLen
        self.gaussianSmoothWidth = gaussianSmoothWidth
        self.bidirectional = bidirectional
        self.rnn_type = rnn_type.lower()
        self.post_ffn_layers = max(0, int(post_ffn_layers))
        self.post_ffn_hidden = post_ffn_hidden if post_ffn_hidden is not None else hidden_dim
        self.post_ffn_dropout = post_ffn_dropout
        self.inputLayerNonlinearity = torch.nn.Softsign()
        self.unfolder = torch.nn.Unfold(
            (self.kernelLen, 1), dilation=1, padding=0, stride=self.strideLen
        )
        self.gaussianSmoother = GaussianSmoothing(
            neural_dim, 20, self.gaussianSmoothWidth, dim=1
        )
        self.dayWeights = torch.nn.Parameter(torch.randn(nDays, neural_dim, neural_dim))
        self.dayBias = torch.nn.Parameter(torch.zeros(nDays, 1, neural_dim))

        for x in range(nDays):
            self.dayWeights.data[x, :, :] = torch.eye(neural_dim)

        # Recurrent layers
        if self.rnn_type not in {"gru", "lstm"}:
            raise ValueError("rnn_type must be 'gru' or 'lstm'")
        rnn_cls = nn.LSTM if self.rnn_type == "lstm" else nn.GRU
        self.rnn_decoder = rnn_cls(
            (neural_dim) * self.kernelLen,
            hidden_dim,
            layer_dim,
            batch_first=True,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
        )

        for name, param in self.rnn_decoder.named_parameters():
            if "weight_hh" in name:
                nn.init.orthogonal_(param)
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)

        # Input layers
        for x in range(nDays):
            setattr(self, "inpLayer" + str(x), nn.Linear(neural_dim, neural_dim))

        for x in range(nDays):
            thisLayer = getattr(self, "inpLayer" + str(x))
            thisLayer.weight = torch.nn.Parameter(
                thisLayer.weight + torch.eye(neural_dim)
            )

        # rnn outputs
        if self.bidirectional:
            self.fc_decoder_out = nn.Linear(
                hidden_dim * 2, n_classes + 1
            )  # +1 for CTC blank
        else:
            self.fc_decoder_out = nn.Linear(hidden_dim, n_classes + 1)  # +1 for CTC blank

        norm_dim = hidden_dim * 2 if self.bidirectional else hidden_dim
        self.layer_norm = nn.LayerNorm(norm_dim)
        self.post_ffn = nn.ModuleList()
        if self.post_ffn_layers > 0:
            for _ in range(self.post_ffn_layers):
                self.post_ffn.append(
                    nn.Sequential(
                        nn.Linear(norm_dim, self.post_ffn_hidden),
                        nn.LayerNorm(self.post_ffn_hidden),
                        nn.GELU(),
                        nn.Dropout(self.post_ffn_dropout),
                        nn.Linear(self.post_ffn_hidden, norm_dim),
                        nn.LayerNorm(norm_dim),
                        nn.Dropout(self.post_ffn_dropout),
                    )
                )

    def forward(self, neuralInput, dayIdx):
        neuralInput = torch.permute(neuralInput, (0, 2, 1))
        neuralInput = self.gaussianSmoother(neuralInput)
        neuralInput = torch.permute(neuralInput, (0, 2, 1))

        # apply day layer
        dayWeights = torch.index_select(self.dayWeights, 0, dayIdx)
        transformedNeural = torch.einsum(
            "btd,bdk->btk", neuralInput, dayWeights
        ) + torch.index_select(self.dayBias, 0, dayIdx)
        transformedNeural = self.inputLayerNonlinearity(transformedNeural)

        # stride/kernel
        stridedInputs = torch.permute(
            self.unfolder(
                torch.unsqueeze(torch.permute(transformedNeural, (0, 2, 1)), 3)
            ),
            (0, 2, 1),
        )

        # apply RNN layer
        num_dir = 2 if self.bidirectional else 1
        if self.bidirectional:
            h0 = torch.zeros(
                self.layer_dim * num_dir, transformedNeural.size(0), self.hidden_dim, device=self.device
            ).requires_grad_()
        else:
            h0 = torch.zeros(
                self.layer_dim * num_dir, transformedNeural.size(0), self.hidden_dim, device=self.device
            ).requires_grad_()

        if self.rnn_type == "lstm":
            c0 = torch.zeros_like(h0)
            hid, _ = self.rnn_decoder(stridedInputs, (h0.detach(), c0.detach()))
        else:
            hid, _ = self.rnn_decoder(stridedInputs, h0.detach())
        hid = self.layer_norm(hid)
        for block in self.post_ffn:
            hid = hid + block(hid)

        # get seq
        seq_out = self.fc_decoder_out(hid)
        return seq_out
