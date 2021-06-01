import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import numpy as np

class LM(nn.Module):
    def __init__(self, input_size, embedding_size, hidden_size, output_size, bidirectional=False):
        super(LM, self).__init__()
        # Size of the input vocab
        self.input_size = input_size
        # dims of each embedding
        self.embedding_size = embedding_size
        # dims of the hidden state
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.embedding = nn.Embedding(self.input_size, self.embedding_size)
        # Initial hidden state whose parameters are shared across all examples
        self.h0 = nn.Parameter(torch.rand(self.hidden_size))
        self.c0 = nn.Parameter(torch.rand(self.hidden_size))
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size,num_layers = 2, bidirectional=bidirectional)
        self.classifier = nn.Linear(hidden_size, output_size)
        self.log_softmax = nn.LogSoftmax(dim=2)

    def forward(self, batch, last_hidden):
        """
        Lookup the embedded representation
        of the inputs
        """
        embedded = self.embedding(batch).view(
            1, batch.size(0), self.hidden_size
        )
        # Input: (seq_len, batch_len, embedding_size)
        # Output: output, (hn, cn)
        # TODO: What about bidirectinal case?
        output, hidden = self.lstm(embedded, last_hidden)

        output = self.log_softmax(self.classifier(output))
        #print((torch.exp(output)))
        return output, hidden

    def init_hidden(self, batch_size):
        """Initialize the hidden state to pass to the LSTM

        Note we learn the initial state h0 as a parameter of the model."""
        # (seq_len x batch_size x hidden_size, seq_len x batch_size x hidden_size)
        return self.h0.repeat(2, batch_size, 1), self.c0.repeat(2, batch_size, 1)

    @staticmethod
    def get_loss_func(PAD_index):
        # This should return the function itself
        return torch.nn.NLLLoss(ignore_index=PAD_index)
