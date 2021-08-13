import torch

class Network(torch.nn.Module):
    def __init__(self, in_neuron, num_layers, embedding_dim, hidden_size, out_neuron, drop, **kwargs):
        super(Network, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = torch.nn.Embedding(in_neuron, embedding_dim)

        self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, num_layers=num_layers,
                                  bidirectional=True, batch_first=True,
                                  **kwargs)  # bidirectional multilayer lstm
        
        self.layer_1 = torch.nn.Linear(hidden_size*2, int(hidden_size))
        self.layer_2 = torch.nn.Linear(hidden_size, int(hidden_size/2))
        self.layer_3 = torch.nn.Linear(int(hidden_size/2), int(hidden_size/4))
        self.layer_4 = torch.nn.Linear(int(hidden_size/4), out_neuron)  # last fully connected layer

        self.relu = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(p=drop)  # drop the values by random which comes from previous layer
        
        self.batchnorm1 = torch.nn.BatchNorm1d(hidden_size)
        self.batchnorm2 = torch.nn.BatchNorm1d(int(hidden_size/2))
        self.batchnorm3 = torch.nn.BatchNorm1d(int(hidden_size/4))

    def forward(self, t):
        embedding_t = self.embedding(t)
        drop_emb = self.dropout(embedding_t)
        out, (self.hidden_state, _) = self.lstm(drop_emb)


        
        out = torch.mean(out, 1)
        out = self.layer_1(out)
        out = self.batchnorm1(out)
        out = self.relu(out)

        out = self.layer_2(out)
        out = self.batchnorm2(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.layer_3(out)
        out = self.batchnorm3(out)
        out = self.relu(out)
        out = self.dropout(out)

        out = self.layer_4(out)

        return out