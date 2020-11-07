import torch.nn as nn
import torch 

def create_emb_layer(weights_matrix, non_trainable=False):
    num_embeddings, embedding_dim = weights_matrix.shape
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': torch.from_numpy(weights_matrix).long()})
    if non_trainable:
        emb_layer.weight.requires_grad = False
    return emb_layer, num_embeddings, embedding_dim 

class LSTM_NN(nn.Module):
    
    def __init__(self,weights_matrix, output_size, hidden_size, n_layers, drop_prob=0.5):
        super(LSTM_NN, self).__init__()
        self.output_size = output_size
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        
        self.embed, num_embeddings, embedding_dim = create_emb_layer(weights_matrix, True)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, n_layers, dropout=drop_prob, batch_first=True)
        self.dropout = nn.Dropout(drop_prob)
        self.l1 = nn.Linear(hidden_size, output_size)
        self.prob = nn.Sigmoid()

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_size).zero_())  
        return hidden

    def forward(self,x, hidden):
        batch_size = x.size(0)
        x = x.long()
        embeds = self.embed(x)
        lstm_out, hidden = self.lstm(embeds, hidden)
        lstm_out=lstm_out.contiguous().view(-1, self.hidden_size)
        
        out = self.dropout(lstm_out)
        out_l1 = self.l1(out)
        sig_out = self.prob(out_l1)

        sig_out = sig_out.view(batch_size, -1)
        sig_out = sig_out[:, -1]
        
        
        return sig_out, hidden        