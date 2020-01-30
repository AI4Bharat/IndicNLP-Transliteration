import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, RNN_type, vocab_size, enc_units, embedding_dim,
                 bidirectional=False, num_layers=1):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.num_directions = 2 if bidirectional else 1
        self.inp2emb = nn.Linear(self.vocab_size, embedding_dim)
        self.rnn = RNN_type(embedding_dim, self.enc_units//self.num_directions,
                            bidirectional=bidirectional, num_layers=num_layers)
        
    def forward(self, x, lens):
        '''
        Arguments:
            x:      Padded input (batch_size, max_length, vocab_size)
            lens:   Actual lengths of each sequence (batch_size)
        Returns:
            output: RNN's output (max_length, batch_size, enc_units)
            hidden: RNN's hidden (num_layers*num_directions, batch_size, enc_units/num_directions)
        '''
        
        x = self.inp2emb(x)
        # x transformed = (max_len X batch_size X embedding_dim)
        x = pack_padded_sequence(x, lens) # unpad
        
        # returns hidden state of all timesteps as well as hidden state at last timestep
        output, self.hidden = self.rnn(x) 
        
        # pad the sequence to the max length in the batch
        output, _ = pad_packed_sequence(output)
        
        return output, self.hidden