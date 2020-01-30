import torch
import torch.nn as nn

class Decoder(nn.Module):
    def __init__(self, RNN_type, vocab_size, dec_units, enc_units, embedding_dim,
                num_layers=1):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        # FC to convert vocab_size to a size to concat with hidden
        self.out2emb = nn.Linear(self.vocab_size, self.embedding_dim)
        self.rnn = RNN_type(self.embedding_dim + self.enc_units, 
                          self.dec_units, num_layers=num_layers,
                          batch_first=True)
        
        # DecoderRNN -> FC1(emb_size) -> FC2(vocab_size)
        self.fc = nn.Sequential(
            nn.Linear(self.dec_units, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, self.vocab_size))
        
        # used for attention
        self.W1 = nn.Linear(self.enc_units, self.dec_units)
        self.W2 = nn.Linear(self.dec_units, self.dec_units)
        self.V = nn.Linear(self.dec_units, 1)
    
    def attention(self, hidden, enc_output):
        # Calculates attention using H and Encoder outputs
        # Returns context_vector that can be concat'ed to input
        
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = hidden[-1::1].permute(1, 0, 2) # Take only the last layer's hidden
        
        # score: (batch_size, max_length, hidden_size) # Bahdanaus's
        # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
        # It doesn't matter which FC we pick for each of the inputs
        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
          
        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = torch.softmax(self.V(score), dim=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)
        return context_vector, attention_weights
    
    def forward(self, x, hidden, enc_output):
        '''
        Arguments:
            x:      Decoder Input (batch_sz, 1, vocab_size)
            hidden: Decoder Hidden (num_layers, batch_sz, dec_units)
            enc_out:Encoder's Output (max_len, batch_sz, enc_units)
            
        Returns:
            out:    Decoder Output (batch_sz, vocab_size)
            state:  Decoder Hidden (num_layers, batch_sz, dec_units)
            attention_weights: (batch_sz, max_len, 1)
        '''
        # hidden shape == (1, batch_size, hidden size)
        context_vector, attention_weights = self.attention(hidden, enc_output)
        
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        # takes case of the right portion of the model above (illustrated in red)
        x = self.out2emb(x)
        
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        #x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        x = torch.cat((context_vector.unsqueeze(1), x), -1)
        
        # passing the concatenated vector to the GRU
        # out: (batch_size, 1, hidden_size)
        out, state = self.rnn(x.float(), hidden.float())
        
        # output shape == (batch_size, vocab)
        out = self.fc(out.squeeze())
        
        return out, state, attention_weights