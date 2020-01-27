"""
Deep Learning based Encoder-Decoder models.

Seq2Seq with Attention inspired from:
https://medium.com/dair-ai/neural-machine-translation-with-attention-using-pytorch-a66523f1669f
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, vocab_size, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.gru = nn.GRU(self.vocab_size, self.enc_units)
        
    def forward(self, x, lens):
        # x: batch_size, max_length, vocab_size
                
        # x transformed = max_len X batch_size X vocab_size
        x = pack_padded_sequence(x, lens) # unpad
        
        # output: max_length, batch_size, enc_units
        # self.hidden: 1, batch_size, enc_units
        output, self.hidden = self.gru(x) # gru returns hidden state of all timesteps as well as hidden state at last timestep
        
        # pad the sequence to the max length in the batch
        output, _ = pad_packed_sequence(output)
        
        return output, self.hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, dec_units, enc_units, embedding_dim):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.out2hidden = nn.Linear(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim + self.enc_units, 
                          self.dec_units,
                          batch_first=True)
        self.fc = nn.Linear(self.dec_units, self.vocab_size)
        
        # used for attention
        self.W1 = nn.Linear(self.enc_units, self.dec_units)
        self.W2 = nn.Linear(self.enc_units, self.dec_units)
        self.V = nn.Linear(self.enc_units, 1)
    
    def forward(self, x, hidden, enc_output):
        # enc_output original: (max_length, batch_size, enc_units)
        # enc_output converted == (batch_size, max_length, hidden_size)
        enc_output = enc_output.permute(1,0,2)
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        hidden_with_time_axis = hidden.permute(1, 0, 2)
        
        # score: (batch_size, max_length, hidden_size) # Bahdanaus's
        # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
        # It doesn't matter which FC we pick for each of the inputs
        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
        
        #score = torch.tanh(self.W2(hidden_with_time_axis) + self.W1(enc_output))
          
        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = torch.softmax(self.V(score), dim=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)
        
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        # takes case of the right portion of the model above (illustrated in red)
        x = self.out2hidden(x)
        
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        #x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # ? Looks like attention vector in diagram of source
        x = torch.cat((context_vector.unsqueeze(1), x), -1)
        
        # passing the concatenated vector to the GRU
        # output: (batch_size, 1, hidden_size)
        output, state = self.gru(x.float(), hidden.float())
        
        
        # output shape == (batch_size * 1, hidden_size)
        output =  output.view(-1, output.size(2))
        
        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)
        
        return x, state, attention_weights

class EncoderDecoder(nn.Module):
    def __init__(self, units, input_vocab, output_vocab, embedding_dim):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(len(input_vocab), units)
        self.decoder = Decoder(len(output_vocab), units, units, embedding_dim)
        self.start_code, self.end_code = input_vocab['$'], input_vocab['#']
        self.output_vocab_size = len(output_vocab)
        self.MAX_DECODE_STEPS = 25
    
    def decode(self, dec_hidden, enc_output, y_ohe=None, teacher_force=False):
        outputs = []
        if teacher_force and y_ohe is not None:
            dec_input = y_ohe[:, 0].unsqueeze(1)
            outputs.append(y_ohe[:, 0])
            for t in range(1, y_ohe.size(1)):
                prediction, dec_hidden, _ = self.decoder(dec_input.float(), dec_hidden, enc_output)
                outputs.append(prediction)
                # use teacher forcing - feeding the target as the next input (via dec_input)
                dec_input = y_ohe[:, t].unsqueeze(1)
        else:
            dec_input = torch.zeros(enc_output.shape[1], 1, self.output_vocab_size).to(enc_output.device) #(batch_size, 1, out_size)
            dec_input[:, 0, self.start_code] = 1
            outputs.append(dec_input.squeeze(1))
            
            time_steps = y_ohe.size(1) if y_ohe is not None else self.MAX_DECODE_STEPS
            for t in range(1, time_steps):
                prediction, dec_hidden, _ = self.decoder(dec_input.float(), dec_hidden, enc_output)
                outputs.append(prediction)
                max_idx = torch.argmax(prediction, 1, keepdim=True)
                one_hot = torch.zeros(prediction.shape).to(prediction.device)
                one_hot.scatter_(1, max_idx, 1) # In dim 1, set max_idx's as 1
                dec_input = one_hot.detach().unsqueeze(1)
        return outputs
    
    def forward(self, x, x_len, y_ohe=None, teacher_force=False):
        # Run encoder
        enc_output, enc_hidden = self.encoder(x.float(), x_len)
        # Run decoder step-by-step
        return self.decode(enc_hidden, enc_output, y_ohe, teacher_force)
