import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, enc_embed_dim, hidden_dim ,
                       enc_layers = 1, enc_dropout = 0):
        super(Encoder, self).__init__()

        self.enc_layers = enc_layers
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim #src_vocab_sz
        self.enc_embed_dim = enc_embed_dim
        self.embedding = nn.Embedding(self.input_dim, self.enc_embed_dim)
        self.gru = nn.GRU(input_size= self.enc_embed_dim,
                          hidden_size= self.hidden_dim,
                          num_layers= self.enc_layers )

    def forward(self, x, x_sz):
        """
        src_sz: (batch_size, 1) -  Unpadded sequence lengths used for pack_pad
        """
        # x: batch_size, max_length, enc_embed_dim
        x = self.embedding(x)

        ## pack the padded data
        # x: max_length, batch_size, enc_embed_dim -> for pack_pad
        x = x.permute(1,0,2)
        x = pack_padded_sequence(x, x_sz, enforce_sorted=False) # unpad

        # output: packed_size, batch_size, enc_embed_dim
        # hidden: n_layer, batch_size, hidden_dim
        output, hidden = self.gru(x) # gru returns hidden state of all timesteps as well as hidden state at last timestep

        ## pad the sequence to the max length in the batch
        # output: max_length, batch_size, hidden_dim)
        output, _ = pad_packed_sequence(output)

        # output: batch_size, max_length, hidden_dim
        output = output.permute(1,0,2)

        return output, hidden

    def inference(x):
        x = self.embedding(x)
        output, hidden = self.gru(x)
        return output, hidden




class Decoder(nn.Module):
    def __init__(self, output_dim, dec_embed_dim, hidden_dim,
                       dec_layers = 1, dec_dropout = 0):
        super(Decoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.output_dim = output_dim #tgt_vocab_sz
        self.dec_embed_dim = dec_embed_dim
        self.dec_layers = dec_layers
        self.embedding = nn.Embedding(self.output_dim, self.dec_embed_dim)
        self.gru = nn.GRU(input_size= self.dec_embed_dim + self.hidden_dim,
                          hidden_size= self.hidden_dim,
                          num_layers= self.dec_layers,
                          batch_first = True )
        self.fc = nn.Linear(self.hidden_dim, self.output_dim)


        ##----- Attention ----------
        self.W1 = nn.Linear( self.hidden_dim, self.hidden_dim)
        self.W2 = nn.Linear( self.dec_layers * self.hidden_dim, self.hidden_dim)
        self.V = nn.Linear( self.hidden_dim, 1)

    def attention(self, x, hidden, enc_output):
        '''
        x: (batch_size, 1, dec_embed_dim) -> after Embedding
        enc_output: batch_size, max_length, dec_embed_dim
        hidden: n_layers, batch_size, hidden_size
        '''

        ## perform addition to calculate the score

        # hidden (batch_size, n_layers, hidden_dim) -> to prevent distortion
        hidden = hidden.permute(1, 0, 2)

        # W1x_enc_out: batch_size, max_length, dec_embed_dim
        # W2x_hidden: batch_size, dec_embed_dim
        W1x_enc_out = self.W1(enc_output)
        W2x_hidden = self.W2(hidden.reshape(-1, self.dec_layers*self.hidden_dim))

        # W2x_hidden: batch_size, 1, dec_embed_dim -> with axis for time step
        W2x_hidden = W2x_hidden.unsqueeze(1)

        # score: (batch_size, max_length, hidden_dim)
        score = torch.tanh( W1x_enc_out + W2x_hidden)

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = torch.softmax(self.V(score), dim=1)

        # context_vector shape after sum == (batch_size, hidden_dim)
        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)
        # context_vector: batch_size, 1, hidden_dim
        context_vector = context_vector.unsqueeze(1)

        # attend_out (batch_size, 1, dec_embed_dim + hidden_size)
        attend_out = torch.cat((context_vector, x), -1)

        return attend_out

    def forward(self, x, hidden, enc_output):
        '''
        x: (batch_size, 1)
        enc_output: batch_size, max_length, dec_embed_dim
        hidden: 1, batch_size, hidden_size
        '''

        # x (batch_size, 1, dec_embed_dim) -> after embedding
        x = self.embedding(x)

        # x (batch_size, 1, dec_embed_dim + hidden_size) -> after attention
        x = self.attention( x, hidden, enc_output)

        # passing the concatenated vector to the GRU
        # output: (batch_size, 1, hidden_size)
        # hidden: 1, batch_size, hidden_size
        output, hidden = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output =  output.view(-1, output.size(2))

        # output shape == (batch_size * 1, output_dim)
        output = self.fc(output)

        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, src, tgt, src_sz, teacher_forcing_ratio = 0):
        '''
        src: (batch_size, sequence_len.padded)
        tgt: (batch_size, sequence_len.padded)
        src_sz: [batch_size, 1] -  Unpadded sequence lengths
        '''

        # enc_output: (batch_size, max_length, hidden_dim)
        # enc_hidden: (batch_size, hidden_dim)
        enc_output, enc_hidden = self.encoder(src, src_sz)

        batch_size = tgt.shape[0]
        dec_hidden = enc_hidden

        # pred_vecs: (sequence_sz, batch_size, output_dim)
        pred_vecs = torch.zeros(tgt.size(1) , batch_size, self.decoder.output_dim).to(self.device)

        # dec_input: (batch_size, 1)
        dec_input = tgt[:,0].unsqueeze(1) # initialize to start token
        pred_vecs[0] = dec_input

        for t in range(1, tgt.size(1)):
            # dec_hidden: 1, batch_size, hidden_dim
            # dec_output: batch_size, output_dim
            # dec_input: (batch_size, 1)
            dec_output, dec_hidden = self.decoder( dec_input,
                                               dec_hidden,
                                               enc_output,  )
            pred_vecs[t] = dec_output

            # # prediction: batch_size
            prediction = torch.argmax(dec_output, dim=1)

            # Teacher Forcing
            if random.random() < teacher_forcing_ratio:
                dec_input = tgt[:, t].unsqueeze(1)
            else:
                dec_input = prediction.unsqueeze(1)

        # pred_vecs: (batch_size, sequence_sz)
        pred_vecs = pred_vecs.permute(1,0,2)

        return pred_vecs

    def inference(self, src, start_tok, end_tok, max_tgt_sz = 50 ):

        enc_output, enc_hidden = self.encoder(src)
        dec_hidden = enc_hidden
        pred_vec = torch.zeros(max_tgt_sz).to(self.device)
        dec_input = start_tok.unsqueeze(1)
        pred[0] = dec_input
        for t in range(1, max_tgt_sz):

            dec_output, dec_hidden = self.decoder( dec_input,
                                               dec_hidden,
                                               enc_output,  )
            prediction = torch.argmax(dec_output, dim=1)
            pred_vec[t] = prediction
            dec_input = prediction.unsqueeze(1)

            if torch.eq(prediction, end_tok):
                break

        return pred_vec




