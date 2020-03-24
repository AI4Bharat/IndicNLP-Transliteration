import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, enc_hid_dim ,
                       enc_layers = 1, enc_dropout = 0):
        super(Encoder, self).__init__()

        self.enc_layers = enc_layers
        self.enc_hid_dim = enc_hid_dim
        self.input_dim = input_dim #src_vocab_sz
        self.embed_dim = embed_dim
        self.embedding = nn.Embedding(self.input_dim, self.embed_dim)
        self.gru = nn.GRU(input_size= self.embed_dim,
                          hidden_size= self.enc_hid_dim,
                          num_layers= self.enc_layers )

    def forward(self, x, x_sz):
        """
        src_sz: [batch_size, 1] -  Unpadded sequence lengths used for pack_pad
        """
        # x: batch_size, max_length, embed_dim
        x = self.embedding(x)

        ## pack the padded data
        # x: max_length, batch_size, embed_dim -> for pack_pad
        x = x.permute(1,0,2)
        x = pack_padded_sequence(x, x_sz, enforce_sorted=False) # unpad

        # output: packed_size, batch_size, embed_dim
        # hidden: 1, batch_size, enc_hid_dim
        output, hidden = self.gru(x) # gru returns hidden state of all timesteps as well as hidden state at last timestep

        ## pad the sequence to the max length in the batch
        # output: max_length, batch_size, embed_dim)
        output, _ = pad_packed_sequence(output)


        # x: batch_size, max_length, embed_dim
        # hidden: 1, batch_size, enc_hid_dim
        output = output.permute(1,0,2)
        hidden = hidden.permute(1,0,2)

        return output, hidden




class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, dec_hid_dim, enc_hid_dim,
                       dec_layers = 1, dec_dropout = 0):
        super(Decoder, self).__init__()

        self.dec_hid_dim = dec_hid_dim
        self.enc_hid_dim = enc_hid_dim
        self.output_dim = output_dim #tgt_vocab_sz
        self.embed_dim = embed_dim
        self.dec_layers = dec_layers
        self.embedding = nn.Embedding(self.output_dim, self.embed_dim)
        self.gru = nn.GRU(input_size= self.embed_dim + self.enc_hid_dim,
                          hidden_size= self.dec_hid_dim,
                          num_layers= self.dec_layers,
                          batch_first = True )
        self.fc = nn.Linear(self.enc_hid_dim, self.output_dim)


        ##----- Attention ----------
        self.W1 = nn.Linear(self.enc_hid_dim, self.dec_hid_dim)
        self.W2 = nn.Linear(self.enc_hid_dim, self.dec_hid_dim)
        self.V = nn.Linear(self.enc_hid_dim, 1)

    def forward(self, x, hidden, enc_output):

        ##----- Attention ----------

        # enc_output original: (max_length, batch_size, enc_hid_dim)
        # enc_output converted == (batch_size, max_length, hidden_size)

        enc_output = enc_output.permute(1,0,2)

        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size) -> to perform addition to calculate the score
        print("hiddedshape", hidden.shape)
        hidden_with_time_axis = hidden.permute(1, 0, 2)

        # score: (batch_size, max_length, hidden_size)
        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))

        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = torch.softmax(self.V(score), dim=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)

        ## -------------------------

        # x shape after passing through embedding == (batch_size, 1, emb_dim)

        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, emb_dim + hidden_size)
        #x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # ? Looks like attention vector in diagram of source
        x = torch.cat((context_vector.unsqueeze(1), x), -1)

        # passing the concatenated vector to the GRU
        # output: (batch_size, 1, hidden_size)
        output, state = self.gru(x)


        # output shape == (batch_size * 1, hidden_size)
        output =  output.view(-1, output.size(2))

        # output shape == (batch_size * 1, vocab)
        output = self.fc(output)

        return output, state


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt, src_sz, tgt_sz):
        '''
        src: (batch_size, sequence_len.padded)
        tgt: (batch_size, sequence_len.padded)
        src_sz: [batch_size, 1] -  Unpadded sequence lengths
        '''

        # enc_output: (batch_size, max_length, enc_hid_dim)
        # enc_hidden: (batch_size, enc_hid_dim)
        enc_output, enc_hidden = self.encoder(src, src_sz)

        batch_size = tgt.shape[0]
        dec_hidden = enc_hidden


        # preds: (sequence_sz, batch_size, output_dim )
        preds = torch.zeros(tgt.size(1) , batch_size, self.decoder.output_dim)


        # dec_input: (batch_size, 1)
        dec_input = tgt[:,0].unsqueeze(1) # initialize to start token

        for t in range(1, tgt.size(1)):
            # enc_hidden: 1, batch_size, enc_hidden
            # output: max_length, batch_size, enc_hidden
            prediction, dec_hidden = self.decoder( dec_input,
                                               dec_hidden,
                                               enc_output,  )

            preds[t] = prediction

        return preds
