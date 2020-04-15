import torch
import torch.nn as nn
import random

class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, hidden_dim ,
                       layers = 1, dropout = 0, bidirectional =False,
                       device = "cpu"):
        super(Encoder, self).__init__()

        self.device = device
        self.input_dim = input_dim #src_vocab_sz
        self.enc_layers = layers
        self.enc_directions = 2 if bidirectional else 1
        self.enc_hidden_dim = hidden_dim
        self.enc_embed_dim = embed_dim
        self.embedding = nn.Embedding(self.input_dim, self.enc_embed_dim)
        self.enc_gru = nn.GRU(input_size= self.enc_embed_dim,
                          hidden_size= self.enc_hidden_dim,
                          num_layers= self.enc_layers,
                          bidirectional= bidirectional)

    def forward(self, x, x_sz, hidden = None):
        """
        src_sz: (batch_size, 1) -  Unpadded sequence lengths used for pack_pad
        """
        batch_sz = x.shape[0]
        # x: batch_size, max_length, enc_embed_dim
        x = self.embedding(x)

        if not hidden:
            # hidden: n_layer*num_directions, batch_size, hidden_dim
            hidden = torch.zeros((self.enc_layers** self.enc_directions, batch_sz,
                        self.enc_hidden_dim )).to(self.device)

        ## pack the padded data
        # x: max_length, batch_size, enc_embed_dim -> for pack_pad
        x = x.permute(1,0,2)
        x = nn.utils.rnn.pack_padded_sequence(x, x_sz, enforce_sorted=False) # unpad

        # output: packed_size, batch_size, enc_embed_dim
        # hidden: n_layer, batch_size, hidden_dim*num_directions
        output, hidden = self.enc_gru(x, hidden) # gru returns hidden state of all timesteps as well as hidden state at last timestep

        ## pad the sequence to the max length in the batch
        # output: max_length, batch_size, hidden_dim)
        output, _ = nn.utils.rnn.pad_packed_sequence(output)

        # output: batch_size, max_length, hidden_dim
        output = output.permute(1,0,2)

        return output, hidden




class Decoder(nn.Module):
    def __init__(self, output_dim, embed_dim, hidden_dim,
                       layers = 1, dropout = 0,
                       enc_outstate_dim = None, # enc_directions *enc_hidden_dim
                       device = "cpu"):
        super(Decoder, self).__init__()

        self.device = device
        self.output_dim = output_dim #tgt_vocab_sz
        self.dec_hidden_dim = hidden_dim
        self.dec_embed_dim = embed_dim
        self.dec_layers = layers
        self.enc_outstate_dim = enc_outstate_dim if enc_outstate_dim else hidden_dim

        self.embedding = nn.Embedding(self.output_dim, self.dec_embed_dim)
        self.gru = nn.GRU(input_size= self.dec_embed_dim + self.enc_outstate_dim, # to concat attention_output
                          hidden_size= self.dec_hidden_dim, # previous Hidden
                          num_layers= self.dec_layers,
                          batch_first = True )
        self.fc = nn.Sequential(
            nn.Linear(self.dec_hidden_dim, self.dec_embed_dim), nn.LeakyReLU(),
            # nn.Linear(self.dec_embed_dim, self.dec_embed_dim), nn.LeakyReLU(), # removing to reduce size
            nn.Linear(self.dec_embed_dim, self.output_dim),
            )

        ##----- Attention ----------

        self.W1 = nn.Linear( self.enc_outstate_dim, self.dec_hidden_dim)
        self.W2 = nn.Linear( self.dec_hidden_dim, self.dec_hidden_dim)
        self.V = nn.Linear( self.dec_hidden_dim, 1)

    def attention(self, x, hidden, enc_output):
        '''
        x: (batch_size, 1, dec_embed_dim) -> after Embedding
        enc_output: batch_size, max_length, enc_hidden_dim *num_directions
        hidden: n_layers, batch_size, hidden_size
        '''

        ## perform addition to calculate the score

        # hidden_with_time_axis shape == (batch_size, 1, hidden_dim)
        ## hidden_with_time_axis = hidden.permute(1, 0, 2) ## replaced with below 2lines
        hidden_with_time_axis = torch.sum(hidden, axis=0)
        hidden_with_time_axis = hidden_with_time_axis.unsqueeze(1)

        # score: (batch_size, max_length, hidden_dim)
        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))

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
        hidden: n_layer, batch_size, hidden_size
        '''

        # x (batch_size, 1, dec_embed_dim) -> after embedding
        x = self.embedding(x)

        # x (batch_size, 1, dec_embed_dim + hidden_size) -> after attention
        x = self.attention( x, hidden, enc_output)

        # passing the concatenated vector to the GRU
        # output: (batch_size, n_layers, hidden_size)
        # hidden: n_layers, batch_size, hidden_size
        output, hidden = self.gru(x, hidden)

        # output shape == (batch_size * 1, hidden_size)
        output =  output.view(-1, output.size(2))

        # output shape == (batch_size * 1, output_dim)
        output = self.fc(output)

        return output, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, dropout = 0, device = "cpu"):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

        assert decoder.enc_outstate_dim == encoder.enc_directions*encoder.enc_hidden_dim,"Set `enc_out_dim` correctly in decoder"
        assert decoder.dec_hidden_dim == encoder.enc_hidden_dim, "Hidden Size of encoder and decoder must be same, currently"

        #TODO: Support Different Hidden_size for Enc&Dec; Modify the ConvLayer to facilitate
        self.enc_hid_1ax = encoder.enc_directions * encoder.enc_layers
        self.dec_hid_1ax = decoder.dec_layers
        self.e2d_hidden_conv = nn.Conv1d(self.enc_hid_1ax, self.dec_hid_1ax, 1)

    def enc2dec_hidden(self, enc_hidden):
        """
        enc_hidden: n_layer, batch_size, hidden_dim*num_directions
        """
        batch_sz = enc_hidden.shape[1]
        # hidden: batch_size, enc_layer*num_directions, enc_hidden_dim
        hidden = enc_hidden.permute(1,0,2).contiguous()
        # hidden: batch_size, dec_layers, dec_hidden_dim -> [N,C,Tstep]
        hidden = self.e2d_hidden_conv(hidden)

        # hidden: dec_layers, batch_size , dec_hidden_dim
        hidden_for_dec = hidden.permute(1,0,2).contiguous()

        return hidden_for_dec


    def forward(self, src, tgt, src_sz, teacher_forcing_ratio = 0):
        '''
        src: (batch_size, sequence_len.padded)
        tgt: (batch_size, sequence_len.padded)
        src_sz: [batch_size, 1] -  Unpadded sequence lengths
        '''
        batch_size = tgt.shape[0]

        # enc_output: (batch_size, padded_seq_length, enc_hidden_dim*num_direction)
        # enc_hidden: (enc_layers*num_direction, batch_size, hidden_dim)
        enc_output, enc_hidden = self.encoder(src, src_sz)

        dec_hidden = self.enc2dec_hidden(enc_hidden)

        # pred_vecs: (batch_size, output_dim, sequence_sz) -> shape required for CELoss
        pred_vecs = torch.zeros(batch_size, self.decoder.output_dim, tgt.size(1)).to(self.device)

        # dec_input: (batch_size, 1)
        dec_input = tgt[:,0].unsqueeze(1) # initialize to start token

        for t in range(1, tgt.size(1)):
            # dec_hidden: dec_layers, batch_size, hidden_dim
            # dec_output: batch_size, output_dim
            # dec_input: (batch_size, 1)
            dec_output, dec_hidden = self.decoder( dec_input,
                                               dec_hidden,
                                               enc_output,  )
            pred_vecs[:,:,t] = dec_output

            # # prediction: batch_size
            prediction = torch.argmax(dec_output, dim=1)

            # Teacher Forcing
            if random.random() < teacher_forcing_ratio:
                dec_input = tgt[:, t].unsqueeze(1)
            else:
                dec_input = prediction.unsqueeze(1)

        return pred_vecs #(batch_size, output_dim, sequence_sz)

    def inference(self, src, max_tgt_sz=50):
        '''
        src: (sequence_len)
        '''
        batch_size = 1
        start_tok = src[0]
        end_tok = src[-1]
        src_sz = torch.tensor([len(src)])
        src_ = src.unsqueeze(0)

        enc_output, enc_hidden = self.encoder(src_, src_sz)
        dec_hidden = self.enc2dec_hidden(enc_hidden)

        pred_arr = torch.zeros(max_tgt_sz, 1).to(self.device)
        # dec_input: (batch_size, 1)
        dec_input = start_tok.view(1,1) # initialize to start token

        for t in range(max_tgt_sz):
            dec_output, dec_hidden = self.decoder( dec_input,
                                               dec_hidden,
                                               enc_output,  )
            prediction = torch.argmax(dec_output, dim=1)
            dec_input = prediction.unsqueeze(1)
            pred_arr[t] = prediction
            if torch.eq(prediction, end_tok):
                break
        return pred_arr.squeeze()

    def beam_inference(self, src, beam_width=3, max_tgt_sz=50):
        ''' Search based decoding
        src: (sequence_len)
        '''
        def _avg_score(p_tup):
            """ Used for Sorting
            TODO: Dividing by length of sequence power alpha as hyperparam
            """
            return p_tup[0]

        import sys
        batch_size = 1
        start_tok = src[0]
        end_tok = src[-1]
        src_sz = torch.tensor([len(src)])
        src_ = src.unsqueeze(0)

        enc_output, enc_hidden = self.encoder(src_, src_sz)
        init_dec_hidden = self.enc2dec_hidden(enc_hidden)

        # top_pred[][0] = Σ-log_softmax
        # top_pred[][1] = sequence torch.tensor shape: (1)
        # top_pred[][2] = dec_hidden
        top_pred_list = [ (0, start_tok.unsqueeze(0) , init_dec_hidden) ]

        for t in range(max_tgt_sz):
            cur_pred_list = []

            for p_tup in top_pred_list:
                if p_tup[1][-1] == end_tok:
                    cur_pred_list.append(p_tup)
                    continue

                # dec_hidden: dec_layers, 1, hidden_dim
                # dec_output: 1, output_dim
                dec_output, dec_hidden = self.decoder( x = p_tup[1][-1].view(1,1), #dec_input: (1,1)
                                                    hidden = p_tup[2],
                                                    enc_output = enc_output, )

                ## π{prob} = Σ{log(prob)} -> to prevent diminishing
                # dec_output: (1, output_dim)
                dec_output = nn.functional.log_softmax(dec_output, dim=1)
                # pred_topk.values & pred_topk.indices: (1, beam_width)
                pred_topk = torch.topk(dec_output, k=beam_width, dim=1)

                for i in range(beam_width):
                    sig_logsmx_ = p_tup[0] + pred_topk.values[0][i]
                    # seq_tensor_ : (seq_len)
                    seq_tensor_ = torch.cat( (p_tup[1], pred_topk.indices[0][i].view(1)) )

                    cur_pred_list.append( (sig_logsmx_, seq_tensor_, dec_hidden) )

            cur_pred_list.sort(key = _avg_score, reverse =True) # Maximized order
            top_pred_list = cur_pred_list[:beam_width]

            # check if end_tok of all topk
            end_flags_ = [1 if t[1][-1] == end_tok else 0 for t in top_pred_list]
            if beam_width == sum( end_flags_ ): break

        prediction_list = [t[1] for t in top_pred_list ]

        return prediction_list


