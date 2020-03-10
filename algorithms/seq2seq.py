"""
Deep Learning based Encoder-Decoder models.

Seq2Seq with Attention inspired from:
https://medium.com/dair-ai/neural-machine-translation-with-attention-using-pytorch-a66523f1669f
https://tsdaemon.github.io/2018/07/08/nmt-with-pytorch-encoder-decoder.html
"""

import torch
import torch.nn as nn
from algorithms.encoder import Encoder
from algorithms.decoder import Decoder

class EncoderDecoder(nn.Module):
    def __init__(self, model_cfg, input_vocab, output_vocab):
        super(EncoderDecoder, self).__init__()
        rnn_type = model_cfg.rnn_type.lower()
        if rnn_type == 'gru':
            rnn_type = nn.GRU
        elif rnn_type == 'rnn':
            rnn_type = nn.RNN
        else:
            print(rnn_type, ' rnn_type is not available; using GRU by default')
            rnn_type = nn.GRU
        
        enc_cfg = model_cfg.encoder
        self.encoder = Encoder(rnn_type, len(input_vocab), enc_cfg.hidden_units,
                               enc_cfg.embed_size, enc_cfg.bidirectional, enc_cfg.num_layers)
        
        dec_cfg = model_cfg.decoder
        self.decoder = Decoder(rnn_type, len(output_vocab), dec_cfg.hidden_units,
                               enc_cfg.hidden_units, dec_cfg.embed_size, dec_cfg.num_layers)
        self.start_code, self.end_code = input_vocab['$'], input_vocab['#']
        self.output_vocab_size = len(output_vocab)
        self.MAX_DECODE_STEPS = dec_cfg.max_steps
    
    def decode(self, dec_hidden, enc_output, y_ohe=None, teacher_force=False):
        outputs = []
        if dec_hidden is None:
            # hidden_shape: (num_layers,batch,hidden_size)
            dec_hidden = torch.zeros((self.decoder.rnn.num_layers, enc_output.shape[0], self.decoder.dec_units)).to(enc_output.device)
        
        if teacher_force and y_ohe is not None:
            dec_input = y_ohe[:, 0].unsqueeze(1)
            outputs.append(y_ohe[:, 0])
            for t in range(1, y_ohe.size(1)):
                prediction, dec_hidden, _ = self.decoder(dec_input.float(), dec_hidden, enc_output)
                outputs.append(prediction)
                # use teacher forcing - feeding the target as the next input (via dec_input)
                dec_input = y_ohe[:, t].unsqueeze(1)
        else:
            dec_input = torch.zeros(enc_output.shape[0], 1, self.output_vocab_size).to(enc_output.device) #(batch_size, 1, out_size)
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
        '''
        Arguments:
            x:        Input in OHE (max_len, batch_sz, in_vocab_sz)
            x_len:    Actual lengths of each sequence (batch_size)
            y_ohe:    Outputs in OHE (batch_sz, max_out_len, out_vocab_sz)
        
        Returns:
            outputs:  Seq/Array of outputs of shape (batch_sz, out_vocab_sz)
        '''
        # Run encoder
        enc_output, enc_hidden = self.encoder(x.float(), x_len)
        # enc_output original: (max_length, batch_size, enc_units)
        # enc_output converted == (batch_size, max_length, hidden_size)
        enc_output = enc_output.permute(1,0,2) # since batch_first=True
        
        # TODO: What if I don't pass the last hidden state itself? (since we're having attention)
        # Convert (num_layers*num_directions,batch,hidden_size) to (num_layers,batch,hidden_size*num_directions)
        if self.encoder.num_directions > 1:
            enc_hidden = enc_hidden.permute(1, 0, 2) \
                .reshape(-1, self.encoder.rnn.num_layers, self.encoder.enc_units) \
                .permute(1, 0, 2).contiguous()
        
        # Run decoder step-by-step
        dec_hidden = None
        if self.encoder.enc_units == self.decoder.dec_units and \
            self.encoder.rnn.num_layers == self.decoder.rnn.num_layers:
            dec_hidden = enc_hidden
        return self.decode(dec_hidden, enc_output, y_ohe, teacher_force)
