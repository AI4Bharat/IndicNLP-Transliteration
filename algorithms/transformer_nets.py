import torch
import torch.nn as nn
import math


class XFMR_Seq2Seq(nn.Module):
    '''
    TODO: Not yet complete Decoding part has to be fixed
    '''
    def __init__(self, input_vcb_sz, output_vcb_sz,
                    emb_dim, enc_layers, dec_layers,
                    attention_head = 8, feedfwd_dim = 1024,
                    dropout = 0,
                    device = "cpu"):
        super(XFMR_Seq2Seq, self).__init__()

        self.input_vcb_sz = input_vcb_sz
        self.output_vcb_sz = output_vcb_sz
        self.vector_dim = emb_dim  # same size will be used for all layers in transformer
        self.atten_head = attention_head
        self.enc_layers = enc_layers
        self.dec_layers =dec_layers
        self.feedfwd_dim = feedfwd_dim
        self.dropout = dropout


        self.in2embed = nn.Embedding(self.input_vcb_sz, self.vector_dim)
        self.out2embed = nn.Embedding(self.output_vcb_sz, self.vector_dim)
        self.xfmr = nn.Transformer(d_model= self.vector_dim, nhead= self.atten_head,
                                    num_encoder_layers= self.enc_layers,
                                    num_decoder_layers= self.dec_layers,
                                    dim_feedforward= self.feedfwd_dim,
                                    activation='relu',
                                    dropout= self.dropout)
        self.out_fc = nn.Sequential( nn.LeakyReLU(),
            nn.Linear(self.vector_dim, self.output_vcb_sz)
            )

    def forward(self, src, tgt):
        '''
        src: (batch, in_seq_len-padded)
        tgt: (batch, out_seq_len-padded)
        '''

        # src_emb: (batch, in_seq_len, vector_dim)
        src_emb = self.in2embed(src)
        # tgt_emb: (batch, out_seq_len, vector_dim)
        tgt_emb = self.out2embed(tgt)

        # Z_emb: (seq_len,batch,vector_dim) -> for transformer
        src_emb = src_emb.permute(1,0,2)
        tgt_emb = tgt_emb.permute(1,0,2)

        # out: (out_seq_len,batch,vector_dim)
        out = self.xfmr(src_emb, tgt_emb)

        # out: (batch,out_seq_len,vector_dim)
        out = out.permute(1,0,2).contiguous()
        print(out)
        # out: (batch,out_seq_len,out_vcb_dim)
        out = self.out_fc(out)

        return out

##------------------------------------------------------------------------------

class XFMR_Neophyte(nn.Module):
    def __init__(self, input_vcb_sz, output_vcb_sz,
                    emb_dim, n_layers,
                    attention_head = 8, feedfwd_dim = 1024,
                    max_seq_len = 50,
                    dropout = 0, device = "cpu"):

        super(XFMR_Neophyte, self).__init__()
        self.device = device

        self.input_vcb_sz = input_vcb_sz
        self.output_vcb_sz = output_vcb_sz
        self.vector_dim = emb_dim  # same size will be used for all layers in transformer
        self.atten_head = attention_head
        self.n_layers = n_layers
        self.feedfwd_dim = feedfwd_dim
        self.max_seq_len = max_seq_len
        self.dropout = dropout


        self.in2embed = nn.Embedding(self.input_vcb_sz, self.vector_dim)
        # pos_encoder non-learnable layer
        self.pos_encoder = PositionalEncoding(self.vector_dim, self.dropout,
                                                device = self.device)

        _enc_layer = nn.TransformerEncoderLayer(d_model= self.vector_dim,
                                                nhead= self.atten_head,
                                                dim_feedforward= self.feedfwd_dim,
                                                dropout= self.dropout
                                                )
        self.xfmr_enc = nn.TransformerEncoder(_enc_layer, num_layers= n_layers)

        self.out_fc = nn.Sequential( nn.LeakyReLU(),
            nn.Linear(self.vector_dim, self.vector_dim),
            nn.LeakyReLU(),
            nn.Linear(self.vector_dim, self.output_vcb_sz),
            )

    def forward(self, src, src_sz):
        '''
        src: (batch, max_seq_len-padded)
        tgt: (batch, max_seq_len-padded)
        '''

        # src_emb: (batch, in_seq_len, vector_dim)
        src_emb = self.in2embed(src)
        # src_emb: (max_seq_len, batch, vector_dim) -> for transformer
        src_emb = src_emb.permute(1,0,2)

        # src_emb: (max_seq_len, batch, vector_dim)
        src_emb =  src_emb * math.sqrt(self.vector_dim)
        src_emb = self.pos_encoder(src_emb)
        # out: (max_seq_len, batch, vector_dim)
        out = self.xfmr_enc(src_emb)

        # out: (batch, max_seq_len, vector_dim)
        out = out.permute(1,0,2).contiguous()

        # out: (batch, max_seq_len, out_vcb_dim)
        out = self.out_fc(out)
        # out: (batch, out_vcb_dim, max_seq_len)
        out = out.permute(0,2,1).contiguous()

        return out

    def inference(self, x):

        # inp: (1, max_seq_len)
        inp = torch.zeros(1, self.max_seq_len, dtype= torch.long).to(self.device)
        in_sz = min(x.shape[0], self.max_seq_len)
        inp[0, 0:in_sz ] = x[0:in_sz]

        # src_emb: (1, max_seq_len, vector_dim)
        src_emb = self.in2embed(inp)
        # src_emb: (max_seq_len,1,vector_dim) -> for transformer
        src_emb = src_emb.permute(1,0,2)

        # src_emb: (max_seq_len, 1, vector_dim)
        src_emb =  src_emb * math.sqrt(self.vector_dim)
        src_emb = self.pos_encoder(src_emb)
        # out: (max_seq_len, 1, vector_dim)
        out = self.xfmr_enc(src_emb)

        # out: (1, max_seq_len, vector_dim)
        out = out.permute(1,0,2).contiguous()

        # out: (1, max_seq_len, out_vcb_dim)
        out = self.out_fc(out)
        # prediction: ( max_seq_len )
        prediction = torch.argmax(out, dim=2).squeeze()

        return prediction

##------------------------------------------------------------------------------

class PositionalEncoding(nn.Module):

    def __init__(self, vector_dim, dropout=0, max_seq_len=50, device = "cpu"):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        #pe :shp: (max_seq_len, vector_dim)
        self.pe = torch.zeros(max_seq_len, vector_dim).to(device)
        position = torch.arange(0,max_seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, vector_dim, 2).float() * (-math.log(10000.0) / vector_dim))
        self.pe[:, 0::2] = torch.sin(position * div_term)
        self.pe[:, 1::2] = torch.cos(position * div_term)
        #pe :shp: max_seq_len, 1 ,vector_dim
        self.pe = self.pe.unsqueeze(0).transpose(0, 1)

    def forward(self, x):

        # x :shp: (seq_len, batch_size, vector_dim)
        x = x + self.pe
        return self.dropout(x)