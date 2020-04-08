import torch
import torch.nn as nn



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

        self.input_dim = input_vcb_sz
        self.output_dim = output_vcb_sz
        self.vector_dim = emb_dim  # same size will be used for all layers in transformer
        self.atten_head = attention_head
        self.enc_layers = enc_layers
        self.dec_layers =dec_layers
        self.feedfwd_dim = feedfwd_dim
        self.dropout = dropout


        self.in2embed = nn.Embedding(self.input_dim, self.vector_dim)
        self.out2embed = nn.Embedding(self.output_dim, self.vector_dim)
        self.xfmr = nn.Transformer(d_model= self.vector_dim, nhead= self.atten_head,
                                    num_encoder_layers= self.enc_layers,
                                    num_decoder_layers= self.dec_layers,
                                    dim_feedforward= self.feedfwd_dim,
                                    activation='relu',
                                    dropout= self.dropout)
        self.out_fc = nn.Sequential( nn.LeakyReLU(),
            nn.Linear(self.vector_dim, self.output_dim)
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


class XFMR_Neophyte(nn.Module):
    def __init__(self, input_vcb_sz, output_vcb_sz,
                    emb_dim, n_layers,
                    attention_head = 8, feedfwd_dim = 1024,
                    dropout = 0,
                    device = "cpu"):
        super(XFMR_Neophyte, self).__init__()

        self.input_dim = input_vcb_sz
        self.output_dim = output_vcb_sz
        self.vector_dim = emb_dim  # same size will be used for all layers in transformer
        self.atten_head = attention_head
        self.n_layers = n_layers
        self.feedfwd_dim = feedfwd_dim
        self.dropout = dropout


        self.in2embed = nn.Embedding(self.input_dim, self.vector_dim)

        _enc_layer = nn.TransformerEncoderLayer(d_model=self.vector_dim,
                                                nhead=self.atten_head)
        self.xfmr_enc = nn.TransformerEncoder(_enc_layer, num_layers= n_layers)

        self.out_fc = nn.Sequential( nn.LeakyReLU(),
            nn.Linear(self.vector_dim, self.output_dim)
            )

    def forward(self, src):
        '''
        src: (batch, in_seq_len-padded)
        tgt: (batch, out_seq_len-padded)
        '''

        # src_emb: (batch, in_seq_len, vector_dim)
        src_emb = self.in2embed(src)

        # src_emb: (seq_len,batch,vector_dim) -> for transformer
        src_emb = src_emb.permute(1,0,2)

        # out: (out_seq_len,batch,vector_dim)
        out = self.xfmr_enc(src_emb)

        # out: (batch,out_seq_len,vector_dim)
        out = out.permute(1,0,2).contiguous()

        # out: (batch,out_seq_len,out_vcb_dim)
        out = self.out_fc(out)
        # out: (batch,out_vcb_dim, out_seq_len)
        out = out.permute(0,2,1).contiguous()

        return out

    def inference(self, x, max_seq_size = 50):

        # inp: (1, max-seq_size)
        inp = torch.zeros(1, max_seq_size, dtype= torch.long).to(self.device)
        inp[0, 0:x.shape[0] ] = x

        # src_emb: (1, in_seq_len, vector_dim)
        src_emb = self.in2embed(inp)

        # src_emb: (seq_len,1,vector_dim) -> for transformer
        src_emb = src_emb.permute(1,0,2)

        # out: (max_seq_len,1,vector_dim)
        out = self.xfmr_enc(src_emb)

        # out: (1, max_seq_len, vector_dim)
        out = out.permute(1,0,2).contiguous()

        # out: (1, max_seq_len, out_vcb_dim)
        out = self.out_fc(out)
        # prediction: ( max_seq_len )
        prediction = torch.argmax(out, dim=2).squeeze()

        return prediction

