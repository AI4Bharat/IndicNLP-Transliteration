import torch
import torch.nn as nn



class XFMR_Seq2Seq(nn.Module):
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


