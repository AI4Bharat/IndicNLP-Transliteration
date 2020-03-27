from utilities.lang_data_utils import GlyphStrawboss

hi_glyph = GlyphStrawboss("hi")
en_glyph = GlyphStrawboss("en")

##============ RNN Based =======================================================
import torch
from algorithms.recurrent_nets import Encoder, Decoder, Seq2Seq

## Network Code Tag: rnn_0.1
weight_path = "hypotheses/training-90/Training_90_model-103.pth"
input_dim = en_glyph.size()
output_dim = hi_glyph.size()
enc_emb_dim = 128
dec_emb_dim = 128
hidden_dim = 256
n_layers = 1

enc = Encoder(input_dim= input_dim,
                enc_embed_dim = enc_emb_dim,
                hidden_dim= hidden_dim,
                enc_layers= n_layers)
dec = Decoder(output_dim= output_dim,
                dec_embed_dim = dec_emb_dim,
                hidden_dim= hidden_dim,
                dec_layers= n_layers)

rnn_model = Seq2Seq(enc, dec, "cpu")
weights = torch.load( weight_path, map_location=torch.device('cpu'))
rnn_model.load_state_dict(weights)


def inferencer(word):
    in_vec = torch.from_numpy(en_glyph.word2xlitvec(word))
    out = rnn_model.inference(in_vec)
    result = hi_glyph.xlitvec2word(out.numpy())
    return result


if __name__ == "__main__":
    while(1):
        a = input()
        result = inferencer(a)
        print(result)