from utilities.lang_data_utils import GlyphStrawboss

hi_glyph = GlyphStrawboss("hi")
en_glyph = GlyphStrawboss("en")

##============ RNN Based =======================================================
import torch
from hypotheses.training_85.recurrent_nets_85 import model
weight_path = "hypotheses/training_85/Training_85_model.pth"
# load Model from source_files itself

weights = torch.load( weight_path, map_location=torch.device('cpu'))
model.load_state_dict(weights)
model.eval()

def inferencer(word, topk = 3):
    if topk == 1:
        in_vec = torch.from_numpy(en_glyph.word2xlitvec(word))
        out = model.inference(in_vec)
        result =[ hi_glyph.xlitvec2word(out.numpy()) ]
        return result
    else:
        in_vec = torch.from_numpy(en_glyph.word2xlitvec(word))
        out_list = model.beam_inference(in_vec, beam_width = topk)
        result = [ hi_glyph.xlitvec2word(out.numpy()) for out in out_list]
        return result


if __name__ == "__main__":
    while(1):
        a = input()
        result = inferencer(a)
        print(result)