import os
from utilities.lang_data_utils import GlyphStrawboss, VocabSanitizer, VocableStrawboss
import utilities.running_utils as rutl

''' VacabSanitizer usage
voc_sanitize = VocabSanitizer("data/X_word_list.json")
result = voc_sanitize.reposition(result)
'''

hi_glyph = GlyphStrawboss("hi")
en_glyph = GlyphStrawboss("en")
hi_vocab = VocableStrawboss("data/maithili/mai_all_words_sorted.json")
device = "cpu"

##=============== Models =======================================================
import torch

from hypotheses.training_mai_103.recurrent_nets_mai_103 import model
weight_path = "hypotheses/training_mai_103/weights/Training_mai_103_model.pth"
# voc_sanitize = VocabSanitizer("data/X_word_list.json")

weights = torch.load( weight_path, map_location=torch.device(device))
model.load_state_dict(weights)
model.eval()

# --- Correction model ---
from hypotheses.training_mai_103.recurrent_nets_mai_103 import corr_model
corr_weight_path = "hypotheses/training_mai_103/weights/Training_mai_103_corrnet.pth"

corr_weights = torch.load( corr_weight_path, map_location=torch.device(device))
corr_model.load_state_dict(corr_weights)
corr_model.eval()


##==============================================================================

def pred_contrive(corr_lst, pred_lst):
    out =[]
    for l in corr_lst:
        if (l not in out) and (l != "<UNK>"):
            out.append(l)
    for l in pred_lst:
        if l not in out:
            out.append(l)
    return out[:len(corr_lst)]


def inferencer(word, topk = 3):
    if topk == 0:
        in_vec = torch.from_numpy(en_glyph.word2xlitvec(word)).to(device)
        out = model.inference(in_vec)
        out = corr_model.inference(out)
        # result =[ hi_glyph.xlitvec2word(out.cpu().numpy()) ]
        result = [ hi_vocab.get_word(out.cpu().numpy()) ]
        return result
    else:
        in_vec = torch.from_numpy(en_glyph.word2xlitvec(word)).to(device)
        ## change to active or passive beam
        out_list = model.active_beam_inference(in_vec, beam_width = topk)
        p_result = [ hi_glyph.xlitvec2word(out.cpu().numpy()) for out in out_list]

        out_list = [ corr_model.inference(out) for out in out_list]
        c_result = [ hi_vocab.get_word(out.cpu().numpy()) for out in out_list]
        result = pred_contrive(c_result, p_result)
        return result



def infer_analytics(word):
    """Analytics by ploting values
    """
    save_path = os.path.dirname(weight_path) + "/viz_log/"
    if not os.path.exists(save_path): os.makedirs(save_path)

    in_vec = torch.from_numpy(en_glyph.word2xlitvec(word))
    out, aw = model.inference(in_vec, debug=1)
    result = hi_glyph.xlitvec2word(out.numpy())

    rutl.attention_weight_plotter(result, word, aw.detach().numpy()[:len(result)],
                                    save_path=save_path )
    return result


if __name__ == "__main__":
    while(1):
        a = input()
        result = inferencer(a)
        print(result)