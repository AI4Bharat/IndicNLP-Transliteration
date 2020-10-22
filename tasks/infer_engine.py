import os
import sys
import torch
import utilities.lang_data_utils as lutl
import utilities.running_utils as rutl
from utilities.logging_utils import LOG2CSV

''' VacabSanitizer usage
voc_sanitize = lutl.VocabSanitizer("data/X_word_list.json")
result = voc_sanitize.reposition(result)
'''


tgt_glyph = lutl.GlyphStrawboss(glyphs = "data/hindi/hi_scripts.json")
en_glyph = lutl.GlyphStrawboss("en")
voc_sanitize = lutl.VocabSanitizer("data/hindi/mono/hi_words_sorted.json")

device = "cpu"

##=============== Models =======================================================

from tasks.rnn_xlit_runner import model

weight_path = "hypotheses/Training_hi_110/weights/Training_hi_110_model.pth"
weights = torch.load( weight_path, map_location=torch.device(device))

model.to(device)
model.load_state_dict(weights)
model.eval()

def inferencer(word, topk = 5):

    in_vec = torch.from_numpy(en_glyph.word2xlitvec(word)).to(device)
    ## change to active or passive beam
    p_out_list = model.active_beam_inference(in_vec, beam_width = topk)
    p_result = [ tgt_glyph.xlitvec2word(out.cpu().numpy()) for out in p_out_list]

    result = voc_sanitize.reposition(p_result)

    return result


##=============== Corr/ Emb Stacked

# ------------- Correction model -----------------------------------------------
''' Multinominal
from tasks.corr_xlit_runner import corr_model
corr_weight_path = "hypotheses/Training_mai_116_corr3_a/weights/Training_mai_116_corr3_a_corrnet.pth"

corr_weights = torch.load( corr_weight_path, map_location=torch.device(device))
corr_model.load_state_dict(corr_weights)
corr_model.eval()

hi_vocab = lutl.VocableStrawboss("data/konkani/gom_all_words_sorted.json")

'''

### -------------- Annoy based correction --------------------------------------
'''
import utilities.embed_utils as eutl

from tasks.emb_xlit_runner import emb_model
emb_weight_path = "hypotheses/Training_gom_emb5/weights/Training_gom_emb5_embnet.pth"

emb_weights = torch.load( emb_weight_path, map_location=torch.device(device))
emb_model.load_state_dict(emb_weights)
emb_model.eval()

## To Create fresh
# eutl.create_annoy_index_from_model(
#         voc_json_file = "data/konkani/gom_all_words_sorted.json",
#         glyph_obj = hi_glyph,
#         model_func = emb_model.get_word_embedding,
#         vec_sz = 512,
#         save_prefix= 'hypotheses/Training_gom_emb6/Gom_emb6')
# sys.exit()

annoy_obj = eutl.AnnoyStrawboss(
                voc_json_file = "data/konkani/gom_all_words_sorted.json",
                annoy_tree_path = "hypotheses/Training_gom_emb5/Gom_emb5_word_vec.annoy",
                vec_sz = 1024)
'''

def pred_contrive(corr_lst, pred_lst):
    out =[]
    for l in corr_lst:
        if (l not in out) and (l != "<UNK>"):
            out.append(l)
    for l in pred_lst:
        if l not in out:
            out.append(l)
    return out[:len(corr_lst)]

'''
def inferencer(word, topk = 5, knear = 1):

    in_vec = torch.from_numpy(en_glyph.word2xlitvec(word)).to(device)
    ## change to active or passive beam
    p_out_list = model.active_beam_inference(in_vec, beam_width = topk)
    p_result = [ hi_glyph.xlitvec2word(out.cpu().numpy()) for out in p_out_list]

    emb_list = [ emb_model.get_word_embedding(out) for out in p_out_list]
    c_result = [annoy_obj.get_nearest_vocab(emb, count = knear) for emb in emb_list ]
    c_result = sum(c_result, []) # delinieate 2d list

    #c_out_list = [ corr_model.inference(out) for out in out_list]
    #c_result = [ hi_vocab.get_word(out.cpu().numpy()) for out in c_out_list]

    result = pred_contrive(c_result, p_result)

    return result
'''


##=============== For Fused Variant
'''
from tasks.lm_fusion_runner import model
model.eval()
def inferencer(word, topk = 5):

    in_vec = torch.from_numpy(en_glyph.word2xlitvec(word)).to(device)

    p_out_list = model.basenet_inference(in_vec, beam_width = topk)
    # p_out_list.sort(reverse=True, key=model.lm_heuristics)
    p_result = [ hi_glyph.xlitvec2word(out.cpu().numpy()) for out in p_out_list]
    result = p_result

    return result

def lambda_experimenter(word, topk = 10):

    in_vec = torch.from_numpy(en_glyph.word2xlitvec(word)).to(device)

    ## [0]log_smx [0]pred_tnsrs
    p_out_list = model.basenet_inference(in_vec, beam_width = topk, heuristics = True)
    p_out_heur = []
    for out in p_out_list:
        prd_prob = float( out[0] )
        lm_prob = float( model.lm_heuristics(out[1]) )
        word = hi_glyph.xlitvec2word(out[1].cpu().numpy())
        p_out_heur.append( (word, prd_prob, lm_prob)  )

    return p_out_heur

'''

##==================

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


def infer_annoy_analytics(word, topk = 1, knear = 1):
    ''' Analytics with respect to Annoy usage
    '''

    in_vec = torch.from_numpy(en_glyph.word2xlitvec(word)).to(device)
    ## change to active or passive beam
    p_out_list = model.active_beam_inference(in_vec, beam_width = topk)
    p_result = [ hi_glyph.xlitvec2word(out.cpu().numpy()) for out in p_out_list]

    emb_list = [ emb_model.get_word_embedding(out) for out in p_out_list]

    c_result = []
    for i, emb in enumerate(emb_list):
        c_res, c_val = annoy_obj.get_nearest_vocab_details(emb, count = knear)
        c_result.append(c_res)
        LOG2CSV([word, i+1, p_result[i], c_res[0], c_val[0]], csv_file="Annoy_115e5_setup.csv")

    c_result = sum(c_result, []) # delinieate 2d list
    result = pred_contrive(c_result, p_result)
    return result


if __name__ == "__main__":
    while(1):
        a = input()
        result = inferencer(a)
        print(result)