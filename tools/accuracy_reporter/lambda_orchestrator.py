""" Simple automation
To be run from Repo Root directory
"""
import os
import sys
import json
from numpy import arange as np_arange
from tqdm import tqdm

def save_to_json(path, data_dict):
    with open(path ,"w", encoding = "utf-8") as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4, sort_keys=True,)

def toggle_json(read_path, save_prefix=""):
    with open(read_path, 'r', encoding = "utf-8") as f:
        data = json.load(f)

    tog_dict = dict()
    for d in data.keys():
        for v in data[d]:
            tog_dict[v] = set()

    for d in data.keys():
        for v in data[d]:
            tog_dict[v].add(d)

    for t in tog_dict.keys():
        tog_dict[t] = list(tog_dict[t])

    save_file = save_prefix+"/Toggled-"+ os.path.basename(read_path)
    with open(save_file,"w", encoding = "utf-8") as f:
        json.dump(tog_dict, f, ensure_ascii=False, indent=4, sort_keys=True,)

    return save_file

def get_from_json(path, ret_data = "key"):
    with open(path, 'r', encoding = "utf-8") as f:
        data = json.load(f)

    if ret_data == "key":
        out = list(data.keys())
    elif ret_data == "value":
        temp = data.values()
        temp = { i for t in temp for i in t }
        out = list(temp)
    elif ret_data == "both":
        out = []
        for k in data.keys():
            for v in data[k]:
                out.append([k,v])

    return sorted(out)

def merge_pred_truth_json(pred_path, truth_path ):
    with open(pred_path) as f:
        pred_data = json.load(f)
    with open(truth_path) as f:
        truth_data = json.load(f)
    new_dict = {}
    for k in truth_data:
        new_dict[k] = {"gtruth": truth_data[k], "prediction": pred_data[k] }

    save_file = os.path.dirname(pred_path) +"/Merged-truth_"+ os.path.basename(pred_path)
    save_to_json(save_file, new_dict)

##------------------------------------------------------------------------------

def inference_looper(in_words, topk =10):
    from tasks.infer_engine import lambda_experimenter
    heur_dict = {}
    for w in tqdm(in_words):
        heur_dict[w] = lambda_experimenter(w, topk=topk)

    save_to_json( "Heur_data.json", heur_dict )
    sys.exit()
    return heur_dict


def sort_by_lambda(heur_dict, lmbd = 1):
    from tasks.infer_engine import hi_glyph

    def _energy(tup):
        egr = lmbd*tup[1] + (1-lmbd)*tup[2]
        return egr

    out_dict = {}
    for w in heur_dict:
        word_heur = heur_dict[w]
        word_heur.sort(reverse=True, key=_energy)
        out_dict[w] = [ tup[0] for tup in word_heur]

    return out_dict



ROOT_PATH= ""
files = [

#    ROOT_PATH+"data/konkani/GomEn_ann1_copy.json",
    # ROOT_PATH+"data/konkani/GomEn_ann1_train.json",
    # ROOT_PATH+"data/konkani/GomEn_ann1_valid.json",
    # ROOT_PATH+"data/konkani/GomEn_ann1_test.json",

    # ROOT_PATH+"data/maithili/MaiEn_ann1_train.json",
    ROOT_PATH+"data/maithili/MaiEn_ann1_valid.json",
    # ROOT_PATH+"data/maithili/MaiEn_ann1_test.json",

    # ROOT_PATH+"data/marathi/MrEn_dakshina_train.json",
    # ROOT_PATH+"data/marathi/MrEn_dakshina_valid.json",
    # ROOT_PATH+"data/marathi/MrEn_dakshina_test.json",

    # ROOT_PATH+"data/hindi/HiEn_xlit_valid.json",
    # ROOT_PATH+"data/hindi/HiEn_news18_dev.json",
    # ROOT_PATH+"data/hindi/HiEn_dakshina_test.json",
    # ROOT_PATH+"data/hindi/HiEn_ann1_test.json",
    # ROOT_PATH+"data/hindi/zHiEn_merged_test.json",

]

SAVE_DIR = "hypotheses/ACCL/Lambda_exp_ZZ/"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

if __name__ == "__main__":

    for fi in files:
        tfi =  toggle_json(fi, save_prefix=SAVE_DIR)
        words = get_from_json(tfi, "key")
        heur_dict = inference_looper(words, topk = 10)

        for lmbd in np_arange(0,1.01, 0.1):
            out_dict = sort_by_lambda(heur_dict, lmbd = lmbd)

            pred_path = os.path.join(SAVE_DIR, "pred_{}_".format(lmbd)+os.path.basename(fi) )
            save_to_json(pred_path, out_dict)

            gt_json = tfi
            pred_json = pred_path
            save_prefix = os.path.join(SAVE_DIR, os.path.basename(fi).replace(".json", ""))

            for topk in [1]:
                ## GT json file passed to below script must be in { En(input): [NativeLang (predict)] } format
                run_accuracy_news = "( echo {} Lambda:{} && python tools/accuracy_reporter/accuracy_news.py --gt-json {} --pred-json {} --topk {} --save-output-csv {}_top{}-scores.csv ) | tee -a {}/Summary.txt".format(
                                os.path.basename(fi), lmbd,
                                gt_json, pred_json, topk,
                                save_prefix, topk, SAVE_DIR )

                os.system(run_accuracy_news)
