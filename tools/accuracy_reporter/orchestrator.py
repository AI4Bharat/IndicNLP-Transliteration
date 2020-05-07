""" Simple automation
To be run from Repo Root directory
"""
import os
import sys
import json
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

    save_file = save_prefix+"Toggled-"+ os.path.basename(path)
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

def inference_looper(in_words, topk = 3):
    from tasks.infer_engine import inferencer
    out_dict = {}
    for i in tqdm(in_words):
        out_dict[i] = inferencer(i, topk=topk)
    return out_dict

def vocab_sanity_runner(pred_json, voc_json):
    '''
    Re-Clean Prediction json based on the known Vocabulary of the langauge
    '''
    from utilities.lang_data_utils import VocabSanitizer
    voc_sanity = VocabSanitizer(voc_json)

    pred_dict = json.load(open(pred_json))
    out_dict = {}
    for k in pred_dict.keys():
        out_dict[k] = voc_sanity.remove_astray(pred_dict[k])

    return out_dict


ROOT_PATH= ""
files = [
    # ROOT_PATH+"tools/accuracy_reporter/logs/EnLang-data/EnHi_news18_dev.json",
    # ROOT_PATH+"tools/accuracy_reporter/logs/EnLang-data/EnHi_fire13_dev.json",
    # ROOT_PATH+"tools/accuracy_reporter/logs/EnLang-data/EnHi_varnam_test.json",
    # ROOT_PATH+"tools/accuracy_reporter/logs/EnLang-data/EnHi_varnam_special_test.json",
    ROOT_PATH+"tools/accuracy_reporter/logs/EnLang-data/EnKnk_ann1_test.json"
]

SAVE_DIR = "hypotheses/training_temp/acc_log"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

if __name__ == "__main__":

    for fi in files:
        words = get_from_json(fi, "key")
        out_dict = inference_looper(words, topk = 10)
        ## Testing with LM adjustments
        # out_dict = vocab_sanity_runner( "hypotheses/prediction.json",
        #                                 "data/word_list.json")

        sv_path = os.path.join(SAVE_DIR, "pred_"+os.path.basename(fi) )
        save_to_json(sv_path, out_dict)

        gt_json = fi
        pred_json = sv_path
        save_prefix = os.path.join(SAVE_DIR, os.path.basename(fi).replace(".json", ""))

        for topk in [10, 5, 3, 2, 1]:
            ## GT json file passed to below script must be in { En(input): [NativeLang (predict)] } format
            run_accuracy_news = "python tools/accuracy_reporter/accuracy_news.py --gt-json {} --pred-json {} --topk {} --save-output-csv {}_top{}-scores.csv | tee -a {}/Summary.txt".format(
                            gt_json, pred_json, topk, save_prefix, topk, SAVE_DIR )

            os.system(run_accuracy_news)
