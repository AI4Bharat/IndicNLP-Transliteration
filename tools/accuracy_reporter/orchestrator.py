""" Simple automation
To be run from Repo Root directory
"""
import os
import json
from tqdm import tqdm

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

def save_to_json(path, data_dict):
    with open(path ,"w", encoding = "utf-8") as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4, sort_keys=True,)

##------------------------------------------------------------------------------

def inference_looper(in_words):
    from timeit import timeit
    from tasks.infer_engine import inferencer
    out_dict = {}
    for i in tqdm(in_words):
        out_dict[i] = inferencer(i)
    return out_dict

ROOT_PATH= ""
SAVE_DIR = "tools/accuracy_reporter/temp-train105"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)

files = [
    ROOT_PATH+"tools/accuracy_reporter/temp-data/EnHi_news18_dev.json",
    ROOT_PATH+"tools/accuracy_reporter/temp-data/EnHi_fire13_dev.json",
    # ROOT_PATH+"tools/accuracy_reporter/temp-data/EnHi_varnam_test.json",
    # ROOT_PATH+"tools/accuracy_reporter/temp-data/EnHi_varnam_special_test.json",
]

if __name__ == "__main__":

    for fi in files:
        words = get_from_json(fi, "key")
        out_dict = inference_looper(words)
        sv_path = os.path.join(SAVE_DIR, "pred_"+os.path.basename(fi) )
        save_to_json(sv_path, out_dict)

        gt_json = fi
        pred_json = sv_path

        save_prefix = os.path.join(SAVE_DIR, os.path.basename(fi).replace(".json", ""))
        run_accuracy_news = "python tools/accuracy_reporter/accuracy_news.py --gt-json {} --pred-json {} --save-output-csv {}_scores.csv".format(
                        gt_json, pred_json, save_prefix)

        os.system(run_accuracy_news)
