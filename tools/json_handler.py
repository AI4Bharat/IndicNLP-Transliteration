import os
import sys
import json

def save_to_json(path, data_dict):
    with open(path ,"w", encoding = "utf-8") as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4, sort_keys=True,)

def toggle_json_xlit(read_path, save_prefix=""):
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

def get_from_json_xlit(path, ret_data = "key"):
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

def merge_xlit_jsons(filepath_list, save_path = ""):

    data_list = []
    for fpath in filepath_list:
        with open(fpath, 'r', encoding = "utf-8") as f:
            data_list.append(json.load(f))

    whole_dict = dict()
    for dat in data_list:
        for dk in dat:
            whole_dict[dk] = set()

    for dat in data_list:
        for dk in dat:
            whole_dict[dk].update(dat[dk])

    for k in whole_dict:
        whole_dict[k] = list(whole_dict[k])

    with open(save_path+"merged_file.json","w", encoding = "utf-8") as f:
        json.dump(whole_dict, f, ensure_ascii=False, indent=4, sort_keys=True,)


def convert_ezann_to_xlit(file_path, save_path = ""):
    """
    String Processing:
    1.Converts upper to lower case
    """
    data = []
    with open(file_path) as f:
        for line in f:
            data.append(json.loads(line))

    out_dict = dict()
    for it in data:
        out_dict[it["content"]] = set()

    for it in data:
        ann = it["annotation"].lower()
        ann = ann.replace("\\n", ",").replace("\n", ",")
        ann = ann.split(",")
        out_dict[it["content"]].update(ann)

    for d in out_dict:
        out_dict[d] = list(out_dict[d])

    with open(save_path+"out_xlit_format.json" ,"w", encoding = "utf-8") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=4, sort_keys=True,)


def xlitjson_to_mosestxt(read_file, save_prefix = "moses_format"):
    """ For Converting JSON to Moses-SMT input file format of Trasliteration data
        (Character level spliting for Transliteration)
    """
    def char_token(word):
        unis = list(word)
        outword = ""
        for u in unis:
            outword += u+" "
        return outword[:-1]
    #--------------------------------

    with open(read_file) as f:
        data_dict = json.load(f)

    sr_file = open(save_prefix+".sr", "w")
    tg_file = open(save_prefix+".tg", "w")
    for key in data_dict.keys():
        for word in data_dict[key]:
            sr_file.write(char_token(key)+"\n")
            tg_file.write(char_token(word)+"\n")
    print("Complete Creating Translit-Moses Format")

if __name__ == "__main__":

    # convert_ezann_to_xlit("/home/jgeob/Downloads/Konkani_Literation_A_complete.json")

    # ## Merge JSON
    # files = ["/home/jgeob/quater_ws/transLit/IndianNLP-Transliteration/data/HiEn_fire13_train.json",
    # "/home/jgeob/quater_ws/transLit/IndianNLP-Transliteration/data/HiEn_news18_train.json"]

    # merge_xlit_jsons(files, '/home/jgeob/quater_ws/transLit/IndianNLP-Transliteration/data/')

    ##
    # xlitjson_to_mosestxt("data/konkani/KnkEn_ann1_test.json")
    pass