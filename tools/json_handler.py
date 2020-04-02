import os
import sys
import json

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

def save_to_json(path, data_dict):
    with open(path ,"w", encoding = "utf-8") as f:
        json.dump(data_dict, f, ensure_ascii=False, indent=4, sort_keys=True,)


def convert_ezann_to_xlit(file_path):
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

    save_path = os.path.dirname(file_path)
    with open(save_path+"/out_xlit_format.json" ,"w", encoding = "utf-8") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=4, sort_keys=True,)



if __name__ == "__main__":

    # convert_ezann_to_xlit("/home/jgeob/Downloads/Konkani_Literation_A_complete.json")
    pass