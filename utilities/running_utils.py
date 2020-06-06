import enum
import sys
import os
import json
import torch
import seaborn as sns
import matplotlib.pyplot as plt

class RunMode(enum.Enum):
    train = "training"
    infer = "inference"
    test  = "testing"

def load_pretrained(model, weight_path):
    if not weight_path:
        return model

    pretrain_dict = torch.load(weight_path)
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    print("Pretrained layers Loaded:", pretrain_dict.keys())
    model_dict.update(pretrain_dict)
    model.load_state_dict(model_dict)

    return model


def count_train_param(model):
    train_params_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('The model has {} trainable parameters'.format(train_params_count))
    return train_params_count


def accuracy_score(pred_tnsr, tgt_tnsr):
    '''Simple accuracy calculation for training
    tgt-arr, pred-arr: torch tensors
    '''
    if torch.equal(pred_tnsr.type(torch.long), tgt_tnsr):
        return torch.tensor(1)
    else:
        return torch.tensor(0)

def attention_weight_plotter(out_word, in_word, attention_array, save_path = ""):
    '''
    Plot heat map of attention weights
    '''
    font_sz = 10
    plt.figure(figsize = (10,7))
    sns.set(font = "Lohit Devanagari", )
    conf_plot = sns.heatmap(attention_array, annot=False,
                      xticklabels = ["$"]+ list(in_word) + ["#"],
                      yticklabels = out_word)

    conf_plot.yaxis.set_ticklabels(conf_plot.yaxis.get_ticklabels(),
                                    rotation=0, fontsize = font_sz)
    conf_plot.xaxis.set_ticklabels(conf_plot.xaxis.get_ticklabels(),
                                    rotation=0, fontsize = font_sz)
    plt.xlabel('Pred: '+ out_word, fontsize = font_sz)

    conf_plot.figure.savefig( save_path + "/"+in_word+"_attention.png")

    plt.clf()


def compose_corr_dataset(pred_file = "", truth_file = "",
                         save_path = ""  ):
    """
    Function to create Json for Correction Network from the truth and predition of models
    Return: Path of the composed file { Output: [Input] }
    pred_file: EnLang
    truth_file: LangEn
    """
    pred_dict = json.load(open(pred_file))
    truth_dict = json.load(open(truth_file))


    out_dict = {}
    for k in truth_dict:
        temp_set = set()
        for v in truth_dict[k]:
            temp_set.update(pred_dict[v])
        out_dict[k] = list(temp_set)

    save_file = save_path + "Corr_set_"+os.path.basename(truth_file)
    with open(save_file ,"w", encoding = "utf-8") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=4, sort_keys=True,)

    return save_file


