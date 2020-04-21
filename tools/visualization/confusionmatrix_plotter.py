import sys
import os
import json
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

def list_system_font():
    from matplotlib import font_manager
    font_paths = font_manager.findSystemFonts()
    font_objects = font_manager.createFontList(font_paths)
    font_names = [f.name for f in font_objects]
    print (sorted(font_names))


def find_best_reference(pred_list, truth_list):
    def LCS_length(s1, s2):
        m = len(s1)
        n = len(s2)
        # An (m+1) times (n+1) matrix
        C = [[0] * (n+1) for i in range(m+1)]
        for i in range(1, m+1):
            for j in range(1, n+1):
                if s1[i-1] == s2[j-1]:
                    C[i][j] = C[i-1][j-1] + 1
                else:
                    C[i][j] = max(C[i][j-1], C[i-1][j])
        return C[m][n]

    best_ref = truth_list[0]
    best_cand = pred_list[0]
    best_ref_lcs = LCS_length(pred_list[0], truth_list[0])
    for cand in pred_list[1:]:
        for ref in truth_list[1:]:
            lcs = LCS_length(cand, ref)
            if (len(ref) - 2*lcs) < (len(best_ref) - 2*best_ref_lcs):
                best_ref = ref
                best_cand = cand
                best_ref_lcs = lcs

    return best_cand, best_ref


def generate_confusion(pred_file, truth_file, vocab):
    with open(pred_file) as f:
        pred_data = json.load(f)
    with open(truth_file) as f:
        truth_data = json.load(f)

    conf_arr = np.zeros((len(vocab), len(vocab)), dtype = np.int64 )
    for k in pred_data:
        pred_list = pred_data[k]
        truth_list = truth_data[k]
        best_pred, best_truth = find_best_reference(pred_list, truth_list)

        max_len_ = max(len(best_pred), len(best_truth))
        pred = best_pred + ( "_" * (max_len_-len(best_pred)) )
        truth = best_truth + ( "_" * (max_len_-len(best_truth)) )

        for p,t in zip(pred, truth):
            p_idx = vocab.index(p)
            t_idx = vocab.index(t)
            conf_arr[p_idx][t_idx] += 1

    return conf_arr


def plot_confusion(array, axis_title, save_prefix=""):
    plt.figure(figsize = (256,200))
    sns.set(font = "Lohit Devanagari", font_scale = 2 )
    conf_plot = sns.heatmap(array, annot=True,
                      xticklabels = axis_title,
                      yticklabels = axis_title)

    conf_plot.yaxis.set_ticklabels(conf_plot.yaxis.get_ticklabels(),
                                    ha='right',rotation=0, fontsize=60)
    conf_plot.xaxis.set_ticklabels(conf_plot.xaxis.get_ticklabels(),
                                    ha='right',rotation=0, fontsize=60)


    plt.ylabel('Predicted Character', fontsize =250)
    plt.xlabel('True Character', fontsize =250)

    conf_plot.figure.savefig( save_prefix+"plot.png")



dgri = ["_"] + [chr(alpha) for alpha in range(2304, 2432)] + [
    chr(0x200c), # ZeroWidth-NonJoiner U+200c
    chr(0x200d), # ZeroWidthJoiner U+200d
]

SAVE_DIR = "tools/visualization/logs/"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
if __name__ == "__main__":

    pred_file = "/home/jgeob/Downloads/pred_EnKnk_ann1_test.json"
    truth_file ="/home/jgeob/Downloads/EnKnk_ann1_test.json"

    arr = generate_confusion(pred_file, truth_file, dgri)
    arr = np.clip(arr, a_min = 0, a_max = 200)
    #np.set_printoptions(threshold=sys.maxsize)
    #print(arr)

    plot_confusion(arr, dgri, save_prefix= SAVE_DIR+"confusion")