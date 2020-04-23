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
    '''
    Returns a pandas dataframe with confusion matrix values
    '''
    with open(pred_file) as f:
        pred_data = json.load(f)
    with open(truth_file) as f:
        truth_data = json.load(f)

    conf_df = pd.DataFrame(0, columns=vocab, index=vocab)
    for k in pred_data:
        pred_list = pred_data[k]
        truth_list = truth_data[k]
        best_pred, best_truth = find_best_reference(pred_list, truth_list)

        max_len_ = max(len(best_pred), len(best_truth))
        pred = best_pred + ( "_" * (max_len_-len(best_pred)) )
        truth = best_truth + ( "_" * (max_len_-len(best_truth)) )

        for p,t in zip(pred, truth):
            conf_df.loc[p][t] += 1

    return conf_df


def plot_confusion(conf_df, show_chars = None, save_prefix=""):
    plot_df = conf_df

    ## Drop rows/columns full of zeros
    plot_df = plot_df.loc[:,(df != 0).any(axis=0)] #remove columns
    plot_df = plot_df.loc[(df!=0).any(axis=1), :] #remove rows

    ## Remove unnecessary char counts
    if show_chars:
        plot_df = plot_df.drop("_", axis = 0)
        plot_df = plot_df.drop("_", axis = 1)

    if show_chars:
        dfrows = list(plot_df.index.values)
        dfcols = list(plot_df.columns.values)
        plot_df.loc["other",:] = 0
        plot_df.loc[:, "other"] = 0

        for r in dfrows:
            if r not in show_chars:
                plot_df.loc["other",:] += plot_df.loc[r,:]
                plot_df = plot_df.drop(r, axis = 0)

        for c in dfcols:
            if c not in show_chars:
                plot_df.loc[:, "other"] += plot_df.loc[:, c]
                plot_df = plot_df.drop(c, axis = 1)

    ## Clip Values
    plot_df = plot_df.clip(0, 200)

    ##------
    font_sz = 10; fig_sz = 20
    plt.rcParams['figure.constrained_layout.use'] = True
    sns.set(font = "Lohit Devanagari", font_scale = 1 )
    plt.figure(figsize = (fig_sz,fig_sz))

    conf_plot = sns.heatmap(plot_df, annot=False)

    conf_plot.yaxis.set_ticklabels(conf_plot.yaxis.get_ticklabels(),
                                    ha='right', rotation=0, fontsize = font_sz)
    conf_plot.xaxis.set_ticklabels(conf_plot.xaxis.get_ticklabels(),
                                    ha='left', rotation=0, fontsize = font_sz)

    # conf_plot.tick_params(axis='both', which='major', pad=10)
    plt.ylabel('Predicted Character', fontsize = font_sz)
    plt.xlabel('True Character', fontsize = font_sz)
    plt.title ('ALL', fontsize = font_sz)
    # plt.show()

    conf_plot.figure.savefig( save_prefix+"plot.png")


## -----------------------------------------------------------------------------

dgri_unicodes =  [chr(alpha) for alpha in range(2304, 2432)] + [
    chr(0x200c), # ZeroWidth-NonJoiner U+200c
    chr(0x200d), # ZeroWidthJoiner U+200d
    "_", # empty pading
]

dgri_seg = {
    "vowel": ['अ', 'आ', 'इ', 'ई', 'उ', 'ऊ','ऍ', 'ऎ', 'ए', 'ऐ', 'ऑ', 'ऒ', 'ओ', 'औ','ऋ','ॠ','ऌ','ॡ'],
    "cons" : ['क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण',
              'त', 'थ', 'द', 'ध', 'न', 'ऩ', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ऱ', 'ल',
              'ळ', 'ऴ', 'व', 'श', 'ष', 'स', 'ह', 'क़', 'ख़', 'ग़', 'ज़', 'ड़', 'ढ़', 'फ़', 'य़'],
    "vow_symb": [ '्', 'ा', 'ि', 'ी', 'ु', 'ू', 'ॅ', 'ॆ', 'े', 'ै', 'ॉ', 'ॊ', 'ो', 'ौ', 'ृ', 'ॄ', 'ॢ', 'ॣ']
}

SAVE_DIR = "hypotheses/training_temp"  +"/viz_log/"
if not os.path.exists(SAVE_DIR): os.makedirs(SAVE_DIR)
if __name__ == "__main__":

    truth_file = "tools/accuracy_reporter/logs/EnLang-data/EnKnk_ann1_test.json"
    pred_file ="hypotheses/training_knk_103/acc_log/pred_EnKnk_ann1_test.json"

    df = generate_confusion(pred_file, truth_file, dgri_unicodes)

    plot_confusion(df,  dgri_seg["vow_symb"],
                    save_prefix= SAVE_DIR+"confusion")
