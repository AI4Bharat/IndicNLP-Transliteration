import os
from os.path import join

LOSS_FILE = 'train_loss.txt'
SCORES_FILE = 'scores.txt'
MODEL_FILE = 'model-e%03d.pt'
BEST_MODEL_FILE = 'model-best.pt'

def get_train_files(ckpt_folder):
    if not os.path.isdir(ckpt_folder):
        os.makedirs(ckpt_folder, exist_ok=True)
    loss_file = join(ckpt_folder, LOSS_FILE)
    scores_file = join(ckpt_folder, SCORES_FILE)
    model_file = join(ckpt_folder, MODEL_FILE)
    return loss_file, scores_file, model_file