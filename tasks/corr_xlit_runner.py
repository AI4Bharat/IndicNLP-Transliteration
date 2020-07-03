''' RNN Seq2Seq (Encoder-Decoder) training setup
'''

import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from tqdm import tqdm
import utilities.running_utils as rutl
import utilities.lang_data_utils as lutl
from utilities.logging_utils import LOG2CSV
from algorithms.recurrent_nets import VocabCorrectorNet
from algorithms.transformer_nets import XFMR_CorrectorNet

##===== Init Setup =============================================================
INST_NAME = "Training_Test"

##------------------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

LOG_PATH = "hypotheses/"+INST_NAME+"/"
WGT_PREFIX = LOG_PATH+"weights/"+ INST_NAME + "_corr"
if not os.path.exists(LOG_PATH+"weights"): os.makedirs(LOG_PATH+"weights")

##===== Running Configuration =================================================

glyph_obj = lutl.GlyphStrawboss("hi")
vocab_obj = lutl.VocableStrawboss("data/maithili/mai_all_words_sorted.json")

num_epochs = 1000
batch_size = 2
acc_grad = 1
learning_rate = 1e-3
pretrain_wgt_path = None

train_dataset = lutl.MonoVocabLMData( glyph_obj, vocab_obj,
                    json_file = "data/maithili/mai_all_words_sorted.json",
                    input_type = "compose",
                    padding = True,
                    max_seq_size = 50,)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0)


val_file = lutl.compose_corr_dataset(  pred_file= "hypotheses/training_mai_103/acc_train_log/pred_EnMai_ann1_valid.json",
                                    truth_file= "data/maithili/MaiEn_ann1_valid.json",
                                    save_path= LOG_PATH)

val_dataset = lutl.MonoVocabLMData( glyph_obj, vocab_obj,
                    json_file = val_file, ## <<<<<<<<<<<<<<<<
                    input_type = "readfromfile",
                    padding = True,
                    max_seq_size = 50,)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0)

test_file = lutl.compose_corr_dataset(  pred_file= "hypotheses/training_mai_103/acc_train_log/pred_EnMai_ann1_test.json",
                                    truth_file= "data/maithili/MaiEn_ann1_test.json",
                                    save_path= LOG_PATH)

# for i in range(len(train_dataset)):
#     print(train_dataset.__getitem__(i))

##======== Model Configuration =================================================

input_dim = glyph_obj.size()
output_dim = vocab_obj.size()
char_embed_dim = 512
hidden_dim = 1024
rnn_type = 'lstm'
layers = 1
bidirectional = True
dropout = 0

corr_model = VocabCorrectorNet(input_dim = input_dim, output_dim = output_dim,
                    char_embed_dim = char_embed_dim, hidden_dim = hidden_dim,
                    rnn_type = rnn_type, layers = layers,
                    bidirectional = bidirectional,
                    dropout = 0,
                    device = device)

corr_model = corr_model.to(device)

hi_emb_vecs = np.load("data/embeds/hi_char_512_ftxt.npy")
corr_model.embedding.weight.data.copy_(torch.from_numpy(hi_emb_vecs))


# corr_model = rutl.load_pretrained(model,pretrain_wgt_path) #if path empty returns unmodified

##--------- Model Details ------------------------------------------------------
rutl.count_train_param(corr_model)
print(corr_model)
# sys.exit()

##======== Optimizer Zone ======================================================

criterion = torch.nn.CrossEntropyLoss()

def loss_estimator(pred, truth):
    """
    """
    loss_ = criterion(pred, truth)

    return torch.mean(loss_)


optimizer = torch.optim.AdamW(corr_model.parameters(), lr=learning_rate,
                             weight_decay=0)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#===============================================================================

if __name__ =="__main__":

    best_loss = float("inf")
    best_accuracy = 0
    for epoch in range(num_epochs):

        #-------- Training -------------------
        corr_model.train()
        acc_loss = 0
        running_loss = []

        for ith, (src, tgt, src_sz) in enumerate(train_dataloader):

            src = src.to(device)
            tgt = tgt.to(device)

            #--- forward ------
            output = corr_model(src = src, src_sz =src_sz)
            loss = loss_estimator(output, tgt) / acc_grad
            acc_loss += loss

            #--- backward ------
            loss.backward()
            if ( (ith+1) % acc_grad == 0):
                optimizer.step()
                optimizer.zero_grad()

                print('epoch[{}/{}], MiniBatch-{} loss:{:.4f}'
                    .format(epoch+1, num_epochs, (ith+1)//acc_grad, acc_loss.data))
                running_loss.append(acc_loss.item())
                acc_loss=0
                # break

        LOG2CSV(running_loss, LOG_PATH+"trainLoss.csv")

        #--------- Validate ---------------------
        corr_model.eval()
        val_loss = 0
        val_accuracy = 0
        for jth, (v_src, v_tgt, v_src_sz) in enumerate(tqdm(val_dataloader)):
            v_src = v_src.to(device)
            v_tgt = v_tgt.to(device)
            with torch.no_grad():
                v_output = corr_model(src = v_src ,src_sz = v_src_sz)
                val_loss += loss_estimator(v_output, v_tgt)

                val_accuracy += rutl.accuracy_score_multinominal(v_output, v_tgt, vocab_obj)
            #break
        val_loss = val_loss / len(val_dataloader)
        val_accuracy = val_accuracy / len(val_dataloader)

        print('epoch[{}/{}], [-----TEST------] loss:{:.4f}  Accur:{:.4f}'
              .format(epoch+1, num_epochs, val_loss.data, val_accuracy.data))
        LOG2CSV([val_loss.item(), val_accuracy.item()],
                    LOG_PATH+"valLoss.csv")

        #-------- save Checkpoint -------------------
        if val_accuracy > best_accuracy:
        # if val_loss < best_loss:
            print("***saving best optimal state [Loss:{}] ***".format(val_loss.data))
            best_loss = val_loss
            best_accuracy = val_accuracy
            torch.save(corr_model.state_dict(), WGT_PREFIX+"_corrnet-{}.pth".format(epoch+1))
            LOG2CSV([epoch+1, val_loss.item(), val_accuracy.item()],
                    LOG_PATH+"bestCheckpoint.csv")

        # LR step
        # scheduler.step()





