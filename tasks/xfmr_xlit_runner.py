''' RNN Seq2Seq (Encoder-Decoder) training setup
'''

import torch
from torch.utils.data import DataLoader
import numpy as np
import os
import sys
from tqdm import tqdm
import utilities.running_utils as rutl
from utilities.lang_data_utils import XlitData, GlyphStrawboss, MonoLMData
from utilities.logging_utils import LOG2CSV
from algorithms.transformer_nets import XFMR_Neophyte


##===== Init Setup =============================================================
MODE = rutl.RunMode.train
INST_NAME = "Training_Test"

##------------------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

LOG_PATH = "hypotheses/"+INST_NAME+"/"
WGT_PREFIX = LOG_PATH+"weights/"+INST_NAME
if not os.path.exists(LOG_PATH+"weights"): os.makedirs(LOG_PATH+"weights")

##===== Running Configuration =================================================

src_glyph = GlyphStrawboss("en")
tgt_glyph = GlyphStrawboss("hi")

num_epochs = 1000
batch_size = 10
acc_grad = 1
learning_rate = 1e-4
pretrain_wgt_path = None
max_char_size = 50

train_dataset = XlitData( src_glyph_obj = src_glyph, tgt_glyph_obj = tgt_glyph,
                        json_file='data/checkup-test.json', file_map = "LangEn",
                        padding=True, max_seq_size=max_char_size)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0)

val_dataset = XlitData( src_glyph_obj = src_glyph, tgt_glyph_obj = tgt_glyph,
                        json_file='data/checkup-test.json', file_map = "LangEn",
                        padding=True, max_seq_size=max_char_size)

val_dataloader = DataLoader(val_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0)

# for i in range(len(train_dataset)):
#     print(train_dataset.__getitem__(i))

##===== Model Configuration =================================================

input_dim = src_glyph.size()
output_dim = tgt_glyph.size()
emb_vec_dim = 512
n_layers = 3
attention_head = 16
feedforward_dim = 512
m_dropout = 0

model = XFMR_Neophyte(input_vcb_sz = input_dim, output_vcb_sz = output_dim,
                    emb_dim = emb_vec_dim,
                    n_layers = n_layers,
                    attention_head = attention_head, feedfwd_dim = feedforward_dim,
                    dropout = 0,
                    device = device)
model = model.to(device)

# model = rutl.load_pretrained(model,pretrain_wgt_path) #if path empty returns unmodified

## ----- Load Embeds -----

# hi_emb_vecs = np.load("data/embeds/hi_char_512_ftxt.npy")
# model.decoder.embedding.weight.data.copy_(torch.from_numpy(hi_emb_vecs))

en_emb_vecs = np.load("data/embeds/en_char_512_ftxt.npy")
model.in2embed.weight.data.copy_(torch.from_numpy(en_emb_vecs))


##------ Model Details ---------------------------------------------------------
rutl.count_train_param(model)
print(model)
sys.exit()

##====== Optimizer Zone ===================================================================


criterion = torch.nn.CrossEntropyLoss()

def loss_estimator(pred, truth):
    """ Only consider non-zero inputs in the loss; mask needed
    pred: batch
    """
    # pred = pred[:,:,1:]
    # truth = truth[:,1:]
    mask = truth.ge(1).type(torch.FloatTensor).to(device)
    loss_ = criterion(pred, truth) * mask
    return torch.mean(loss_)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)


#===============================================================================

if __name__ =="__main__":

    best_loss = float("inf")
    best_accuracy = 0
    for epoch in range(num_epochs):

        #-------- Training -------------------
        model.train()
        acc_loss = 0
        running_loss = []

        for ith, (src, tgt, src_sz) in enumerate(train_dataloader):

            src = src.to(device)
            tgt = tgt.to(device)

            #--- forward ------
            output = model(src = src, src_sz = src_sz)
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
                #break

        LOG2CSV(running_loss, LOG_PATH+"trainLoss.csv")

        #--------- Validate ---------------------
        model.eval()
        val_loss = 0
        val_accuracy = 0
        for jth, (v_src, v_tgt, v_src_sz) in enumerate(tqdm(val_dataloader)):
            v_src = v_src.to(device)
            v_tgt = v_tgt.to(device)
            with torch.no_grad():
                v_output = model(src = v_src, src_sz = v_src_sz)
                val_loss += loss_estimator(v_output, v_tgt)
                val_accuracy += rutl.accuracy_score(v_output, v_tgt, tgt_glyph)
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
            torch.save(model.state_dict(), WGT_PREFIX+"_model-{}.pth".format(epoch+1))
            LOG2CSV([epoch+1, val_loss.item(), val_accuracy.item()],
                    LOG_PATH+"bestCheckpoint.csv")
