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
from algorithms.recurrent_nets import EmbedSeqNet

##===== Init Setup =============================================================
INST_NAME = "Training_Test"

##------------------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'

LOG_PATH = "hypotheses/"+INST_NAME+"/" #+ "_emb"
WGT_PREFIX = LOG_PATH+"/weights/"+ INST_NAME
if not os.path.exists(LOG_PATH+"/weights"): os.makedirs(LOG_PATH+"/weights")

##===== Running Configuration =================================================

glyph_obj = lutl.GlyphStrawboss("gom")
# annoy_obj =lutl.AnnoyStrawboss( lang = "gom",
#                 voc_json_file = "data/konkani/gom_mini_list.json",
#                 hdf5_file = "data/konkani/Gom-vocab_mini_embeddings.hdf5",
#                 save_prefix = LOG_PATH,
#                 mode = "compose")

num_epochs = 1000
batch_size = 1
acc_grad = 1
learning_rate = 1e-3
pretrain_wgt_path = None

train_dataset = lutl.MonoCharLMData(glyph_obj,
                    data_file = "data/konkani/gom_mini_list.json",
                    padding = True,
                    )
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0)


val_dataloader = train_dataloader

# for i in range(10):
#     print(train_dataset.__getitem__(i))

##======== Model Configuration =================================================

vocab_dim = glyph_obj.size()
char_embed_dim = 300
hidden_dim = 300
rnn_type = 'lstm'
layers = 1
bidirectional = True
dropout = 0

emb_model = EmbedSeqNet( voc_dim = vocab_dim,
                        embed_dim = char_embed_dim,
                        hidden_dim = hidden_dim,
                        rnn_type = 'gru', layers = 1,
                        bidirectional = True,
                        dropout = 0, device = device)

emb_model = emb_model.to(device)

hi_emb_vecs = np.load("data/embeds/fasttext/hi_99_char_300d_fasttext.npy")
emb_model.embedding.weight.data.copy_(torch.from_numpy(hi_emb_vecs))


# emb_model = rutl.load_pretrained(emb_model,pretrain_wgt_path) #if path empty returns unmodified

##--------- Model Details ------------------------------------------------------
rutl.count_train_param(emb_model)
print(emb_model)
# sys.exit()

##======== Optimizer Zone ======================================================

criterion = torch.nn.CrossEntropyLoss()

def loss_estimator(pred, truth):
    """
    """
    mask = truth.ge(1).type(torch.FloatTensor).to(device)
    loss_ = criterion(pred, truth) * mask
    return torch.mean(loss_)


optimizer = torch.optim.AdamW(emb_model.parameters(), lr=learning_rate,
                             weight_decay=0)

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

#===============================================================================

if __name__ =="__main__":

    best_loss = float("inf")
    best_accuracy = 0
    for epoch in range(num_epochs):

        #-------- Training -------------------
        emb_model.train()
        acc_loss = 0
        running_loss = []

        for ith, (src, tgt, src_sz) in enumerate(train_dataloader):

            src = src.to(device)
            tgt = tgt.to(device)

            #--- forward ------
            output = emb_model(src = src, tgt = tgt, src_sz =src_sz)
            loss = loss_estimator(output, tgt) / acc_grad
            acc_loss += loss
            print(src, tgt)
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
        emb_model.eval()
        val_loss = 0
        val_accuracy = 0
        for jth, (v_src, v_tgt, v_src_sz) in enumerate(tqdm(val_dataloader)):
            v_src = v_src.to(device)
            v_tgt = v_tgt.to(device)
            with torch.no_grad():
                v_output = emb_model(src = v_src ,src_sz = v_src_sz)
                val_loss += loss_estimator(v_output, v_tgt)

                val_accuracy += rutl.accuracy_score(v_output, v_tgt, glyph_obj)
            # break
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
            torch.save(emb_model.state_dict(), WGT_PREFIX+"_embnet-{}.pth".format(epoch+1))
            LOG2CSV([epoch+1, val_loss.item(), val_accuracy.item()],
                    LOG_PATH+"bestCheckpoint.csv")

        # LR step
        # scheduler.step()





