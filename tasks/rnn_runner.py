'''
'''

import torch
from torch import nn
from torch.utils.data import DataLoader
import os
import sys
from tqdm import tqdm
import utilities.running_utils as rutl
from utilities.lang_data_utils import XlitData, GlyphStrawboss
from utilities.logging_utils import LOG2CSV
from algorithms.recurrent_nets import Encoder, Decoder, Seq2Seq


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

paralleize_load = True
num_epochs = 1000
batch_size = 1024
acc_grad = 1
learning_rate = 1e-4
teacher_forcing, teach_force_till = 0.50, 3
pretrain_wgt_path = None

train_dataset = XlitData( src_glyph_obj = src_glyph, tgt_glyph_obj = tgt_glyph,
                        json_file='data/checkup-train.json', file_map = "LangEn",
                        padding=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0)

val_dataset = XlitData( src_glyph_obj = src_glyph, tgt_glyph_obj = tgt_glyph,
                        json_file='data/checkup-test.json', file_map = "LangEn",
                        padding=True)
val_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0)

# for i in range(len(train_dataset)):
#     print(train_dataset.__getitem__(i))

##===== Model Configuration =================================================

input_dim = src_glyph.size()
output_dim = tgt_glyph.size()
enc_emb_dim = 256
dec_emb_dim = 256
hidden_dim = 512
n_layers = 3
m_dropout = 0

enc = Encoder(  input_dim= input_dim, enc_embed_dim = enc_emb_dim,
                hidden_dim= hidden_dim,
                enc_layers= n_layers,
                enc_dropout= m_dropout, device = device)
dec = Decoder(  output_dim= output_dim, dec_embed_dim = dec_emb_dim,
                hidden_dim= hidden_dim,
                dec_layers= n_layers,
                dec_dropout= m_dropout,  device = device)

model = Seq2Seq(enc, dec, device=device)

if (torch.cuda.device_count() > 1) and paralleize_load :
    print("Multiple GPUs", torch.cuda.device_count(), "Parallelizing")
    model = nn.DataParallel(model)
else:
    print("Running in Single GPU")
    paralleize_load = False

model = model.to(device)

# model = rutl.load_pretrained(model,pretrain_wgt_path) #if path empty returns unmodified

##------ Model Details ---------------------------------------------------------
# rutl.count_train_param(model)
# print(model)


##====== Optimizer Zone ===================================================================


criterion = torch.nn.CrossEntropyLoss()

def loss_estimator(pred, truth):
    """ Only consider non-zero inputs in the loss; mask needed
    pred: batch
    """
    pred = pred[:,:,1:]
    truth = truth[:,1:]
    mask = truth.ge(1).type(torch.FloatTensor).to(device)
    loss_ = criterion(pred, truth) * mask
    return torch.mean(loss_)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)


#===============================================================================

if __name__ =="__main__":

    best_loss = float("inf")
    for epoch in range(num_epochs):

        #-------- Training -------------------
        model.train()
        acc_loss = 0
        running_loss = []
        if epoch >= teach_force_till: teacher_forcing = 0

        for ith, (src, tgt, src_sz, tgt_sz) in enumerate(train_dataloader):

            src = src.to(device)
            tgt = tgt.to(device)

            #--- forward ------
            output = model(src, tgt, src_sz, teacher_forcing)
            loss = loss_estimator(output, tgt) / acc_grad
            acc_loss += loss

            #--- backward ------
            if paralleize_load: loss.mean().backward()
            else: loss.backward()

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
        for jth, (v_src, v_tgt, v_src_sz, v_tgt_sz) in enumerate(tqdm(val_dataloader)):
            v_src = v_src.to(device)
            v_tgt = v_tgt.to(device)
            with torch.no_grad():
                v_output = model(v_src, v_tgt, v_src_sz, v_tgt_sz)
                val_loss += loss_estimator(v_output, v_tgt)

                val_accuracy += (1 - val_loss)
            #break
        val_loss = val_loss / len(val_dataloader)
        val_accuracy = val_accuracy / len(val_dataloader)

        print('epoch[{}/{}], [-----TEST------] loss:{:.4f}  Accur:{:.4f}'
              .format(epoch+1, num_epochs, val_loss.data, val_accuracy.data))
        LOG2CSV([val_loss.item(), val_accuracy.item()],
                    LOG_PATH+"valLoss.csv")

        #-------- save Checkpoint -------------------
        if val_loss < best_loss:
            print("***saving best optimal state [Loss:{}] ***".format(val_loss.data))
            best_loss = val_loss
            torch.save(model.state_dict(), WGT_PREFIX+"_model-{}.pth".format(epoch))
            LOG2CSV([epoch+1, val_loss.item(), val_accuracy.item()],
                    LOG_PATH+"bestCheckpoint.csv")
