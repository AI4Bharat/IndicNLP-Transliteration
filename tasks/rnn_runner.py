'''
'''

import torch
from torch.utils.data import DataLoader
import os
import sys
import utilities.running_utils as rutl
from utilities.lang_data_utils import XlitData, GlyphStrawboss
from algorithms.recurrent_nets import Encoder, Decoder, Seq2Seq


##===== Init Setup =============================================================
MODE = rutl.RunMode.train
INST_NAME = "Training_101"

##------------------------------------------------------------------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = "cpu"
if not os.path.exists("logs/"+INST_NAME): os.makedirs("logs/"+INST_NAME)


##===== Running Configuration =================================================

src_glyph = GlyphStrawboss("en")
tgt_glyph = GlyphStrawboss("hi")

num_epochs = 10
batch_size = 2
acc_grad = 1
learning_rate = 1e-5
pretrain_wgt_path = None


train_dataset = XlitData( src_glyph_obj = src_glyph, tgt_glyph_obj = tgt_glyph,
                        json_file='data/HiEn_fire13_dev.json', file_map = "LangEn",
                        padding=True)
train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                                shuffle=True, num_workers=0)


# for i in range(len(train_dataset)):
#     print(train_dataset.__getitem__(i))

##===== Model Configuration =================================================

input_dim = src_glyph.size()
output_dim = tgt_glyph.size()
enc_emb_dim = 256
dec_emb_dim = 256
hid_dim = 512
n_layers = 1
m_dropout = 0

enc = Encoder(  input_dim= input_dim, embed_dim = enc_emb_dim,
                enc_hid_dim= hid_dim,
                enc_layers= n_layers, enc_dropout= m_dropout)
dec = Decoder(  output_dim= output_dim, embed_dim = dec_emb_dim,
                dec_hid_dim= hid_dim, enc_hid_dim= hid_dim,
                dec_layers= n_layers, dec_dropout= m_dropout)

model = Seq2Seq(enc, dec).to(device)

# model = rutl.load_pretrained(model,pretrain_wgt_path) #if path empty returns unmodified

##------ Model Details ---------------------------------------------------------
# rutl.count_train_param(model)
# print(model)
##==============================================================================


criterion = torch.nn.CrossEntropyLoss()

def loss_estimator(pred, truth):
    """ Only consider non-zero inputs in the loss; mask needed """
    #mask = 1 - np.equal(real, 0) # assign 0 to all above 0 and 1 to all 0s
    #print(mask)

    mask = truth.ge(1).type(torch.FloatTensor)
    loss_ = criterion(pred, truth) * mask.device()
    return torch.mean(loss_)

optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                             weight_decay=1e-5)


#===============================================================================

if __name__ =="__main__":

    best_loss = float("inf")
    for epoch in range(num_epochs):
        #--- Train
        acc_loss = 0
        running_loss = []
        for ith, (src, tgt, src_sz, tgt_sz) in enumerate(train_dataloader):

            src = src.to(device)
            tgt = tgt.to(device)

            #--- forward
            output = model(src, tgt, src_sz, tgt_sz)

            loss = loss_estimator(output, tgt) / acc_batch
            acc_loss += loss

            #--- backward
            loss.backward()
            if ( (ith+1) % acc_batch == 0):
                optimizer.step()
                optimizer.zero_grad()
                print('epoch[{}/{}], Mini Batch-{} loss:{:.4f}'
                    .format(epoch+1, num_epochs, (ith+1)//acc_batch, acc_loss.data))
                running_loss.append(acc_loss.data)
                acc_loss=0
                #break
        log_to_csv(running_loss, "logs/"+INST_NAME+"/trainLoss.csv")

        #--- Validate
        val_loss = 0
        val_accuracy = 0
        for jth, (val_src, val_tgt, val_src_sz) in enumerate(test_dataloader):
            val_src = val_src.to(device)
            val_tgt = val_tgt.to(device)
            with torch.no_grad():
                val_output = model(val_img)
                val_loss += loss_estimator(val_output, val_tgt)

                val_accuracy += (1 - val_loss)
            #break
        val_loss = val_loss / len(test_dataloader)
        val_accuracy = val_accuracy / len(test_dataloader)

        print('epoch[{}/{}], [-----TEST------] loss:{:.4f}  Accur:{:.4f}'
              .format(epoch+1, num_epochs, val_loss.data, val_accuracy.data))
        log_to_csv([val_loss.item(), val_accuracy.item()],
                    "logs/"+INST_NAME+"/testLoss.csv")

        #--- save Checkpoint
        if val_loss < best_loss:
            print("***saving best optimal state [Loss:{}] ***".format(val_loss.data))
            best_loss = val_loss
            torch.save(model.state_dict(), "weights/"+INST_NAME+"_model.pth")
            log_to_csv([epoch+1, val_loss.item(), val_accuracy.item()],
                    "logs/"+INST_NAME+"/bestCheckpoint.csv")