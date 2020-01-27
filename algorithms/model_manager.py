import torch, json, time
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utilities.dataloader import sort_tensorbatch
from utilities.accuracy_news import evaluate

class Xlit_ModelMgr(nn.Module):
    def __init__(self, model, input_vocab, output_vocab, device='cpu'):
        super(Xlit_ModelMgr, self).__init__()
        self.device = device
        self.model = model.to(device)
        self.criterion = nn.NLLLoss(ignore_index=0)#, size_average=True)
        self.softmax = nn.LogSoftmax(dim=1)
        self.input_vocab = input_vocab
        self.output_rev_vocab = {id:alpha for alpha, id in output_vocab.items()}
        
    def forward(self, *args):
        return self.model(*args)
        
    def masked_loss(self, real, pred):
        """ Only consider non-zero inputs in the loss; mask needed """
        #mask = 1 - np.equal(real, 0) # assign 0 to all above 0 and 1 to all 0s
        #print(mask)
        mask = real.ge(1).type(torch.float)

        loss_ = self.criterion(self.softmax(pred), real) * mask 
        return torch.mean(loss_)
    
    def train_epoch(self, dataloader, epoch, teacher_train_epochs, loss_file=None):
        start = time.time()
        self.model.train()
        teacher_force = epoch <= teacher_train_epochs

        epoch_losses = []
        for batch_num, (inp, inp_len, target_ohe, target, _) in enumerate(dataloader, start=1):
            loss = 0
            x, x_len, y_ohe, y, _ = sort_tensorbatch(inp, inp_len, target_ohe, target, _, self.device)
            outputs = self.model(x, x_len, y_ohe, teacher_force)
            for t in range(1, y_ohe.size(1)):
                loss += self.masked_loss(y[:, t], outputs[t])
            batch_loss = (loss / int(y.size(1)))
            epoch_losses.append(batch_loss)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if batch_num % 100 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch, batch_num, batch_loss.detach().item()))
        
        # Write losses to file
        if loss_file:
            epoch_csv = ','.join(map(str, epoch_losses)) + '\n'
            loss_file.write(epoch_csv)
        
        return epoch_losses, time.time() - start
    
    def trainer(self, hparams, dataloader, test_dataloader=None):
        num_epochs, ckpt_dir = hparams.epochs, hparams.ckpt_dir
        loss_file, score_file = hparams.loss_file, hparams.score_file
        self.optimizer = optim.Adam(list(self.model.parameters()), lr=hparams.lr)
        teacher_train_epochs = hparams.teacher_epochs if 'teacher_epochs' in hparams else 0
        
        if loss_file:
            loss_file = open(loss_file, 'w')
        if score_file:
            score_file = open(score_file, 'w')
            score_file.write('ACCURACY,MEAN_F\n') # Write Header
        
        for epoch in range(1, num_epochs+1):
            epoch_losses, time_taken = self.train_epoch(dataloader, epoch, teacher_train_epochs, loss_file)
            total_loss = sum(epoch_losses)
                    
            print('END OF EPOCH {} -- AvgLoss {:.4f} -- TIME {:.3f}'.format(
                epoch, total_loss/len(dataloader), time_taken))
            
            # Write checkpoint and test scores
            if ckpt_dir:
                torch.save(self.model.state_dict(), ckpt_dir+'/model-e%d.pt' % epoch)
            if test_dataloader:
                scores = self.validate_test(test_dataloader)
                scores_csv = '%f,%f\n' % (scores['accuracy'], scores['mean_f'])
                print('Accuracy, Mean_F: ', scores_csv)
                if score_file:
                    score_file.write(scores_csv)
            
        
        # End of training
        if loss_file:
            loss_file.close()
        if score_file:
            score_file.close()
        return
    
    def validate_test(self, dataloader):
        self.model.eval()
        pred_data = {}
        for inp, inp_len, target_ohe, target, indices in dataloader:
            x, x_len, y_ohe, y, indices = sort_tensorbatch(inp, inp_len, target_ohe, target, indices, self.device)
            outputs = torch.stack(self.model(x, x_len), dim=0).transpose(0,1) # (batch_size, seq_len, out_vocab_size)
            for idx, out in enumerate(outputs):
                pred = self.softmax2word(out)
                eng = dataloader.dataset.eng_words[indices[idx]]
                pred_data[eng.lower()] = [pred]
        
        acc, f, f_best_match, mrr, map_ref = evaluate(pred_data, dataloader.dataset.data, verbose=False)
        scores = {}
        N = len(acc)
        scores['accuracy'] = float(sum([acc[src_word] for src_word in acc.keys()]))/N
        scores['mean_f'] = float(sum([f[src_word] for src_word in f.keys()]))/N
        return scores
    
    def softmax2word(self, softmax):
        result = ''
        for out in softmax[1:]:
            val, indices = out.topk(1)
            index = indices.tolist()[0]
            if index == 0 or index == self.input_vocab['#']:
                break
            result += self.output_rev_vocab[index]
        return result
    
    def infer(self, word):
        self.model.eval()
        # Add markers and convert to OHE
        word = '$' + word + '#'
        word = [self.input_vocab[c] for c in word]
        ohe = torch.zeros((len(word), 1, len(self.input_vocab)))
        for i, c in enumerate(word):
            ohe[i][0][c] = 1
        # Run inference!
        ohe = ohe.to(self.device)
        lens = torch.Tensor([len(word)]).to(self.device)
        output = torch.stack(self.model(ohe, lens), dim=0)[0]
        return self.softmax2word(output)
    
    def run_test(self, test_dataset, save_json=None):
        self.model.eval()
        data = {}
        for eng_word in tqdm(test_dataset.eng_words):
            data[eng_word.lower()] = [self.infer(eng_word)]
        if save_json:
            import json
            with open(save_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True)
        return data
