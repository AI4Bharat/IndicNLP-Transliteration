import torch, json, time
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from utilities.dataloader import sort_tensorbatch

class Xlit_ModelMgr(nn.Module):
    def __init__(self, model, input_vocab, output_vocab):
        super(Xlit_ModelMgr, self).__init__()
        self.model = model
        self.criterion = nn.NLLLoss(ignore_index=0)#, size_average=True)
        self.softmax = nn.LogSoftmax(dim=1)
        self.input_vocab = input_vocab
        self.output_rev_vocab = {id:alpha for alpha, id in output_vocab.items()}
        
    def masked_loss(self, real, pred):
        """ Only consider non-zero inputs in the loss; mask needed """
        #mask = 1 - np.equal(real, 0) # assign 0 to all above 0 and 1 to all 0s
        #print(mask)
        mask = real.ge(1).type(torch.float)

        loss_ = self.criterion(self.softmax(pred), real) * mask 
        return torch.mean(loss_)
    
    def trainer(self, hparams, dataloader, device='cpu'):
        num_epochs, loss_file, ckpt_dir = hparams.epochs, hparams.loss_file, hparams.ckpt_dir
        self.optimizer = optim.Adam(list(self.model.parameters()), lr=hparams.lr)
        if loss_file:
            loss_file = open(loss_file, 'w')
        for epoch in range(1, num_epochs+1):
            start = time.time()
            self.model.train()
            
            total_loss = 0
            for batch_num, (inp, inp_len, target_ohe, target) in enumerate(dataloader, start=1):
                loss = 0
                x, x_len, y_ohe, y = sort_tensorbatch(inp, inp_len, target_ohe, target, device)
                outputs = self.model(x, x_len, y_ohe, device)
                for t in range(1, y_ohe.size(1)):
                    loss += self.masked_loss(y[:, t], outputs[t])
                batch_loss = (loss / int(y.size(1)))
                total_loss += batch_loss
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                if batch_num % 100 == 0:
                    print('Epoch {} Batch {} Loss {:.4f}'.format(epoch, batch_num, batch_loss.detach().item()))
                    if loss_file:
                        loss_file.write(str(batch_loss.detach().item())); loss_file.write('\n')
                    
            print('END OF EPOCH {} -- AvgLoss {:.4f} -- TIME {:.3f}'.format(
                epoch, total_loss/len(dataloader), time.time() - start))
            if ckpt_dir:
                torch.save(self.model.state_dict(), ckpt_dir+'/model-e%d.pt' % epoch)
        
        # End of training
        if loss_file:
            loss_file.close()
        return
    
    def infer(self, word):
        self.model.eval()
        # Add markers and convert to OHE
        word = '$' + word + '#'
        word = [self.input_vocab[c] for c in word]
        ohe = torch.zeros((len(word), 1, len(self.input_vocab)))
        for i, c in enumerate(word):
            ohe[i][0][c] = 1
        # Run inference!
        output = self(ohe, [len(word)])
        
        # Softmax outputs to lang-word
        result = ''
        for out in output[1:]:
            val, indices = out.topk(1)
            index = indices.tolist()[0][0]
            if index == 0 or index == self.input_vocab['#']:
                break
            result += self.output_rev_vocab[index]
        return result
    
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
    
    