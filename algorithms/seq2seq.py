"""
Deep Learning based Encoder-Decoder models.

Seq2Seq with Attention inspired from:
https://medium.com/dair-ai/neural-machine-translation-with-attention-using-pytorch-a66523f1669f
"""

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class Encoder(nn.Module):
    def __init__(self, vocab_size, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.gru = nn.GRU(self.vocab_size, self.enc_units)
        
    def forward(self, x, lens):
        # x: batch_size, max_length, vocab_size
                
        # x transformed = max_len X batch_size X vocab_size
        x = pack_padded_sequence(x, lens) # unpad
        
        # output: max_length, batch_size, enc_units
        # self.hidden: 1, batch_size, enc_units
        output, self.hidden = self.gru(x) # gru returns hidden state of all timesteps as well as hidden state at last timestep
        
        # pad the sequence to the max length in the batch
        output, _ = pad_packed_sequence(output)
        
        return output, self.hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, dec_units, enc_units, embedding_dim):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.out2hidden = nn.Linear(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim + self.enc_units, 
                          self.dec_units,
                          batch_first=True)
        self.fc = nn.Linear(self.dec_units, self.vocab_size)
        
        # used for attention
        self.W1 = nn.Linear(self.enc_units, self.dec_units)
        self.W2 = nn.Linear(self.enc_units, self.dec_units)
        self.V = nn.Linear(self.enc_units, 1)
    
    def forward(self, x, hidden, enc_output):
        # enc_output original: (max_length, batch_size, enc_units)
        # enc_output converted == (batch_size, max_length, hidden_size)
        enc_output = enc_output.permute(1,0,2)
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        
        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        hidden_with_time_axis = hidden.permute(1, 0, 2)
        
        # score: (batch_size, max_length, hidden_size) # Bahdanaus's
        # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
        # It doesn't matter which FC we pick for each of the inputs
        score = torch.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis))
        
        #score = torch.tanh(self.W2(hidden_with_time_axis) + self.W1(enc_output))
          
        # attention_weights shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying score to self.V
        attention_weights = torch.softmax(self.V(score), dim=1)
        
        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = torch.sum(context_vector, dim=1)
        
        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        # takes case of the right portion of the model above (illustrated in red)
        x = self.out2hidden(x)
        
        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        #x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
        # ? Looks like attention vector in diagram of source
        x = torch.cat((context_vector.unsqueeze(1), x), -1)
        
        # passing the concatenated vector to the GRU
        # output: (batch_size, 1, hidden_size)
        output, state = self.gru(x.float(), hidden.float())
        
        
        # output shape == (batch_size * 1, hidden_size)
        output =  output.view(-1, output.size(2))
        
        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)
        
        return x, state, attention_weights
    
import time
import torch.optim as optim
from tqdm import tqdm
from utilities.dataloader import sort_tensorbatch

class EncoderDecoder(nn.Module):
    def __init__(self, units, input_vocab, output_vocab, embedding_dim):
        super(EncoderDecoder, self).__init__()
        self.encoder = Encoder(len(input_vocab), units)
        self.decoder = Decoder(len(output_vocab), units, units, embedding_dim)
        self.criterion = nn.NLLLoss(ignore_index=0)#, size_average=True)
        self.softmax = nn.LogSoftmax(dim=1)
        self.start_code, self.end_code = input_vocab['$'], input_vocab['#']
        self.input_vocab = input_vocab
        self.output_rev_vocab = {id:alpha for alpha, id in output_vocab.items()}
        self.output_vocab_size = len(output_vocab)
        self.MAX_DECODE_STEPS = 25
        
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), 
                       lr=0.001)
        
    def masked_loss(self, real, pred):
        """ Only consider non-zero inputs in the loss; mask needed """
        #mask = 1 - np.equal(real, 0) # assign 0 to all above 0 and 1 to all 0s
        #print(mask)
        mask = real.ge(1).type(torch.float)

        loss_ = self.criterion(self.softmax(pred), real) * mask 
        return torch.mean(loss_)
    
    def decode(self, dec_hidden, enc_output, y_ohe):
        outputs = []
        if y_ohe is not None:
            dec_input = y_ohe[:, 0].unsqueeze(1)
            outputs.append(y_ohe[:, 0])
            for t in range(1, y_ohe.size(1)):
                prediction, dec_hidden, _ = self.decoder(dec_input.float(), dec_hidden, enc_output)
                outputs.append(prediction)
                # use teacher forcing - feeding the target as the next input (via dec_input)
                dec_input = y_ohe[:, t].unsqueeze(1)
        else:
            dec_input = torch.zeros(enc_output.shape[1], 1, self.output_vocab_size) #(batch_size, 1, out_size)
            dec_input[:, 0, self.start_code] = 1
            outputs.append(dec_input.squeeze(1))
            
            time_steps = y_ohe.size(1) if y_ohe is not None else self.MAX_DECODE_STEPS
            for t in range(1, time_steps):
                prediction, dec_hidden, _ = self.decoder(dec_input.float(), dec_hidden, enc_output)
                outputs.append(prediction)
                max_idx = torch.argmax(prediction, 1, keepdim=True)
                one_hot = torch.zeros(prediction.shape)
                one_hot.scatter_(1, max_idx, 1) # In dim 1, set max_idx's as 1
                dec_input = one_hot.detach().unsqueeze(1)
        return outputs
    
    def forward(self, x, x_len, y_ohe=None, device='cpu'):
        # Run encoder
        enc_output, enc_hidden = self.encoder(x.float(), x_len)

        # Run decoder step-by-step
        return self.decode(enc_hidden, enc_output, y_ohe)
    
    def trainer(self, num_epochs, dataloader, device='cpu', ckpt_dir=None, loss_file=None):
        if loss_file:
            loss_file = open(loss_file, 'w')
        for epoch in range(1, num_epochs+1):
            start = time.time()
            self.train()
            
            total_loss = 0
            for batch_num, (inp, inp_len, target_ohe, target) in enumerate(dataloader, start=1):
                loss = 0
                x, x_len, y_ohe, y = sort_tensorbatch(inp, inp_len, target_ohe, target, device)
                outputs = self(x, x_len, y_ohe, device)
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
                    
            print('END OF EPOCH {} -- LOSS {:.4f} -- TIME {:.3f}'.format(
                epoch, len(dataloader)//dataloader.batch_size, time.time() - start))
            if ckpt_dir:
                torch.save(self.state_dict(), ckpt_dir+'/model-e%d.pt' % epoch)
        if loss_file:
            loss_file.close()
            
    def infer(self, word):
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
            if index == 0 or index == self.end_code:
                break
            result += self.output_rev_vocab[index]
        return result
    
    def run_test(self, test_dataloader, save_json=None):
        data = {}
        for eng_word in tqdm(test_dataloader.eng_words):
            data[eng_word.lower()] = [self.infer(eng_word)]
        if save_json:
            import json
            with open(save_json, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True)
        return data