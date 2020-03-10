from torch.utils.data import DataLoader
import os, sys, yaml, torch
sys.path.append('.')

from utilities.dataloader import Transliteration_Dataset, sort_tensorbatch
from utilities.lang_utils import get_lang_chars
from algorithms.model_manager import Xlit_ModelMgr
from utilities.config import load_and_validate_cfg

def get_model(model_cfg, data):
    if model_cfg.type == 'seq2seq':
        from algorithms.seq2seq import EncoderDecoder
        model = EncoderDecoder(model_cfg, data.eng_alpha2index, data.lang_alpha2index)
    else:
        sys.exit(model_cfg.type, 'NOT SUPPORTED')
    return model

def trainer_loop(model, config, train_dataloader, test_dataloader=None, device='cpu'):
    model_mgr = Xlit_ModelMgr(model, train_dataloader.dataset.eng_alpha2index, train_dataloader.dataset.lang_alpha2index, device)
    model_mgr.trainer(config.hyperparams, train_dataloader, test_dataloader)

def main(config):
    try:
        lang_alphabets = get_lang_chars(config.lang)
    except Exception as e:
        print(e)
        sys.exit('ERROR: Language not found')
    
    train_data = Transliteration_Dataset(config.train_data_json, lang_alphabets)
    train_dataloader = DataLoader(train_data, batch_size=config.hyperparams.batch_size, 
                     drop_last=True, pin_memory=True, shuffle=True)
    test_data = None
    if os.path.isfile(config.test_data_json):
        test_data = Transliteration_Dataset(config.test_data_json, lang_alphabets)
        test_dataloader = DataLoader(test_data, batch_size=config.hyperparams.infer_batch_size, 
                     pin_memory=True, shuffle=False)
        
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(config.model, train_data)
    
    trainer_loop(model, config, train_dataloader, test_dataloader, device)
    

if __name__ == '__main__':
    config = load_and_validate_cfg(sys.argv[1])
    main(config)
