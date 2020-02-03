"""
Expose Transliteration Engine as an HTTP API.

USAGE:
    Example 1: Running on port 80
    $ sudo env PATH=$PATH python3 api_expose.py
"""

from flask import Flask, jsonify, request
from datetime import datetime
from utilities.lang_utils import code2lang, get_lang_chars
from utilities.config import load_and_validate_cfg
from utilities.constants import DEFAULT_CONFIG, BEST_MODEL_FILE
import traceback, torch

LANGS = ['hi']
MODELS_PATH = 'hypotheses/api'

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

engine = None

class InferenceManager:
    def __init__(self, models_path):
        self.init_eng()
        self.init_lang()
        self.models_path = models_path
        self.load_models(models_path)
        
    def load_models(self, models_path):
        from algorithms.seq2seq import EncoderDecoder
        from algorithms.model_manager import Xlit_ModelMgr
        self.models = {}
        for lang in LANGS:
            cfg = load_and_validate_cfg('%s/%s/%s' % (models_path, lang, DEFAULT_CONFIG))
            ckpt_path = '%s/%s/%s' % (models_path, lang, BEST_MODEL_FILE)
            net = EncoderDecoder(cfg.model, self.eng_alpha2index, self.lang_alpha2index[lang])
            net.load_state_dict(torch.load(ckpt_path))
            self.models['hi'] = Xlit_ModelMgr(net, self.eng_alpha2index, self.lang_alpha2index[lang])
    
    def init_eng(self):
        self.eng_alphabets = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.start_char, self.end_char, self.pad_char = '$', '#', '*'
        self.eng_alpha2index = {self.pad_char: 0, self.start_char: 1, self.end_char: 2}
        for index, alpha in enumerate(self.eng_alphabets):
            self.eng_alpha2index[alpha] = index + 3
    
    def init_lang(self):
        self.lang_alphabets = {}
        self.lang_alpha2index = {}
        for lang in LANGS:
            self.lang_alphabets[lang] = get_lang_chars(code2lang[lang])
            self.lang_alpha2index[lang] = {self.pad_char: 0, self.start_char: 1, self.end_char: 2}
            for index, alpha in enumerate(self.lang_alphabets[lang]):
                self.lang_alpha2index[lang][alpha] = index + 3
    
    def xlit(self, lang_code, eng_word):
        return [self.models[lang_code].infer(eng_word.upper())]
    
@app.route('/tl/<lang_code>/<eng_word>', methods = ['GET', 'POST']) 
def varnam_xlit(lang_code, eng_word): 
    response = {
        'success': True,
        'error': '',
        'at': str(datetime.utcnow()) + ' +0000 UTC',
        'input': eng_word
    }
    if lang_code not in code2lang:
        response['error'] = 'Invalid scheme identifier'
        response['success'] = False
        return jsonify(response)
    try:
        response['result'] = engine.xlit(lang_code, eng_word)
    except Exception as e:
        response['error'] = 'Man you crashed me ;('
        response['success'] = False
        # TODO: Save crashes to logs
        print(traceback.format_exc())
    return jsonify(response)

if __name__ == '__main__': 
    engine = InferenceManager(MODELS_PATH)
    app.run(debug = True, host='0.0.0.0', port=80) 