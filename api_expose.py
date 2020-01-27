"""
Expose Transliteration Engine as an HTTP API.

USAGE:
    Example 1: Running on port 80
    $ sudo env PATH=$PATH python3 api_expose.py
"""

from flask import Flask, jsonify, request 
from datetime import datetime
from utilities.lang_utils import code2lang
import traceback

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

engine = None

class InferenceManager:
    def __init__(self):
        pass
    
    def xlit(self, lang_code, eng_word):
        return ['hello', 'भारत !']
    
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
    engine = InferenceManager()
    app.run(debug = True, host='0.0.0.0', port=80) 