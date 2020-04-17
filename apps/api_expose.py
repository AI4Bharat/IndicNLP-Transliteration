"""
Expose Transliteration Engine as an HTTP API.

USAGE:
    1. $ sudo env PATH=$PATH python3 api_expose.py
    2. Run in browser: http://localhost:8000/tl/hi/a
"""

from flask import Flask, jsonify, request
from datetime import datetime
import traceback
import os
import sys
import enum

class XlitError(enum.Enum):
    lang_err = "Unsupported langauge ID requested"
    string_err = "String passed is incompatable"
    internal_err = "Internal crash ;("
    unknown_err = "Unknown Failure"

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

## Set in order to host in specific domain
SSL_FILES = None
'''
SSL_FILES = ('/etc/letsencrypt/live/domain.com/fullchain.pem',
            '/etc/letsencrypt/live/domain.com/privkey.pem')
'''

@app.route('/tl/<lang_code>/<eng_word>', methods = ['GET', 'POST'])
def ai4bharat_xlit(lang_code, eng_word):
    response = {
        'success': False,
        'error': 'Unknown',
        'at': str(datetime.utcnow()) + ' +0000 UTC',
        'input': eng_word,
        'result': ''
    }

    if lang_code not in engine.langs:
        response['error'] = 'Invalid scheme identifier. Supported languages are'+ str(engine.langs)
        return jsonify(response)

    try:
        xlit_result = engine.transliterate(lang_code, eng_word)
    except Exception as e:
        xlit_result = XlitError.internal_err
        print(traceback.format_exc())


    if isinstance(xlit_result, XlitError):
        response['error'] = xlit_result.value
        print(traceback.format_exc())
    else:
        response['result'] = xlit_result
        response['success'] = True

    return jsonify(response)

##------------------------------------------------------------------------------

BASEPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(BASEPATH)

class XlitEngine():
    def __init__(self):
        self.langs = ["hi"]

        from bin.hindi.program85 import inference_engine as hindi_engine
        self.hindi_engine = hindi_engine

    def transliterate(self, lang_code, eng_word):
        try:
            if lang_code == "hi":
                return self.hindi_engine(eng_word)
            else:
                print("Unknown Langauge")
                return XlitError.lang_err
        except:
            return XlitError.unknown_err



if __name__ == '__main__':
    engine = XlitEngine()
    app.run(debug=True, host='0.0.0.0', port=8000)