"""
Expose Transliteration Engine as an HTTP API.

USAGE:
    1. $ sudo env PATH=$PATH python3 api_expose.py
    2. Run in browser: http://localhost:8000/tl/gom/a
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
# SSL_FILES = None
SSL_FILES = ('/etc/letsencrypt/live/xlit-api.ai4bharat.org/fullchain.pem',
            '/etc/letsencrypt/live/xlit-api.ai4bharat.org/privkey.pem')


@app.route('/languages', methods = ['GET', 'POST'])
def supported_languages():
    # Format: https://api.varnamproject.com/languages
    langs = []
    for code, name in engine.langs.items():
        langs.append({
            "LangCode": code,
            "Identifier": code,
            "DisplayName": name,
            "Author": "AI4Bharat",
            "CompiledDate": "IDK when",
            "IsStable": True
        })
    # TODO: Save this variable permanently, as it will be constant
    return jsonify(langs)

@app.route('/tl/<lang_code>/<eng_word>', methods = ['GET', 'POST'])
def xlit_api(lang_code, eng_word):
    # Format: https://api.varnamproject.com/tl/hi/bharat
    response = {
        'success': False,
        'error': '',
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


    if isinstance(xlit_result, XlitError):
        response['error'] = xlit_result.value
        print(traceback.format_exc())
    else:
        response['result'] = xlit_result
        response['success'] = True

    return jsonify(response)

@app.route('/rtl/<lang_code>/<word>', methods = ['GET', 'POST'])
def reverse_xlit_api(lang_code, word):
    # Format: https://api.varnamproject.com/rtl/hi/%E0%A4%AD%E0%A4%BE%E0%A4%B0%E0%A4%A4
    response = {
        'success': False,
        'error': '',
        'at': str(datetime.utcnow()) + ' +0000 UTC',
        'input': word,
        'result': ''
    }
    # TODO: Implement?
    respose['error'] = 'Not yet implemented!'
    return jsonify(response)

##------------------------------------------------------------------------------

BASEPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(BASEPATH)

class XlitEngine():
    def __init__(self):
        self.langs = {"hi": "Hindi", "gom": "Konkani (Goan)"}

        try:
            from bins.hindi.program85 import inference_engine as hindi_engine
            self.hindi_engine = hindi_engine
        except:
            print("Failure in loading Hindi")
            del self.langs['hi']

        try:
            from bins.konkani.gom_program104 import inference_engine as konkani_engine
            self.konkani_engine = konkani_engine
        except:
            print("Failure in loading Konkani")
            del self.langs['gom']


    def transliterate(self, lang_code, eng_word):
        eng_word = self._clean(eng_word)
        if eng_word == "":
            return []

        if lang_code not in self.langs:
            print("Unknown Langauge requested", lang_code)
            return XlitError.lang_err

        try:
            if lang_code == "hi":
                return self.hindi_engine(eng_word)
            elif lang_code == "gom":
                return self.konkani_engine(eng_word)

        except:
            return XlitError.unknown_err

    def _clean(self, word):
        word = word.lower()
        accepted = "abcdefghijklmnopqrstuvwxyz"
        word = ''.join([i for i in word if i in accepted])
        return word


if __name__ == '__main__':
    engine = XlitEngine()
    if SSL_FILES: # Production Server
        from flask_cors import CORS, cross_origin
        cors = CORS(app, resources={r"/*": {"origins": "*"}})
        # app.run(host='0.0.0.0', port=443, ssl_context=SSL_FILES)

        from gevent.pywsgi import WSGIServer
        http_server = WSGIServer(('0.0.0.0', 443), app,
                                 certfile=SSL_FILES[0], keyfile=SSL_FILES[1])
        print('Starting HTTPS Server...')
        http_server.serve_forever()
    else: # Development Server
        app.run(debug=True, host='0.0.0.0', port=8000)