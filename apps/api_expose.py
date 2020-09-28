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
import csv

class XlitError(enum.Enum):
    lang_err = "Unsupported langauge ID requested"
    string_err = "String passed is incompatable"
    internal_err = "Internal crash ;("
    unknown_err = "Unknown Failure"

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

DEBUG = True
## Set in order to host in specific domain
SSL_FILES = None
# SSL_FILES = ('/etc/letsencrypt/live/xlit-api.ai4bharat.org/fullchain.pem',
#              '/etc/letsencrypt/live/xlit-api.ai4bharat.org/privkey.pem')


## ------------------------- Logging ---------------------------------------- ##

os.makedirs('logs/', exist_ok=True)
USER_CHOICES_LOGS = 'logs/user_choices.csv'

# Write CSV header
USER_DATA_FIELDS = ['user_ip', 'user_id', 'timestamp', 'input', 'lang', 'output', 'topk_index']
if not os.path.isfile(USER_CHOICES_LOGS):
    with open(USER_CHOICES_LOGS, 'w', buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=USER_DATA_FIELDS)
        writer.writeheader()

def write_userdata(data):
    with open(USER_CHOICES_LOGS, 'a', buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=USER_DATA_FIELDS)
        writer.writerow(data)
    return

## ---------------------------- API End-points ------------------------------ ##

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
            "CompiledDate": "28-September-2020",
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

@app.route('/learn', methods=['POST'])
def learn_from_user():
    data = request.get_json(force=True)
    data['user_ip'] = request.remote_addr
    data['user_id'] = None
    data['timestamp'] = str(datetime.utcnow()) + ' +0000 UTC'
    write_userdata(data)
    return jsonify({'status': 'Success'})

@app.route('/learn_context', methods=['POST'])
def learn_from_context():
    data = request.get_json(force=True)
    data['user_ip'] = request.remote_addr
    data['timestamp'] = str(datetime.utcnow()) + ' +0000 UTC'
    write_userdata(data)
    return jsonify({'status': 'Success'})


## ----------------------------- Xlit Engine -------------------------------- ##

BASEPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(BASEPATH)

class XlitEngine():
    def __init__(self):
        self.langs = {"hi": "Hindi", "gom": "Konkani (Goan)", "mai": "Maithili"}

        try:
            from models.hindi.hi_program110 import inference_engine as hindi_engine
            self.hindi_engine = hindi_engine
        except Exception as error:
            print("Failure in loading Hindi \n", error)
            del self.langs['hi']

        try:
            from models.konkani.gom_program116 import inference_engine as konkani_engine
            self.konkani_engine = konkani_engine
        except Exception as error:
            print("Failure in loading Konkani \n", error)
            del self.langs['gom']

        try:
            from models.maithili.mai_program120 import inference_engine as maithili_engine
            self.maithili_engine = maithili_engine
        except Exception as error:
            print("Failure in loading Maithili \n", error)
            del self.langs['mai']

    def transliterate(self, lang_code, eng_word):
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
            elif lang_code == "mai":
                return self.maithili_engine(eng_word)

        except error as Exception:
            print("Error:", error)
            return XlitError.unknown_err


## -------------------------- Server Setup ---------------------------------- ##

def host_https():
    https_server = WSGIServer(('0.0.0.0', 443), app,
                                     certfile=SSL_FILES[0], keyfile=SSL_FILES[1])
    print('Starting HTTPS Server...')
    https_server.serve_forever()
    return

if __name__ == '__main__':
    engine = XlitEngine()
    if not DEBUG: # Production Server
        from flask_cors import CORS, cross_origin
        cors = CORS(app, resources={r"/*": {"origins": "*"}})
        # app.run(host='0.0.0.0', port=443, ssl_context=SSL_FILES)

        from gevent.pywsgi import WSGIServer
        if SSL_FILES:
            from multiprocessing import Process
            Process(target=host_https).start()

        http_server = WSGIServer(('0.0.0.0', 80), app)
        print('Starting HTTP Server...')
        http_server.serve_forever()
    else: # Development Server
        app.run(debug=True, host='0.0.0.0', port=8000)