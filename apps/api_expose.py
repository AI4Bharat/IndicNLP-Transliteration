"""
Expose Transliteration Engine as an HTTP API.

USAGE:
    1. $ sudo env PATH=$PATH python3 api_expose.py
       $ sudo env PATH=$PATH GOOGLE_APPLICATION_CREDENTIALS=/path_to_cred/ python3 api_expose.py

    2. Run in browser: production_port - 80
            http://localhost:8000/tl/ta/amma
            http://localhost:8000/languages

FORMAT:
    Based on the Varnam API standard
    https://api.varnamproject.com/tl/hi/bharat

"""
from flask import Flask, jsonify, request, make_response
from uuid import uuid4
from datetime import datetime
import traceback
import os
import sys
import enum
import csv

class XlitError(enum.Enum):
    lang_err = "Unsupported langauge ID requested ;( Please check available languages."
    string_err = "String passed is incompatable ;("
    internal_err = "Internal crash ;("
    unknown_err = "Unknown Failure"
    loading_err = "Loading failed ;( Check if metadata/paths are correctly configured."

app = Flask(__name__)
app.config['JSON_AS_ASCII'] = False

## ------------------------- Configure ---------------------------------------- ##

DEBUG = False
## Set in order to host in specific domain
SSL_FILES = None
# SSL_FILES = ('/etc/letsencrypt/live/xlit-api.ai4bharat.org/fullchain.pem',
#              '/etc/letsencrypt/live/xlit-api.ai4bharat.org/privkey.pem')

CLOUD_STORE = False #

## ------------------------- Logging ---------------------------------------- ##
os.makedirs('logs/', exist_ok=True)
USER_CHOICES_LOGS = 'logs/user_choices.tsv'
ANNOTATION_LOGS = 'logs/annotation_data.tsv'

USER_DATA_FIELDS = ['user_ip', 'user_id', 'timestamp', 'input', 'lang', 'output', 'topk_index']
ANNOTATE_DATA_FIELDS = ['user_ip', 'user_id', 'timestamp','lang', 'native', 'ann1', 'ann2', 'ann3']

def create_log_files():
    if not os.path.isfile(USER_CHOICES_LOGS):
        with open(USER_CHOICES_LOGS, 'w', buffering=1) as f:
            writer = csv.DictWriter(f, fieldnames=USER_DATA_FIELDS)
            writer.writeheader()

    if not os.path.isfile(ANNOTATION_LOGS):
        with open(ANNOTATION_LOGS, 'w', buffering=1) as f:
            writer = csv.DictWriter(f, fieldnames=ANNOTATE_DATA_FIELDS)
            writer.writeheader()

create_log_files()

## ----- Google FireStore
"""
Requires gcp credentials
"""
if CLOUD_STORE:
    from google.cloud import firestore
    db = firestore.Client()
    usrch_coll = "path_to_collection"
    annot_coll = "path_to_collection"

def add_document(coll, data): # FireStore
    doc_title = str(uuid4().hex)
    ref = db.collection(coll).document(doc_title)
    ref.set(data)

## --------------------

def write_userdata(data):
    with open(USER_CHOICES_LOGS, 'a', buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=USER_DATA_FIELDS)
        writer.writerow(data)
    if CLOUD_STORE:
        add_document(usrch_coll, data)
    return

def write_annotatedata(data):
    with open(ANNOTATION_LOGS, 'a', buffering=1) as f:
        writer = csv.DictWriter(f, fieldnames=ANNOTATE_DATA_FIELDS)
        writer.writerow(data)
    if CLOUD_STORE:
        add_document(annot_coll, data)
    return

## ---------------------------- API End-points ------------------------------ ##

@app.route('/languages', methods = ['GET', 'POST'])
def supported_languages():
    # Format - https://xlit-api.ai4bharat.org/languages
    langs = []
    for code, name in engine.langs.items():
        langs.append({
            "LangCode": code,
            "Identifier": code,
            "DisplayName": name,
            "Author": "AI4Bharat",
            "CompiledDate": "07-November-2020",
            "IsStable": True
        })
    # TODO: Save this variable permanently, as it will be constant

    response = make_response(jsonify(langs))
    if 'xlit_user_id' not in request.cookies:
        # host = request.environ['HTTP_ORIGIN'].split('://')[1]
        host = '.ai4bharat.org'
        response.set_cookie('xlit_user_id', uuid4().hex, max_age=365*24*60*60, domain=host, samesite='None', secure=True, httponly=True)
    return response

@app.route('/tl/<lang_code>/<eng_word>', methods = ['GET', 'POST'])
def xlit_api(lang_code, eng_word):
    # Format: https://xlit-api.ai4bharat.org/tl/ta/bharat
    response = {
        'success': False,
        'error': '',
        'at': str(datetime.utcnow()) + ' +0000 UTC',
        'input': eng_word.strip(),
        'result': ''
    }

    if lang_code not in engine.langs:
        response['error'] = 'Invalid scheme identifier. Supported languages are'+ str(engine.langs)
        return jsonify(response)

    try:
        ## Limit char count to --> 70
        xlit_result = engine.translit_word(eng_word[:70], lang_code)
    except Exception as e:
        xlit_result = XlitError.internal_err


    if isinstance(xlit_result, XlitError):
        response['error'] = xlit_result.value
        print("XlitError:", traceback.format_exc())
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
    if 'user_id' not in data:
        data['user_id'] = request.cookies['xlit_user_id'] if 'xlit_user_id' in request.cookies else None
    data['timestamp'] = str(datetime.utcnow()) + ' +0000 UTC'
    write_userdata(data)
    return jsonify({'status': 'Success'})

@app.route('/annotate', methods=['POST'])
def annotate_by_user():
    data = request.get_json(force=True)
    data['user_ip'] = request.remote_addr
    if 'user_id' not in data:
        data['user_id'] = request.cookies['xlit_user_id'] if 'xlit_user_id' in request.cookies else None
    data['timestamp'] = str(datetime.utcnow()) + ' +0000 UTC'
    write_annotatedata(data)
    return jsonify({'status': 'Success'})


## -------------------------- Server Setup ---------------------------------- ##

def host_https():
    https_server = WSGIServer(('0.0.0.0', 443), app,
                                     certfile=SSL_FILES[0], keyfile=SSL_FILES[1])
    print('Starting HTTPS Server...')
    https_server.serve_forever()
    return



## ----------------------------- Xlit Engine -------------------------------- ##

BASEPATH = os.path.dirname(os.path.realpath(__file__))
sys.path.append(BASEPATH)
from xlit_src import XlitEngine


## -------------------------------------------------------------------------- ##

if __name__ == '__main__':
    engine = XlitEngine()
    if not DEBUG: # Production Server
        from flask_cors import CORS, cross_origin
        cors = CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
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