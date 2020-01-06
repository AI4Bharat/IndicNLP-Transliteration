"""
Get transliteration predictions from online services / libraries.

Usage:
<<script.py>> --lang hindi --xlit-src (ENGINE) --output <out.json> [--json/--txt input_file] [--infer <word>]

ENGINES:
- google
- quillpad
"""

import os, sys, json, requests
from tqdm import tqdm

TOP_K = 6
LANG2CODE = {
	'tamil': 'ta',
	'hindi': 'hi',
	'telugu': 'te',
	'marathi': 'mr',
	'punjabi': 'pa',
	'bengali': 'bn',
	'gujarati': 'gu',
	'kannada': 'kn',
	'malayalam': 'ml',
	'nepali': 'ne'
}

G_API = 'https://inputtools.google.com/request?text=%s&itc=%s-t-i0-und&num=%d'
def gtransliterate(word, lang_code, num_suggestions=10):
	response = requests.request('GET', G_API % (word, lang_code, num_suggestions), allow_redirects=False, timeout=5)
	r = json.loads(response.text)
	if 'SUCCESS' not in r[0] or response.status_code != 200:
		print('Request failed with status code: %d\nERROR: %s' % (response.status_code, response.text), file=sys.stderr)
		return []
	return r[1][0][1]

def run_gtransliteration_bulk(input_list, lang_code):
    predictions = {}
    for word in tqdm(input_list, desc='gTransliterating...', unit=' Xlits'):
        predictions[word] = gtransliterate(word, lang_code, TOP_K)
    return predictions

QP_API = 'http://xlit.quillpad.in/quillpad_backend2/processWordJSON?lang=%s&inString=%s'
def qp_transliterate(word, lang):
    response = requests.request('GET', QP_API % (lang, word), allow_redirects=False, timeout=5)
    if response.status_code != 200:
        print('Request failed with status code: %d\nERROR: %s' % (response.status_code, response.text), file=sys.stderr)
        return []
    r = json.loads(response.text)
    suggestions = r['twords'][0]['options']
    if r['itrans'] not in suggestions:
        suggestions.append(r['itrans'])
    return suggestions

def run_qp_transliteration_bulk(input_list, lang):
    predictions = {}
    for word in tqdm(input_list, desc='qpTransliterating...', unit=' Xlits'):
        predictions[word] = qp_transliterate(word, lang)
    return predictions

def predict_transliterations(input_json, input_txt, xlit_engine, lang):
    input_data = set()

    # Read input data from JSON file
    if input_json and os.path.isfile(input_json):
        with open(input_json, encoding='utf8') as f:
            data = json.load(f)
        input_data.update(set(data.keys()))
    
    # Read input data from txt file with one word per line
    if input_txt and os.path.isfile(input_txt):
        with open(input_txt, encoding='utf8') as f:
            for line in f: input_data.add(line.strip())
    
    if 'google' in xlit_engine:
        return run_gtransliteration_bulk(input_data, LANG2CODE[lang])
    elif 'quill' in xlit_engine:
        return run_qp_transliteration_bulk(input_data, lang)
    else:
        sys.exit(xlit_engine, 'xlit engine is not supported.')

def save_predictions(data, outfile):
    print('Saving output to', outfile)
    with open(outfile, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True)

def predict_word(word, lang, xlit_engine):
    if 'google' in xlit_engine:
        predictions = gtransliterate(word, LANG2CODE[lang], TOP_K)
    elif 'quill' in xlit_engine:
        predictions = qp_transliterate(word, lang)
    else:
        sys.exit(xlit_engine, 'xlit engine is not supported.')

    print(predictions)

def __main(args):

    if args.lang not in LANG2CODE:
        sys.exit(args.lang, 'language is not supported.')
    
    if args.infer:
        predict_word(args.infer, args.lang, lagnargs.xlit_src.lower())
        return

    if not args.json and not args.txt:
        sys.exit('Enter atleast one source of data', file=sys.stderr)

    predictions = predict_transliterations(args.json, args.txt, args.xlit_src.lower(), args.lang)
    save_predictions(predictions, args.output)
    

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--json', type=str)
    parser.add_argument('--txt', type=str)
    parser.add_argument('--infer', type=str)
    parser.add_argument('--xlit-src', type=str, required=True)
    parser.add_argument('--lang', type=str, required=True)
    parser.add_argument('--output', type=str, required=True)
    __main(parser.parse_args())
    