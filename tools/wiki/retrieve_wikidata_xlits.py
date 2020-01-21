"""
Get Transliteration phrase-pairs from WikiData Proper Nouns
like persons, locations, etc.

USAGE:
    <script.py> --output-folder dumps/Hindi/ --lang hindi

NOTE:
    Data needs heavy post-processing.
"""

import requests, json, sys, os
from tqdm import tqdm

WIKIDATA_SPARQL_URL = 'https://query.wikidata.org/sparql'

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

def phrase_split(phrase):
    return phrase.replace('-', ' ').replace(',', ' ').split()

def save_xlit_json(data, output_file, lang_code):
    en_key = 'entity_en'
    lang_key = 'entity_' + lang_code
    result = {}
    print('Saving', output_file)
    for datum in tqdm(data):
        # Assuming all phrases be unqiue
        key = datum[en_key]['value']
        value = datum[lang_key]['value']
        # if len(phrase_split(key)) == len(phrase_split(value)):
        result[key] = [value]
    print('%d keys parsed' % len(result))
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(result, f, ensure_ascii=False, indent=4, sort_keys=True)
    return result

wikidata_queries = {
    'indian_places': '''
        SELECT ?entity_en ?entity_{lang} WHERE {{
            ?entity wdt:P17 wd:Q668;
                    rdfs:label ?entity_{lang} filter (lang(?entity_{lang}) = "{lang}").
            ?entity rdfs:label ?entity_en filter (lang(?entity_en) = "en").
        }}''',
    
    'usa_places': '''
        SELECT ?entity_en ?entity_{lang} WHERE {{
            hint:Query hint:optimizer "None".
            ?entity wdt:P17 wd:Q30;
                    rdfs:label ?entity_{lang} filter (lang(?entity_{lang}) = "{lang}").
            ?entity rdfs:label ?entity_en filter (lang(?entity_en) = "en").
        }}''',
    
    'indians': '''
        SELECT DISTINCT ?entity_en ?entity_{lang} WHERE {{
            VALUES ?countries {{ wd:Q668 wd:Q1775277 wd:Q129286 wd:Q843 wd:Q902 wd:Q837 wd:Q854 }}.
            ?entity wdt:P27 ?countries;
                    rdfs:label ?entity_{lang} filter (lang(?entity_{lang}) = "{lang}").
            ?entity rdfs:label ?entity_en filter (lang(?entity_en) = "en").
        }}''',
    
    'foreign_indians': '''
        SELECT ?entity_en ?entity_{lang} WHERE {{
            VALUES ?countries {{ wd:Q145 wd:Q408 wd:Q334 wd:Q833 wd:Q252 }}.
            ?entity wdt:P27 ?countries;
                    rdfs:label ?entity_{lang} filter (lang(?entity_{lang}) = "{lang}").
            ?entity rdfs:label ?entity_en filter (lang(?entity_en) = "en").
        }}''',
    
    'usa_persons': '''
        SELECT ?entity_en ?entity_{lang} WHERE {{
            ?entity wdt:P27 wd:Q30;
                    rdfs:label ?entity_{lang} filter (lang(?entity_{lang}) = "{lang}").
            ?entity rdfs:label ?entity_en filter (lang(?entity_en) = "en").
        }}''',
    
    'bollywood_films': '''
        SELECT ?entity_en ?entity_{lang} WHERE {{
            ?entity wdt:P364 wd:Q1568;
                    rdfs:label ?entity_{lang} filter (lang(?entity_{lang}) = "{lang}").
            ?entity rdfs:label ?entity_en filter (lang(?entity_en) = "en").
        }}''',
}

def retrieve_query(query_name, output_folder, lang_code):
    print('Running query for:', query_name)
    query = wikidata_queries[query_name].format(lang=lang_code)
    r = requests.get(WIKIDATA_SPARQL_URL, params = {'format': 'json', 'query': query})
    data = json.loads(r.text)
    output_file = os.path.join(output_folder, '%s_%s.json' % (query_name, lang_code))
    save_xlit_json(data['results']['bindings'], output_file, lang_code)

def retrieve_wikidata(output_folder, lang):
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder, exist_ok=True)
    lang_code = LANG2CODE[lang]
    for q in wikidata_queries:
        retrieve_query(q, output_folder, lang_code)
    return

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-folder", required=True, type=str)
    parser.add_argument("--lang", required=True, type=str)
    args = parser.parse_args()
    if args.lang not in LANG2CODE:
        sys.exit('Language:', args.lang, 'not supported')
    retrieve_wikidata(args.output_folder, args.lang)