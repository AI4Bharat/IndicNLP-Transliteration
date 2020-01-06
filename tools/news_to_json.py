"""
Convert NEWS2018 XML dataset to JSON.
"""

import os, sys
from tqdm import tqdm
import xml.etree.ElementTree as ET
import json

USAGE = 'USAGE: python %s --news-xml EnHi.xml --output EnHi.json <--output-reverse HiEn.json> <--phrase-mode>' % sys.argv[0]

def phrase_split(phrase):
    return phrase.replace('-', ' ').replace(',', ' ').split()

def pretty_write_json(data, outfile):
    with open(outfile, 'w', encoding='utf8') as f:
        json.dump(data, f, ensure_ascii=False, indent=4, sort_keys=True)

# Returns a dict containing src lang phrases as keys
# and correspoding set of dest phrases as values
def read_xml_dataset(filename, phrase_mode):
    transliteration_corpus = ET.parse(filename).getroot()
    output_dict = {}

    if phrase_mode: # Save as it is
        for pair in tqdm(transliteration_corpus, desc='Processing...'):
            if pair[0].text not in output_dict:
                output_dict[pair[0].text] = {pair[1].text}
            else:
                output_dict[pair[0].text].add(pair[1].text)
        return output_dict

    # If not phrase_mode, split each phrase into words and 
    # match with corresponding transliterations
    for pair in tqdm(transliteration_corpus, desc='Processing...'):
        if len(pair) != 2:
            if len(pair) == 1: print('No translit found for', pair[0].text)
            continue
        src_wordlist = phrase_split(pair[0].text)
        dst_wordlist = phrase_split(pair[1].text)

        # Skip noisy data
        if len(src_wordlist) != len(dst_wordlist):
            print('Skipping', '#'+pair.attrib['ID'], pair[0].text, ' - ', pair[1].text)
            continue

        for src, dest in zip(src_wordlist, dst_wordlist):
            if src not in output_dict:
                output_dict[src] = {dest}
            else:
                output_dict[src].add(dest)
    
    return output_dict

def postprocess_data(data):
    for key in data:
        # TODO: Remove non-Indic characters, handle numbers and punctutaions
        data[key] = list(data[key])
    return data

# Converts to back-transliteration and save JSON
def save_reverse_data(forward_data, output_file):
    data = {}
    for key in tqdm(forward_data, desc='Processing...'):
        for val in forward_data[key]:
            if val not in data:
                data[val] = {key}
            else:
                data[val].add(key)
    data = postprocess_data(data)

    print('Writing back-transliterations to', output_file)
    pretty_write_json(data, output_file)

def convert_news_to_json(news_xml, output_file, output_reverse_file, phrase_mode=False):
    if not os.path.isfile(news_xml):
        sys.exit('ERROR: XML Not Found: ', news_xml)
    data = read_xml_dataset(news_xml, phrase_mode)
    data = postprocess_data(data)
    
    print('Writing forward transliterations to', output_file)
    pretty_write_json(data, output_file)

    if output_reverse_file:
        save_reverse_data(data, output_reverse_file)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--news-xml", required=True, type=str)
    parser.add_argument("--output", required=True, type=str)
    parser.add_argument("--output-reverse", type=str)
    parser.add_argument('--phrase-mode', action='store_true')
    args = parser.parse_args()
    convert_news_to_json(args.news_xml, args.output, args.output_reverse, args.phrase_mode)