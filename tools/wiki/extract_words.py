'''
Processes the JSONs dumped by WikiExtractor to extract words with its frequency.

PREREQUISITE:
    python WikiExtractor.py --processes 4 -o extracts/ --json gomwiki-20200101-pages-articles-multistream.xml > extracts/logs.txt 2>&1

USAGE:
    Example-1: Extract all unique words with frequency into a CSV
    $ <script.py> --input-folder extracts/ --output-csv word_list.csv --lang konkani
    
    Example-2: Get only top-1000 words of length >= 5
    $ <script.py> --top-k 1000 --min-chars 5 --top-csv top_words.csv --input-folder extracts/ --output-csv word_list.csv --lang konkani
    
'''
import os, sys, json
from glob import glob
from tqdm import tqdm
import unicodedata as ud
from collections import Counter

unicode_numbers = {
    'devanagari': [chr(alpha) for alpha in range(2406, 2416)]
}

unicode_map = {
    'devanagari': [chr(alpha) for alpha in range(2304, 2432)]
}

LANG2SCRIPT = {
    'gom': 'devanagari',
    'hi': 'devanagari'
}

LANG2CODE = {
    'konkani': 'gom', #Goa-Konkani (Wiki-Standard)
    'hindi': 'hi'
}

allowed_categories = []
def tokenize(string, allowed_charset):
    new_string = ''
    for c in string:
        if c in allowed_charset:
            new_string += c
        else:
            # If it's a punctuation, add a space, else ignore the char
            cat = ud.category(c)
            # Refer for cat_codes: fileformat.info/info/unicode/category
            if cat[0] == 'P' or cat[0] == 'Z' or cat == 'Cc':
                new_string += ' '
    return new_string.split()

def extract_txt_words(data_folder, alphabets, word_freq={}):
    # Find .txt files if any and process them line by line
    txt_files = sorted(glob(os.path.join(data_folder, '*.txt')))
    print('Starting to process %d txt files...' % len(txt_files))
    
    uniq_words = set()
    for txt_file in txt_files:
        with open(txt_file) as f:
            lines = f.readlines()
        for line in tqdm(lines, desc='Processing '+txt_file, unit='lines'):
            for word in tokenize(line, alphabets):
                if word not in word_freq: word_freq[word] = 0
                word_freq[word] += 1
                uniq_words.add(word)
    
    print('Extracted %d Unique words from Txt files\n' % len(uniq_words))
    return word_freq

def extract_wiki_words(data_folder, alphabets, word_freq={}):
    # Find files starting with 'wiki_' (which are assumed JSON)
    json_files = sorted(glob(os.path.join(data_folder, 'wiki_*')))
    print('Starting to process %d WikiJSON files...' % len(json_files))
    
    uniq_words = set()
    for json_file in json_files:
        with open(json_file) as f:
            lines = f.readlines()
        # Each line is a JSON, in which the key 'text' contains the Wiki body
        for line in tqdm(lines, desc='Processing '+json_file, unit='lines'):
            # Tokenize the words, and update frequencies
            for word in tokenize(json.loads(line)['text'], alphabets):
                if word not in word_freq: word_freq[word] = 0
                word_freq[word] += 1
                uniq_words.add(word)
    
    print('Extracted %d Unique words from Wiki\n' % len(uniq_words))
    return word_freq

def extract_words(data_folder, output_csv, script_name, allow_numbers=False):
    # Prepare the set of unicode chars that we're interested in
    alphabets = unicode_map[script_name]
    if not allow_numbers:
        numbers = unicode_numbers[script_name]
        alphabets = [c for c in alphabets if c not in numbers]
    
    word_freq = extract_wiki_words(data_folder, alphabets)
    word_freq = extract_txt_words(data_folder, alphabets, word_freq)
    
    print('\nSaving %d words to %s\n' % (len(word_freq), output_csv))
    with open(output_csv, 'w') as f:
        f.write('word,freq\n')
        for word, freq in word_freq.items():
            f.write('%s,%d\n' % (word, freq))
    return word_freq

def get_top_words(word_freq, top_k, min_chars, out_file):
    if min_chars > 0:
        original_len = len(word_freq)
        word_freq = {word:freq for word, freq in word_freq.items() if len(word) >= min_chars}
        current_len = len(word_freq)
        print('Removed %d words from %d since num_chars<%d\n' % ((original_len-current_len), original_len, min_chars))
    word_counter = Counter(word_freq)
    print('Saving top-%d words to %s\n' % (top_k, out_file))
    with open(out_file, 'w') as f:
        f.write('word,freq\n')
        for word, freq in word_counter.most_common(top_k):
            f.write('%s,%d\n' % (word, freq))
    return word_counter

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-folder", required=True, type=str)
    parser.add_argument("--output-csv", required=True, type=str)
    parser.add_argument("--lang", required=True, type=str)
    parser.add_argument("--top-k", type=int, default=-1)
    parser.add_argument("--min-chars", type=int, default=-1)
    parser.add_argument("--top-csv", type=str)
    args = parser.parse_args()
    if args.lang not in LANG2CODE:
        sys.exit('Language:', args.lang, 'not supported')
    if not os.path.isdir(args.input_folder):
        sys.exit('Input Wiki Folder', args.input_folder, 'NOT FOUND!')
    word_freq = extract_words(args.input_folder, args.output_csv, LANG2SCRIPT[LANG2CODE[args.lang]])
    
    # Extract the top-k words and save it
    if args.top_k > 0:
        if not args.top_csv:
            sys.exit('Specify the --top-csv flag to save the results')
        get_top_words(word_freq, args.top_k, args.min_chars, args.top_csv)