from torch.utils.data import TensorDataset
import re, unicodedata, json
import numpy as np
non_eng_letters_regex = re.compile('[^a-zA-Z ]')

def unicode_to_ascii(s):
    """
    Normalizes latin chars with accent to their canonical decomposition
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn')

def clean_eng(s):
    s = unicode_to_ascii(s)
    s = non_eng_letters_regex.sub('', s)
    return s

### sort batch function to be able to use with pad_packed_sequence
def sort_tensorbatch(X, X_lens, Y, Y_, device='cpu'):
    X_lens, indx = X_lens.sort(dim=0, descending=True)
    X = X[indx]
    Y = Y[indx]
    Y_ = Y_[indx]
    return X.transpose(0,1).to(device), X_lens.to(device), Y.to(device), Y_.to(device) # transpose (batch x seq) to (seq x batch)

class Transliteration_Dataset(TensorDataset):
    # A dataset for Eng-to-<lang> transliterations
    def __init__(self, json_file, lang_alphabets, sort_for_pad=True):
        with open(json_file, encoding='utf-8') as f:
            data = json.load(f)
        self.eng_words, self.lang_words = self.process_json(data, lang_alphabets)
        self.start_char, self.end_char, self.pad_char = '$', '#', '*'
        self.eng_vectors, self.eng_vec_lens = self.process_eng(self.eng_words)
        self.lang_vectors, self.lang_vec_lens = self.process_lang(self.lang_words, lang_alphabets)
        self.eng_max_len, self.lang_max_len = max(self.eng_vec_lens), max(self.lang_vec_lens)
        self.padded_eng_vectors = [self.pad_sequence(x, self.eng_max_len) for x in self.eng_vectors]
        self.padded_lang_vectors = [self.pad_sequence(x, self.lang_max_len) for x in self.lang_vectors]
        self.eng_ohe_data = self.vec2ohe(self.padded_eng_vectors, len(self.eng_alpha2index))
        self.lang_ohe_data = self.vec2ohe(self.padded_lang_vectors, len(self.lang_alpha2index))
        
    def __getitem__(self, index):
        # Returns eng_OHE, actual_len(eng_OHE), lang_OHE, lang_indices
        return self.eng_ohe_data[index], self.eng_vec_lens[index], self.lang_ohe_data[index], self.padded_lang_vectors[index]
    
    def __len__(self):
        return len(self.eng_ohe_data)
    
    def clean_lang(self, s, alpha2index):
        # Retain chars only in alpha2index
        cleaned = [c for c in s if c in alpha2index]
        return ''.join(cleaned)
        
    def process_json(self, data, lang_alphabets):
        # Assumes only 1 reference for each eng_word
        eng_words, lang_words = [], []
        for key in data:
            eng_word = clean_eng(key.upper())
            lang_word = self.clean_lang(data[key][0], lang_alphabets)
            if eng_word and lang_word:
                eng_words.append(eng_word)
                lang_words.append(lang_word)
        return eng_words, lang_words
    
    def process_eng(self, words):
        self.eng_alphabets = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
        self.eng_alpha2index = {self.pad_char: 0, self.start_char: 1, self.end_char: 2}
        for index, alpha in enumerate(self.eng_alphabets):
            self.eng_alpha2index[alpha] = index + 3
        return self.vectorize_words(words, self.eng_alpha2index)
    
    def process_lang(self, words, lang_alphabets):
        self.lang_alphabets = lang_alphabets
        self.lang_alpha2index = {self.pad_char: 0, self.start_char: 1, self.end_char: 2}
        for index, alpha in enumerate(lang_alphabets):
            self.lang_alpha2index[alpha] = index + 3
        return self.vectorize_words(words, self.lang_alpha2index)
        
    def vectorize_words(self, words, alpha2index):
        vectors, vector_lengths = [], []
        for word in words:
            word = self.start_char + word + self.end_char
            vector = [alpha2index[char] for char in word]
            vectors.append(vector)
            vector_lengths.append(len(vector))
        return vectors, vector_lengths
    
    def pad_sequence(self, x, max_len):
        padded = np.zeros((max_len), dtype=np.int64)
        if len(x) > max_len: padded[:] = x[:max_len]
        else: padded[:len(x)] = x
        return padded
    
    def vec2ohe(self, vectors, ohe_size):
        data = np.zeros((len(vectors), len(vectors[0]), ohe_size))
        for i, v in enumerate(vectors):
            for j, n in enumerate(v):
                data[i][j][n] = 1
        return data