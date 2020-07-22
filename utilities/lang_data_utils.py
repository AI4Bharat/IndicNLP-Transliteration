import sys
import os
import random
import json
import h5py
import pickle
from torch.utils.data import Dataset
from annoy import AnnoyIndex
import numpy as np

NP_TYPE = np.int64

##====== Unicodes ==============================================================
'''
Note: Replace dummy chars for adding new characters
      Don't change the order of the Characters in lists
'''
''' Replaced with compact version
indoarab_numeric = [chr(alpha) for alpha in range(48, 58)]

english_smallcase = [chr(alpha) for alpha in range(97, 123)] + [
    "'",         # apostrophe U+0027
    chr(0x2654),chr(0x2655),chr(0x2656),chr(0x2657),chr(0x2658),chr(0x2659), # Dummy placeholder for future addition
]
devanagari_scripts =  [chr(alpha) for alpha in range(2304, 2432)] + [
    chr(0x200c), # ZeroWidth-NonJoiner U+200c
    chr(0x200d), # ZeroWidthJoiner U+200d
    chr(0x265a),chr(0x265b),chr(0x265c),chr(0x265e),chr(0x265e),chr(0x265f), # Dummy placeholders for future addition
]
'''
# ----

indoarab_num = [chr(alpha) for alpha in range(48, 58)]

english_smallcase = [chr(alpha) for alpha in range(97, 123)]

devanagari_scripts = ['ऄ', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ','ऍ', 'ऎ', 'ए', 'ऐ',
    'ऑ', 'ऒ', 'ओ', 'औ','ऋ','ॠ','ऌ','ॡ','ॲ', 'ॐ',
    'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण',
    'त', 'थ', 'द', 'ध', 'न', 'ऩ', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ऱ', 'ल',
    'ळ', 'ऴ', 'व', 'श', 'ष', 'स', 'ह', 'क़', 'ख़', 'ग़', 'ज़', 'ड़', 'ढ़', 'फ़', 'य़',
    '्', 'ा', 'ि', 'ी', 'ु', 'ू', 'ॅ', 'ॆ', 'े', 'ै', 'ॉ', 'ॊ', 'ो', 'ौ',
    'ृ', 'ॄ', 'ॢ', 'ॣ', 'ँ', 'ं', 'ः', '़', '॑',  'ऽ',

    chr(0x200c), # ZeroWidth-NonJoiner U+200c
    chr(0x200d), # ZeroWidthJoiner U+200d
]



## ------------ Glyph handlers -------------------------------------------------

class GlyphStrawboss():
    def __init__(self, lang = 'en'):
        """ list of letters in a language in unicode
        lang: ISO Language code
        """
        self.lang = lang
        if lang == 'en':
            self.glyphs = english_smallcase
        elif lang in ['hi', 'gom', 'mai']:
            self.glyphs = devanagari_scripts

        self.char2idx = {}
        self.idx2char = {}
        self._create_index()

    def _create_index(self):

        self.char2idx['_'] = 0  #pad
        self.char2idx['$'] = 1  #start
        self.char2idx['#'] = 2  #end
        self.char2idx['*'] = 3  #Mask
        self.char2idx["'"] = 4  #apostrophe U+0027
        self.char2idx['%'] = 5  #unused
        self.char2idx['!'] = 6  #unused

        # letter to index mapping
        for idx, char in enumerate(self.glyphs):
            self.char2idx[char] = idx + 7 # +7 token initially

        # index to letter mapping
        for char, idx in self.char2idx.items():
            self.idx2char[idx] = char

    def size(self):
        return len(self.char2idx)


    def word2xlitvec(self, word):
        """ Converts given string of gyphs(word) to vector(numpy)
        Also adds tokens for start and end
        """
        try:
            vec = [self.char2idx['$']] #start token
            for i in list(word):
                vec.append(self.char2idx[i])
            vec.append(self.char2idx['#']) #end token

            vec = np.asarray(vec, dtype=np.int64)
            return vec

        except Exception as error:
            print("Error In word:", word, "Error Char not in Token:", error)
            sys.exit()

    def xlitvec2word(self, vector):
        """ Converts vector(numpy) to string of glyphs(word)
        """
        char_list = []
        for i in vector:
            char_list.append(self.idx2char[i])

        word = "".join(char_list).replace('$','').replace('#','') # remove tokens
        word = word.replace("_", "").replace('*','') # remove tokens
        return word

## ---------- Vocab handlers ---------------------------------------------------

class VocableStrawboss():
    def __init__(self, json_file ):
        """
        word vocab ID generator
        """
        # self.lang = lang
        with open(json_file, 'r', encoding = "utf-8") as f:
            self.words = json.load(f)

        self.word2idx = {}
        self.idx2word = {}
        self._create_index()

    def _create_index(self):

        self.word2idx['<PAD>'] = 0  #pad
        self.word2idx['<SRT>'] = 1  #start
        self.word2idx['<END>'] = 2  #end
        self.word2idx['<MSK>'] = 3  #Mask
        self.word2idx['<UNK>'] = 4  #Unknown Word

        # letter to index mapping
        for idx, word in enumerate(self.words):
            self.word2idx[word] = idx + 5 # +5 token initially

        # index to letter mapping
        for word, idx in self.word2idx.items():
            self.idx2word[idx] = word

    def size(self):
        return len(self.word2idx)

    def get_idx(self, word):
        return self.word2idx.get(word, 4)
    def get_word(self, idx):
        return self.idx2word.get(int(idx), '<UNK>')


class VocabSanitizer():
    def __init__(self, data_file):
        '''
        data_file: path to file conatining vocabulary list
        '''
        extension = os.path.splitext(data_file)[-1]
        if extension == ".json":
            voc_list_ = json.load(open(data_file))
        else:
            print("Only Json file extension supported")

        self.vocab_set = set(voc_list_)

    def remove_astray(self, word_list):
        '''Remove words that are not present in vocabulary
        '''
        new_list = []
        for v in word_list:
            if v in self.vocab_set:
                new_list.append(v)
        if new_list == []:
            return word_list.copy()
            # return [" "]
        return new_list

    def reposition(self, word_list):
        '''Reorder Words in list
        '''
        new_list = []
        temp_ = word_list.copy()
        for v in word_list:
            if v in self.vocab_set:
                new_list.append(v)
                temp_.remove(v)
        new_list.extend(temp_)

        return new_list


##======== Data Reading ==========================================================

class XlitData(Dataset):
    """ Backtransliteration from English to Native Language
    JSON format only
    depends on: Numpy
    """
    def __init__(self, src_glyph_obj, tgt_glyph_obj,
                    json_file, file_map = "LangEn",
                    padding = True, max_seq_size = None,
                 ):
        """
        padding: Set True if Padding with zeros is required for Batching
        max_seq_size: Size for Padding both input and output, Longer words will be truncated
                      If unset computes maximum of source, target seperate
        """
        #Load data
        if file_map == "LangEn": # output-input
            tgt_str, src_str = self._json2_k_v(json_file)
        elif file_map == "EnLang": # input-output
            src_str, tgt_str = self._json2_k_v(json_file)
        else:
            raise Exception('Unknown JSON structure')

        self.src_glyph = src_glyph_obj
        self.tgt_glyph = tgt_glyph_obj

        __svec = self.src_glyph.word2xlitvec
        __tvec = self.tgt_glyph.word2xlitvec
        self.src = [ __svec(s)  for s in src_str]
        self.tgt = [ __tvec(s)  for s in tgt_str]

        self.tgt_class_weights = self._char_class_weights(self.tgt)

        self.padding = padding
        if max_seq_size:
            self.max_tgt_size = max_seq_size
            self.max_src_size = max_seq_size
        else:
            self.max_src_size = max(len(t) for t in self.src)
            self.max_tgt_size = max(len(t) for t in self.tgt)

    def __getitem__(self, index):
        x_sz = len(self.src[index])
        y_sz = len(self.tgt[index])
        if self.padding:
            x = self._pad_sequence(self.src[index], self.max_src_size)
            y = self._pad_sequence(self.tgt[index], self.max_tgt_size)
        else:
            x = self.src[index]
            y = self.tgt[index]
        return x,y, x_sz

    def __len__(self):
        return len(self.src)


    def _json2_k_v(self, json_file):
        ''' Convert JSON lang pairs to Key-Value lists with indexwise one2one correspondance
        '''
        with open(json_file, 'r', encoding = "utf-8") as f:
            data = json.load(f)

        x = []; y = []
        for k in data:
            for v in data[k]:
                x.append(k); y.append(v)

        return x, y


    def _pad_sequence(self, x, max_len):
        """ Pad sequence to maximum length;
        Pads zero if word < max
        Clip word if word > max
        """
        padded = np.zeros((max_len), dtype=np.int64)
        if len(x) > max_len: padded[:] = x[:max_len]
        else: padded[:len(x)] = x
        return padded

    def _char_class_weights(self, x_list, scale = 10):
        """For handling class imbalance in the characters
        Return: 1D-tensor will be fed to CEloss weights for error calculation
        """
        from collections import Counter
        full_list = []
        for x in x_list:
            full_list += list(x)
        count_dict = dict(Counter(full_list))

        class_weights = np.ones(self.tgt_glyph.size(), dtype = np.float32)
        for k in count_dict:
            class_weights[k] = (1/count_dict[k]) * scale

        return class_weights



class MonoCharLMData(Dataset):
    """ Monolingual dataloader for Language Model Character Level prediction
    CSV or txt (or) if JSON keys used
    depends on: Numpy, Pandas, random
    """
    def __init__(self, glyph_obj, data_file,
                    input_type = 'plain',
                    padding = True, ):
        """
        padding: Set True if Padding with zeros is required for Batching
        """
        extension = os.path.splitext(data_file)[-1]
        if extension == ".json":
            self.src_lst = self._json2_x(data_file)
        elif extension == ".txt" or extension == ".csv":
            self.src_lst = self._csv2_x(data_file, extension)
        else:
            raise Exception('Unknown File Extension')

        self.glyph_obj = glyph_obj
        self.glyph_sz = glyph_obj.size()
        self.padding = padding
        self.max_seq_size = max( (len(t)+2) for t in self.src_lst) #s_tok, e_tok
        self.ambiguity_ratio = 0.25 # percentage of misspellings

        self._get_function = self._compose_corrupt_input if input_type == "corrupt" \
                                else self._plain_read_input

    def __getitem__(self, index):
        '''
        x =   [$, g, e, o, #]
        tgt = [$, g, e, o, #]
        '''
        return self._get_function(index)

    def __len__(self):
        return len(self.src_lst)

    def _plain_read_input(self, index):
        x = self.glyph_obj.word2xlitvec(self.src_lst[index])
        x_sz = len(x)
        if self.padding:
            x = self._pad_sequence(x, self.max_seq_size)
        tgt = x
        return x, tgt, x_sz

    def _compose_corrupt_input(self, index):
        x = self.glyph_obj.word2xlitvec(self.src_lst[index])
        x_sz = len(x)
        if self.padding:
            x = self._pad_sequence(x, self.max_seq_size)
        tgt = x
        x_mkd = self._rand_char_insert(x.copy(), x_sz)

        return x_mkd, tgt, x_sz

    def _rand_char_insert(self, arr, arr_sz):
        m_seq = random.sample(range(1, arr_sz), int(arr_sz * self.ambiguity_ratio) )
        arr[m_seq] = random.sample(range(7,self.glyph_sz ), len(m_seq) )
        return arr


    def _pad_sequence(self, x, max_len):
        """ Pad sequence to maximum length;
        Pads zero if word < max
        Clip word if word > max
        """
        padded = np.zeros((max_len), dtype=np.int64)
        if len(x) > max_len: padded[:] = x[:max_len]
        else: padded[:len(x)] = x
        return padded

    def _json2_x(self, json_file):
        ''' Read Mono data from JSON lang pairs
        '''
        with open(json_file, 'r', encoding = "utf-8") as f:
            data = json.load(f)
        x = set()
        for k in data:
            x.add(k)
        return list(x)

    def _csv2_x(self, csv_file, extension = None):
        ''' Read Mono data from csv/txt
        Assumed that .txt will not have column header
        '''
        import pandas
        if extension == ".txt":
            df = pandas.read_csv(csv_file, header = None)
        else:
            df = pandas.read_csv(csv_file)
        x = df.iloc[:,0]
        return list(x)

class MonoVocabLMData(Dataset):
    """
    Vocab based LM training model; Multinominal word level prediction
    depends on VocableStrawboss object
    """
    def __init__(self, glyph_obj, vocab_obj, json_file,
                    input_type = "compose",
                    padding = True,
                    max_seq_size = 50,
                 ):
        """
        input_type: {'compose', 'readfromfile'}
                    compose: create masked inputs from truths
                    readfromfile: read the corresponding inputs from file
        json_file: correct:[predict] format created by compose_corr_dataset
        """

        self.glyph_obj = glyph_obj
        self.glyph_sz = glyph_obj.size()
        self.vocab_obj = vocab_obj
        self.padding = padding
        self.max_seq_size = max_seq_size
        self.ambiguity_ratio = 0.3 # percentage of misspellings

        if input_type == 'compose':
            self.crrt_str = self._json2_x(json_file)
            self.crrt_str += self._garbage_tokens(count = 5)
        elif input_type == 'readfromfile':
            self.crrt_str, self.misp_str = self._json2_k_v(json_file)

        self._get_function = self._compose_corrupt_input if input_type == "compose" \
                                else self._plain_read_input

    def __getitem__(self, index):
        return self._get_function(index)

    def __len__(self):
        return len(self.crrt_str)

    def _plain_read_input(self, index):
        x = self.glyph_obj.word2xlitvec(self.misp_str[index])
        x = x[:self.max_seq_size] # truncate to max len
        x_sz = len(x)
        if self.padding:
            x = self._pad_sequence(x, self.max_seq_size)
        x_word_idx = self.vocab_obj.get_idx(self.crrt_str[index])
        return x, x_word_idx, x_sz

    def _compose_corrupt_input(self, index):
        x = self.glyph_obj.word2xlitvec(self.crrt_str[index])
        x = x[:self.max_seq_size] # truncate to max len
        x_sz = len(x)
        if self.padding:
            x = self._pad_sequence(x, self.max_seq_size)
        x_mkd = self._rand_char_insert(x.copy(), x_sz)
        x_word_idx = self.vocab_obj.get_idx(self.crrt_str[index])
        return x_mkd, x_word_idx, x_sz

    def _rand_char_insert(self, arr, arr_sz):
        m_seq = random.sample(range(1, arr_sz), int(arr_sz * self.ambiguity_ratio) )
        arr[m_seq] = random.sample(range(7,self.glyph_sz ), len(m_seq) )
        return arr

    def _pad_sequence(self, x, max_len):
        """ Pad sequence to maximum length;
        Pads zero if word < max
        Clip word if word > max
        """
        padded = np.zeros((max_len), dtype=np.int64)
        if len(x) > max_len: padded[:] = x[:max_len]
        else: padded[:len(x)] = x
        return padded

    def _json2_k_v(self, json_file):
        ''' Convert JSON lang pairs to Key-Value lists with indexwise one2one correspondance
        '''
        with open(json_file, 'r', encoding = "utf-8") as f:
            data = json.load(f)

        x = []; y = []
        for k in data:
            for v in data[k]:
                x.append(k); y.append(v)

        return x, y

    def _json2_x(self, json_file):
        ''' Read Mono data from JSON lang pairs
        '''
        with open(json_file, 'r', encoding = "utf-8") as f:
            data = json.load(f)
        x = set()
        for k in data:
            x.add(k)
        return list(x)

    def _garbage_tokens(self, count = 1):
        gtk_lst = []
        for i in range(count):
            t = "".join(random.sample(self.glyph_obj.char2idx.keys(), random.randint(5,20) ))
            gtk_lst.append(t)
        return gtk_lst



class MonoEmbedLMData(Dataset):
    """
    Vocab Embed based LM training model;  word embedding prediction
    """
    def __init__(self, glyph_obj, lang, hdf5_file,
                    json_file = None,
                    input_type = "compose",
                    padding = True,
                    max_seq_size = 50,
                 ):
        """
        input_type: {'compose', 'readfromfile'}
                    compose: create masked inputs from truths
                    readfromfile: read the corresponding inputs from file
        json_file: correct:[predict] format created by compose_corr_dataset
        """

        self.glyph_obj = glyph_obj
        self.glyph_sz = glyph_obj.size()
        self.padding = padding
        self.max_seq_size = max_seq_size
        self.ambiguity_ratio = 0.3 # percentage of misspellings
        h5_array = h5py.File(hdf5_file, "r")
        self.embed_h5 = h5_array["/"+lang]

        if input_type == 'readfromfile':
            if json_file == None:
                raise "`readfromfile` mode requies JSON file"
            self.crrt_str, self.misp_str = self._json2_k_v(json_file)
        elif input_type == 'compose':
            self.crrt_str = list(self.embed_h5.keys())

        self._get_function = self._compose_corrupt_input if input_type == "compose" \
                                else self._plain_read_input

    def __getitem__(self, index):
        return self._get_function(index)

    def __len__(self):
        return len(self.crrt_str)

    def _plain_read_input(self, index):
        x = self.glyph_obj.word2xlitvec(self.misp_str[index])
        x = x[:self.max_seq_size] # truncate to max len
        x_sz = len(x)
        if self.padding:
            x = self._pad_sequence(x, self.max_seq_size)
        x_wordemb = self.embed_h5[ self.crrt_str[index] ][0,:]
        return x, x_wordemb, x_sz

    def _compose_corrupt_input(self, index):
        x = self.glyph_obj.word2xlitvec(self.crrt_str[index])
        x = x[:self.max_seq_size] # truncate to max len
        x_sz = len(x)
        if self.padding:
            x = self._pad_sequence(x, self.max_seq_size)
        x_mkd = self._rand_char_insert(x.copy(), x_sz)
        x_wordemb = self.embed_h5[ self.crrt_str[index] ][0,:]
        return x_mkd, x_wordemb, x_sz

    def _rand_char_insert(self, arr, arr_sz):
        m_seq = random.sample(range(1, arr_sz), int(arr_sz * self.ambiguity_ratio) )
        arr[m_seq] = random.sample(range(7,self.glyph_sz ), len(m_seq) )
        return arr

    def _pad_sequence(self, x, max_len):
        """ Pad sequence to maximum length;
        Pads zero if word < max
        Clip word if word > max
        """
        padded = np.zeros((max_len), dtype=np.int64)
        if len(x) > max_len: padded[:] = x[:max_len]
        else: padded[:len(x)] = x
        return padded

    def _json2_k_v(self, json_file):
        ''' Convert JSON lang pairs to Key-Value lists with indexwise one2one correspondance
        '''
        with open(json_file, 'r', encoding = "utf-8") as f:
            data = json.load(f)

        check_set = set(self.embed_h5)
        x = []; y = []; skipped =0
        for k in data:
            if k not in check_set:
                skipped +=1
                continue
            for v in data[k]:
                x.append(k); y.append(v)

        print("Skipped while Loading:", skipped)
        return x, y

## ----- Correction Dataset -----

def compose_corr_dataset(pred_file, truth_file,
                         save_path = "", topk = 1  ):
    """
    Function to create Json for Correction Network from the truth and predition of models
    Return: Path of the composed file { Correct: [Pred] }
    pred_file: EnLang
    truth_file: LangEn
    topk: Number of topk predictions to be used or composing data
    """
    pred_dict = json.load(open(pred_file))
    truth_dict = json.load(open(truth_file))


    out_dict = {}
    for k in truth_dict:
        temp_set = set()
        for v in truth_dict[k]:
            temp_set.update(pred_dict[v][:topk])
        out_dict[k] = list(temp_set)

    save_file = save_path + "Corr_set_"+os.path.basename(truth_file)
    with open(save_file ,"w", encoding = "utf-8") as f:
        json.dump(out_dict, f, ensure_ascii=False, indent=4, sort_keys=True,)

    return save_file


