import sys
import os
import random
from torch.utils.data import Dataset
import numpy as np

NP_TYPE = np.int64

##====== Unicodes ==============================================================
'''
Note: Replace dummy chars for adding new characters
      Don't change the order of the Characters in lists
'''
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

#-------------------------------------------------------------------------------

class GlyphStrawboss():
    def __init__(self, lang = 'en'):
        """ list of letters in a language in unicode
        lang: ISO Language code
        """
        self.lang = lang
        if lang == 'en':
            self.glyphs = indoarab_numeric + english_smallcase
        elif lang in ['hi', 'knk']:
            self.glyphs = indoarab_numeric + devanagari_scripts

        self.char2idx = {}
        self.idx2char = {}
        self._create_index()

    def _create_index(self):

        self.char2idx['_'] = 0  #pad
        self.char2idx['$'] = 1  #start
        self.char2idx['#'] = 2  #end
        self.char2idx['*'] = 3  #Mask
        self.char2idx['&'] = 4  #unused
        self.char2idx['%'] = 5  #unused
        self.char2idx['!'] = 6  #unused

        # letter to index mapping
        for idx, char in enumerate(self.glyphs):
            self.char2idx[char] = idx + 7 # +6 token initially

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

            vec = np.asarray(vec, dtype=NP_TYPE)
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
        if file_map == "LangEn":
            tgt_str, src_str = self._json2_x_y(json_file)
        elif file_map == "EnLang":
            src_str, tgt_str = self._json2_x_y(json_file)
        else:
            raise Exception('Unknown JSON structure')

        self.src_glyph = src_glyph_obj
        self.tgt_glyph = tgt_glyph_obj

        __svec = self.src_glyph.word2xlitvec
        __tvec = self.tgt_glyph.word2xlitvec
        self.src = [ __svec(s)  for s in src_str]
        self.tgt = [ __tvec(s)  for s in tgt_str]

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


    def _json2_x_y(self, json_file):
        ''' Convert JSON lang pairs to Key-Value lists with indexwise one2one correspondance
        '''
        import json
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
        padded = np.zeros((max_len), dtype=NP_TYPE)
        if len(x) > max_len: padded[:] = x[:max_len]
        else: padded[:len(x)] = x
        return padded


class MonoLMData(Dataset):
    """ Monolingual dataloader for Language Model
    CSV or txt (or) if JSON keys used
    depends on: Numpy, Pandas, random
    """
    def __init__(self, glyph_obj, data_file,
                    padding = True, mask_ratio = 0.3, # ratio of characters to be masked
                 ):
        """
        padding: Set True if Padding with zeros is required for Batching
        """
        extension = os.path.splitext(data_file)[-1]
        if extension == ".json":
            src_lst = self._json2_x(data_file)
        elif extension == ".txt" or extension == ".csv":
            src_lst = self._csv2_x(data_file, extension)
        else:
            raise Exception('Unknown File Extension')


        self.glyph_obj = glyph_obj
        self.mask_ratio = mask_ratio
        self.padding = padding
        #TODO: If memory insufficient for dataset move word2xlitvec to __getitem__
        __svec = self.glyph_obj.word2xlitvec
        self.src = [ __svec(s) for s in src_lst]
        self.mask_idx = glyph_obj.char2idx['*']
        self.max_seq_size = max(len(t) for t in self.src) #s_tok, e_tok

    def __getitem__(self, index):
        x_sz = len(self.src[index])
        x = self.src[index]
        if self.padding:
            x = self._pad_sequence(x, self.max_seq_size)
        x_mkd = self._rand_seq_mask(x.copy(), x_sz)

        return x_mkd, x, x_sz

    def __len__(self):
        return len(self.src)

    def _rand_seq_mask(self, arr, arr_sz):
        m_seq = random.sample(range(1, arr_sz), int(arr_sz*self.mask_ratio) )
        arr[m_seq] = self.mask_idx
        return arr

    def _pad_sequence(self, x, max_len):
        """ Pad sequence to maximum length;
        Pads zero if word < max
        Clip word if word > max
        """
        padded = np.zeros((max_len), dtype=NP_TYPE)
        if len(x) > max_len: padded[:] = x[:max_len]
        else: padded[:len(x)] = x
        return padded

    def _json2_x(self, json_file):
        ''' Read Mono data from JSON lang pairs
        '''
        import json
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
