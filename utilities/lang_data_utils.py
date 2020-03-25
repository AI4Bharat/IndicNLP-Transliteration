import sys
from torch.utils.data import Dataset
import numpy as np

NP_TYPE = np.int64

##====== Unicodes ==============================================================


indoarab_numeric = [chr(alpha) for alpha in range(48, 58)]
english_smallcase = [chr(alpha) for alpha in range(97, 123)]
devanagari_scripts =  [chr(alpha) for alpha in range(2304, 2432)]

misc_chars = [ # ! Don't Change the sequence order
    chr(8204), # ZeroWidth-NonJoiner U+200c
    chr(8205), # ZeroWidthJoiner U+200d
]

#-------------------------------------------------------------------------------

class GlyphStrawboss():
    def __init__(self, lang = 'en'):
        """ list of letters in a language in unicode
        lang: ISO Language code
        """
        self.lang = lang
        if lang == 'en':
            self.glyphs = english_smallcase + indoarab_numeric
        elif lang in ['hi']: #TODO: Move misc to last
            self.glyphs = misc_chars + devanagari_scripts + indoarab_numeric

        self.glyph_size = len(self.glyphs)
        self.char2idx = {}
        self.idx2char = {}
        self._create_index()

    def _create_index(self):

        self.char2idx['_'] = 0  #pad
        self.char2idx['$'] = 1  #start
        self.char2idx['#'] = 2  #end

        # letter to index mapping
        for idx, char in enumerate(self.glyphs):
            self.char2idx[char] = idx + 3 # +3 token initially

        # index to letter mapping
        for char, idx in self.char2idx.items():
            self.idx2char[idx] = char

    def size(self):
        return self.glyph_size


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
            print("Error In word:", word, "Erro:", error)
            sys.exit()

    def xlitvec2word(self, vector):
        """ Converts vector(numpy) to string of glyphs(word)
        """
        char_list = []
        for i in vector:
            char_list.append(self.idx2char[i])

        word = "".join(char_list).replace('$','').replace('#','') # remove tokens
        return word





##======== Data Reading ==========================================================

class XlitData(Dataset):
    """ Backtransliteration from English to Native Language
    JSON format only
    depends on: Numpy
    """
    def __init__(self, src_glyph_obj, tgt_glyph_obj,
                    json_file, file_map = "LangEn",
                    padding = True,
                 ):
        """
        padding: Set True if Padding with zeros is required for Batching
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
        return x,y, x_sz, y_sz

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


