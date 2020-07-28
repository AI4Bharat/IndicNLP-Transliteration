import h5py
import json
import numpy as np
from annoy import AnnoyIndex

def create_annoy_index_from_hdf5(lang,
                                voc_json_file, hdf5_file,
                                vec_sz =300,
                                save_prefix= None):
        print("Creating Annoy tree object")
        words = json.load(open(voc_json_file, encoding="utf-8"))

        embeds = h5py.File(hdf5_file, "r")['/'+lang]
        t = AnnoyIndex(vec_sz, 'angular')  # Length of item vector that will be indexed
        for i, w in enumerate(words):
            v = embeds[w][0,:]
            t.add_item(i, v)
        t.build(10)

        if save_prefix:
            t.save(save_prefix +lang+'_word_vec.annoy')

        return t


def create_annoy_index_from_model(voc_json_file, glyph_obj,
                                model_func,
                                vec_sz = 300,
                                save_prefix = None):
        print("Creating Annoy tree object")
        words = json.load(open(voc_json_file, encoding="utf-8"))

        t = AnnoyIndex(vec_sz, 'angular')  # Length of item vector that will be indexed
        for i, w in enumerate(words):
            xv = glyph_obj.word2xlitvec(w)
            v = model_func(xv)
            t.add_item(i, v)
        t.build(10)

        if save_prefix:
            t.save(save_prefix + '_word_vec.annoy')

        return t



## ---------- Annoy Handler ----------------------------------------------------

class AnnoyStrawboss():
    """
    Annoy object creation;
    """
    def __init__(self, voc_json_file, char_emb_pkl = None,
                annoy_tree_path = None,
                vec_sz = 300,):
        """
        voc-json_file: Vocab file with language object
        annoy_tree_obj: annoy index based search object to be laoded directly
        mode: {'compose', 'readfromfile'}
        """
        self.vec_sz = vec_sz
        self.words = json.load(open(voc_json_file, encoding="utf-8"))

        if char_emb_pkl:
            self.char_emb = pickle.load(open(char_emb_pkl, 'rb'))

        self.annoy_tree_obj = self._load_annoy_index(annoy_tree_path)

    def _load_annoy_index(self, annoy_tree_path):
        u = AnnoyIndex(self.vec_sz, 'angular')
        u.load(annoy_tree_path)
        return u

    def get_nearest_vocab(self, vec, count = 1):
        # vec = np.reshape(vec, newshape = (-1) )
        idx_list = self.annoy_tree_obj.get_nns_by_vector(vec, count)
        word_list = [ self.words[idx] for idx in idx_list]
        return word_list

    def get_nearest_vocab_details(self, vec, count =1):
        idx_list = self.annoy_tree_obj.get_nns_by_vector(vec, count, include_distances=True)
        word_list = [ self.words[idx] for idx in idx_list[0] ]
        return word_list, idx_list[1]

    def chars_to_nearest_vocab(self, word_list):
        '''
        gets a word and uses pre loaded char embedding to create the word embedding
        and then finds the nearest neighbour
        '''
        out_list = []
        for word in word_list:
            word_emb = np.zeros(shape = self.vec_sz, dtype = np.float32)
            for c in word:
                word_emb = word_emb + self.char_emb.get(c, 0)

            out_list += self.get_nearest_vocab(word_emb)
        return out_list

