'''
For given character/Vocab set run the fasttext embedding model and obtain correspoding embeddings
'''
import h5py
import numpy as np
import json
import sys
import fasttext
import pickle

## ---------
def create_word_embedding_hdf5():

    vocab_list = json.load(open("/home/jupyter/pratham/IndianNLP-Transliteration/data/konkani/gom_all_words_sorted.json"))
    vocab_sz = len(vocab_list)
    print(vocab_sz)

    hdf_file = h5py.File("Gom-additive_vocab_embeddings.hdf5", "w-")
    lang_group = hdf_file.create_group( "gom" )

    for i,v in enumerate(vocab_list):
        dset = lang_group.create_dataset( v, (1, 300),
                                    dtype=np.float32,
                                    chunks=(1, 300),)
        arr = np.zeros(300, dtype = np.float32)
        for c in list(v):
            arr = arr + emb_dict[c]
        arr = np.reshape(arr, (1,300))
        dset[:,:] = arr

    sample = list(lang_group.keys())
    for i in range(5):
        print(lang_group[sample[i]][0,:10])

    hdf_file.close()

## -----

devanagari_list = ['ऄ', 'अ', 'आ', 'इ', 'ई', 'उ', 'ऊ','ऍ', 'ऎ', 'ए', 'ऐ',
    'ऑ', 'ऒ', 'ओ', 'औ','ऋ','ॠ','ऌ','ॡ','ॲ', 'ॐ',
    'क', 'ख', 'ग', 'घ', 'ङ', 'च', 'छ', 'ज', 'झ', 'ञ', 'ट', 'ठ', 'ड', 'ढ', 'ण',
    'त', 'थ', 'द', 'ध', 'न', 'ऩ', 'प', 'फ', 'ब', 'भ', 'म', 'य', 'र', 'ऱ', 'ल',
    'ळ', 'ऴ', 'व', 'श', 'ष', 'स', 'ह', 'क़', 'ख़', 'ग़', 'ज़', 'ड़', 'ढ़', 'फ़', 'य़',
    '्', 'ा', 'ि', 'ी', 'ु', 'ू', 'ॅ', 'ॆ', 'े', 'ै', 'ॉ', 'ॊ', 'ो', 'ौ',
    'ृ', 'ॄ', 'ॢ', 'ॣ', 'ँ', 'ं', 'ः', '़', '॑',  'ऽ',
    chr(0x200c), # ZeroWidth-NonJoiner U+200c
    chr(0x200d), # ZeroWidthJoiner U+200d
]

def character_embeddings(charset):
    ftxt_model = fasttext.load_model("cc.hi.300.bin")
    emb_dict = dict()
    for c in charset:
        emb_dict[c] = ftxt_model[c]
    return emb_dict

if __name__ == "__main__":
    emb_dict = character_embeddings(devanagari_list)

    ## save as pickle
    with open("Gom_char_embed_dict.pkl", 'wb') as f:
        pickle.dump(emb_dict,f)
