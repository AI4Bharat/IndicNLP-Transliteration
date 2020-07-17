import h5py
import numpy as np
import json
import sys
import fasttext

vocab_list = json.load(open("data/konkani/gom_mini_list.json"))
vocab_sz = len(vocab_list)


ftxt_model = fasttext.load_model("cc.hi.300.bin")


hdf_file = h5py.File("Lang-vocab_embedding.hdf5", "w")
lang_group = hdf_file.create_group( "gom" )

for i,v in enumerate(vocab_list):
    dset = lang_group.create_dataset( v, (1, 300),
                                dtype=np.float32,
                                chunks=(1, 300),)
    arr = ftxt_model[v]
    arr = np.reshape(arr, (1,300))
    dset[:,:] = arr

sample = list(lang_group.keys())
for i in range(5):
    print(lang_group[sample[i]][0,:10])

hdf_file.close()