import numpy as np
import os
import torch
import esm
import sys
import h5py
import warnings

warnings.filterwarnings("ignore")




FastaName = "ISSB26_181.fasta"
ESM1vFolderName = "ESM1v_181_ISSB26"
LabelName = "ISSB26_181_label.npy"
ESM1vFeatureName = "ISSB26_esm1v_181.npy"
PTH5FilePath = "PT"
PTH5FileName = "ISSB26_embeddings_r181.h5"
PTFeatureName = "ISSB26_PT_181.npy"

project_path = "D:\\MutsExperiment\\IDSB70"
rootpath = os.path.abspath(project_path)
sys.path.append(rootpath)

FASTA_PATH = (rootpath + f"\\{FastaName}")
EMB_PATH = (rootpath + f"\\{ESM1vFolderName}")
EMB_LAYER = 33

ys = []
Xs = []
headerdd = []
for header, _seq in esm.data.read_fasta(FASTA_PATH):
    scaled_effect = header.split('_')[2]
    ys.append(scaled_effect)
    header = header.replace('|', '_').replace(
        '\\', '_').replace('/', '_').replace('*', '_').replace(':', '_').replace(
        '?', '_')
    header2=header.replace('.', '_')
    headerdd.append(header2)
    fn = f'{EMB_PATH}/{header[0:]}.pt'
    embs = torch.load(fn)
    Xs.append(embs['representations'][EMB_LAYER])

Xs = torch.stack(Xs, dim=0).numpy()
ys = np.array(ys)

print(Xs.shape)
print(ys.shape)
print(ys)

np.save(rootpath + f'\\features_npy\\{LabelName}', ys)
np.save(rootpath + f'\\features_npy\\{ESM1vFeatureName}', Xs)

print("esm1v!")


keys=[]
with h5py.File(rootpath + f"\\{PTH5FilePath}\\{PTH5FileName} ", 'r') as f:
    embeddings = []
    for ind in headerdd:
        for key in f.keys():
            if ind == key:
                keys.append(key)
                embeddings.append(f[key][:])

    embeddings = np.array(embeddings)
    print(embeddings.shape)

print("esm_header:",headerdd[1:25])
print("pt_header:",keys[1:25])

np.save(rootpath + f'\\features_npy\\{PTFeatureName}', embeddings)
print("prottrans!")
