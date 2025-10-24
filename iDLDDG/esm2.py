import numpy as np
import os
import torch
import esm
import sys
import h5py
import warnings
warnings.filterwarnings("ignore")

"""
将esm模型提取的特征划分为训练集和测试集（划分随机种子设置为42），并且将训练集和测试集保存为.npy文件
"""
project_path = "E:\Projects\FCMSTrans"
rootpath = os.path.abspath(project_path)
sys.path.append(rootpath)
FASTA_PATH ="E:\DataSets\Metal\Feature\ESM\MIX_cut\Mixo_161.fasta"
EMB_PATH = "E:\DataSets\Metal\Feature\ESM\MIX_Feature\leno_161"
EMB_LAYER = 33

ys = []
Xs = []
esm_header = []
for header, _seq in esm.data.read_fasta(FASTA_PATH):
    scaled_effect = header.split('_')[1]
    esm_header.append(header.split('_')[0] + '_' + header.split('_')[1] + '_' + header.split('_')[2])
    ys.append(float(scaled_effect))
    header = header.replace('|', '_').replace('\\', '_').replace('/', '_').replace('*', '_').replace(':', '_').replace(
        '?', '_')
    # fn = f'{EMB_PATH}/{header[1:]}.pt'
    fn = f'{EMB_PATH}/{header[0:]}.pt'
    embs = torch.load(fn)
    # Xs.append(embs['mean_representations'][EMB_LAYER])
    Xs.append(embs['representations'][EMB_LAYER])

Xs = torch.stack(Xs, dim=0).numpy()
ys = np.array(ys)
esm_header = np.array(esm_header)
print(ys.shape,Xs.shape)

np.save(rootpath + '/features_npy/labels/MIX/161/MIXO_labels.npy', ys)
np.save(rootpath + '/features_npy/esm/MIX/161/MIXO_esm.npy',Xs)

print("esm特征文件保存完成！")
