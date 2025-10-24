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

FastaName = "ISSB26_181.fasta"
ESM2FolderName = "ESM2_181_ISSB26"
ESM2FeatureName = "ISSB26_esm2_181.npy"


project_path = "D:\\MutsExperiment\\IDSB70"
rootpath = os.path.abspath(project_path)
sys.path.append(rootpath)
FASTA_PATH = (rootpath + f"\\{FastaName}")
EMB_PATH = (rootpath + f"\\{ESM2FolderName}")
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

#np.save(rootpath + '\\features_npy\\ssDNA54_181_label.npy', ys)  # 保存标签ddg值数据
np.save(rootpath + f'\\features_npy\\{ESM2FeatureName}', Xs)

print("esm2特征文件保存完成！")
