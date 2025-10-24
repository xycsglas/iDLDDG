# iDLDDG
This is the official repo for the paper: "iDLDDG: predicting protein stability changes from missense mutations in DNA-binding proteins using integrated deep learning features "

<img width="715" height="621" alt="image" src="https://github.com/user-attachments/assets/21d64f89-e1bc-453f-b9f9-91c8f307070f" />

# Environment Setup
Create a new conda environment first:

```
conda create --name iDLDDG python=3.8
```
This will create an anaconda environment
Activate this environment by running:

```
conda activate iDLDDG
```
then install dependencies provided by this repo:

```
pip install -r ./requirements.txt
```

# Data Preprocessing and Feature Extraction

<img width="871" height="244" alt="image" src="https://github.com/user-attachments/assets/4a9e7da6-e6e0-41b2-94ff-071e087de3bc" />

Dataset listed in Table1 can be directly downloaded in this repo.

To attain feature files used to train and evaluate iDLDDG, please go through the following steps:

1. Go to 'Dataset'
2. Run '1_DNAfasta.py' to convert your desired dataset into fasta format file. For Example: MPD552.xlsx -> MPD552.fasta.
3. Run '2_cut_squence.py' to cut sequences in MPD552.fasta into equal length L, default L = 180, which gives you equal length 181 sequences. For example: MPD552.fasta -> Data_181.fasta.
4. Run command lines listed in '3_CommandLine.txt' to get feature embeddings of esm1v, esm2 and ProtTrans.
ESM1v：
```
python /esm-main/scripts/extract.py esm1v_t33_650M_UR90S_1  Data_181.fasta output_path --repr_layers 33 --include per_tok
```

ESM2：
```
python /esm-main/scripts/extract.py esm2_t33_650M_UR50D Data_181.fasta output_path --repr_layers 33 --include per_tok
```

ProtTrans:
```
python ProtTrans-master/Embedding/prott5_embedder.py --input Data_181.fasta --output output_path/S552_inde_embeddings_181.h5
```
please download ProtTrans, ESM2 and ESM1v respectively before you use above commands.

These command lines will produce feature embeddings for ProtTrans, ESM2 and ESM1v.

5. Run '4_feature_npy_esm1v_PT.py' and '5_feature_npy_esm2' to get feature embeddings that will be further used to train and evaluate iDLDDG.

# Train iDLDDG

Run 'train_kfold_cross_validation_old.py', please put feature.npy files into 'feature_npy'

This will train iDLDDG using 10-fold cross validation.

To revise model parameters, please refer to file 'model.py'

# Evaluate iDLDDG

Run 'predict.py', chage the model path correspondingly.

You will need to extract features of 'MPD96' to reproduce iDLDDG independent test results. Please refer to section 'Data Preprocessing and Feature Extraction'.


