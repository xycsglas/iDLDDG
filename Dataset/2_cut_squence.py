import os
import sys
import pandas as pd
import openpyxl
# from excel_to_txt import mkdir
import numpy as np



def cut_seq(Sequence, pos, PSA_length):
    posQ = Sequence[:int(pos)-1]
    posH = Sequence[int(pos):]
    var = Sequence[int(pos)-1]



    if (len(posQ) < int(0.5 * PSA_length) and len(posH) < int(0.5 * PSA_length)):
        tempQ = 'X' * (int(0.5 * PSA_length) - len(posQ))
        posQN = tempQ + posQ
        tempH = 'X' * (int(0.5 * PSA_length) - len(posH))
        posHN = posH + tempH

    elif (len(posQ) >= int(0.5 * PSA_length) and len(posH) < int(0.5 * PSA_length)):
        tempQ = posQ[-int(0.5 * PSA_length):]
        posQN = tempQ
        tempH = 'X' * (int(0.5 * PSA_length) - len(posH))
        posHN = posH + tempH

    elif (len(posQ) < int(0.5 * PSA_length) and len(posH) >= int(0.5 * PSA_length)):
        tempQ = 'X' * (int(0.5 * PSA_length) - len(posQ))
        posQN = tempQ + posQ
        tempH = posH[:int(0.5 * PSA_length)]
        posHN = tempH

    else:
        tempQ = posQ[-int(0.5 * PSA_length):]
        posQN = tempQ
        tempH = posH[:int(0.5 * PSA_length)]
        posHN = tempH

    SequenceN = posQN + var + posHN
    return (SequenceN)


if __name__ == "__main__":


    fasta_dir = f"./ISSB26.fasta"
    seq_file = open(fasta_dir, "r")
    out_file = open(f"./ISSB26_181.fasta","w")

    tou = []
    pos = []
    seq = []
    for lineID, line in enumerate(seq_file):
        if lineID == 0:
            tou.append(line)
            line = line.split('_')
            pos.append(line[3][1:-2])
        elif lineID % 2 == 1:
            line = line.replace('\n', '')
            seq.append(line)
            # print(len(seq[0]))
        elif lineID % 2 == 0:
            tou.append(line)
            line = line.split('_')
            pos.append(line[3][1:-2])

    cut_len = 180

    seq_cut = []
    for index, sequence in enumerate(seq):
        seq_cut.append(cut_seq(seq[index], pos[index], cut_len))
        # print(len(seq_cut[0]))


    for i in range(len(seq_cut)):
        out_file.write(tou[i])
        out_file.write(seq_cut[i] + '\n')

    seq_file.close()
    out_file.close()




