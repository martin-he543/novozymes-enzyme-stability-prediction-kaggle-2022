import Levenshtein
from Levenshtein import distance as levenshtein_distance
import numpy as np
import pandas as pd

# seq_id, protein_sequence, pH, data_source, tm\
#      = np.genfromtxt("sample_data/train.csv",skip_header=1,unpack=True,delimiter=",")


base = 'VPVNPEPDATSVENVALKTGSGDSQSDPIKADLEVKGQSALPFDVDCWAILCKGAPNVLQRVNEKTKNSNRDRSGANKGPFKDPQKWGIKALPPKNPSWSAQDFKSPEEYAFASSLQGGTNAILAPVNLASQNSQGGVLNGFYSANKVAQFDPSKPQQTKGTWFQITKFTGAAGPYCKALGSNDKSVCDKNKNIAGDWGFDPAKWAYQYDEKNNKFNYVGK'

trainDF = pd.read_csv("sample_data/train.csv")
listOutput = []
for i in range(len(trainDF)):
    rowProteinSequence = trainDF.iloc[i].protein_sequence ## Taking a row
    LevenstreinDist = Levenshtein.editops(rowProteinSequence, base) ## Using Levenshtein
    listOutput.append(LevenstreinDist)

print(LevenstreinDist)