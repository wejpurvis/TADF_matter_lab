#Imports

import pickle

from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
import pandas as pd

import random

SEED = 23

random.seed(SEED)
np.random.seed(SEED)

# Data Loading
loaded_data = pd.read_csv("./data/TADF_data_DL.txt", sep ="\t", header=None)
all_data = loaded_data


#Manipulate data using Pandas & RDkit
all_data.columns = ["ID", "SMILES","LUMO", "HOMO", "E(S1)", "E(T1)"]
filt_data = all_data.drop(columns = ["ID", "LUMO", "HOMO"])
filt_data["MOL"] = filt_data["SMILES"].apply(lambda x: Chem.MolFromSmiles(x)) #Add column of Molecular objects
filt_data["CANONICAL SMILES"] = filt_data["MOL"].apply(lambda x: Chem.MolToSmiles(x, canonical = True)) #Add column of Canonical Smiles

def extract_list_of_atoms(mol):
    atoms_list = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return atoms_list

filt_data["Atoms"] = filt_data["MOL"].apply(extract_list_of_atoms)

# Get upper & Lower case letters only
upper_chars = []
lower_chars = []

all_atoms = [atom for atoms_list in filt_data["Atoms"] for atom in atoms_list]
unique_atoms = set(all_atoms)

#@@ replaced by Z
filt_data["Proc. Canon. SMI"] =filt_data["CANONICAL SMILES"].str.replace("@@", "Z")

#Unique chars
all_smiles = " ".join(filt_data["Proc. Canon. SMI"])
unique_chars = set(all_smiles)

#Longest SMILES
longest_smiles = max(filt_data["CANONICAL SMILES"], key=len)
smiles_maxlen = len(longest_smiles)

#Create one-hot encoded matrix
def smiles_encoder(smiles, max_len, unique_char):
    """
    Function defined using all unique characters in the
    processed canonical SMILES.

    Parameters
    ----------
    smiles : str
         SMILES of chromophore in string format.
    unique_char : list
         List of unique characters in the string data set.
    max_len : int
         Maximum length of the SMILES string.

    Returns
    -------
    smiles_matrix : numpy.ndarray
         One-hot encoded matrix of fixed shape
         (unique char in smiles, max SMILES length).
    """
    # create dictionary of the unique char data set
    smi2index = {char: index for index, char in enumerate(unique_char)}
    # one-hot encoding
    # zero padding to max_len
    smiles_matrix = np.zeros((len(unique_char), max_len))
    for index, char in enumerate(smiles):
        smiles_matrix[smi2index[char], index] = 1
    return smiles_matrix

#Apply function to processed canonical SMILES strings
filt_data["OHE_matrix"] = filt_data["Proc. Canon. SMI"].apply(smiles_encoder, max_len = smiles_maxlen, unique_char = unique_chars)

OHE_dataset = filt_data[["OHE_matrix", "E(S1)", "E(T1)"]]

with open("./data/OHE.pkl", "wb") as f:
    pickle.dump(OHE_dataset, f)