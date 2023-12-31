import streamlit as st
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from MLP_final import FpMLP

#Load model with associated configurations
def load_model(filename):
    checkpoint = torch.load(filename)
    config = checkpoint["config"]
    
    # Create an instance of your model using the configuration
    model = FpMLP(config)
    
    model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    
    return model, config

loaded_model, config = load_model("MLP_model.pt")

#Run model
def run_model(morgan_fp, model):
    input_tensor = torch.tensor(morgan_fp, dtype=torch.float32)
    #Model in evaluation mode
    model.eval()
    #Pass the input feature through the model to get the predicted output
    with torch.no_grad():
        predicted_output = loaded_model(input_tensor.unsqueeze(0))
    s1_energy = float(predicted_output[0][0])
    t1_energy = float(predicted_output[0][1])

    return s1_energy, t1_energy



def show_predict_page():
    
    st.markdown("<h1 style='text-align: center'>S<sub>1</sub> and T<sub>1</sub> Energy Predictions for TADF Molecules</h1>", unsafe_allow_html=True)

    st.write("""This project was conducted by Will Purvis as part of his summer visit to the Matter Lab (UofT). It aims to predict the S<sub>1</sub> and T<sub>1</sub> energies of TADF molecules for 
             OLEDs at the B3LYP level of theory. The predictions are made using a multi
             layer perceptron (MLP) neural network. Click on model details to read more about the project.
             """, unsafe_allow_html=True)
    st.write("Test the model out by inputing the SMILES of a desire molecule below!")
    st.write("""\n
             (e.g. COC1=CC=CC(=C1)CNC(=O)CSC2=NC3=CC=CC=C3N2C4=CC=CC=C4)
             """)
    
    smiles_placeholder = st.empty()
    smiles_input = st.text_input("Enter a SMILES 👇")
    
    if smiles_input:
        try:
            mol = Chem.MolFromSmiles(smiles_input)
            if mol:
                mfp = AllChem.GetMorganFingerprintAsBitVect(mol, 3, nBits=1024)
                st.write("Molecule:")
                img = Draw.MolToImage(mol)
                st.image(img, width=300, output_format ="PNG")
            else:
                st.error("Invalid SMILES input")
        except Exception as e:
            st.error(f"An error occurred: {e}")
       
    #Submit calculation (must have valid SMILES)
    if "mfp" in locals():
        #Submit button
        if st.button("Calculate energies"):
            #Call run model function
            s1, t1 = run_model(mfp, loaded_model)
            #Display energies
            st.write(f"The predicted S1 energy is <span style = 'color:red; font-weight:bold'> {s1:.4f} </span> eV", unsafe_allow_html=True)
            st.write(f"The predicted T1 energy is <span style = 'color:red; font-weight:bold'> {t1:.4f} </span> eV", unsafe_allow_html=True)

            
            

