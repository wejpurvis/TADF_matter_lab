import streamlit as st
import pandas as pd


@st.cache_data
def load_data():
    df = pd.read_csv("./Quantum Project/data/TADF_data_DL.txt", sep ="\t", header=None)
    df.columns = ["ID", "SMILES","LUMO", "HOMO", "E(S1)", "E(T1)"]
    filt_data = df.drop(columns = ["ID", "LUMO", "HOMO"])
    return filt_data

def show_info_page():
    st.write("# Background")

    st.write("Thermally activated delayed fluorescent (TADF) materials have attracted significant attention in organic light emitting diodes (OLEDs) due to their ability to harvest both singlet and triplet excitons for fluorescence through intersystem crossing (ISC) and reverse intersystem crossing (RISC). The TADF mechanism relies on a characteristically small singlet-triplet energy gap (~ 0.1 eV) that enables triplet excitons to undergo RISC to the singlet state resulting in a theoretical maximum internal quantum efficiency of 100%.")
    st.write("Development of novel TADF molecules employs molecular designs aimed at minimizing the singlet-triplet energy gap to enable RISC. This workflow usually involves modelling the candidate molecules using computational chemistry methods, synthesizing promising molecules, and validating the photophysical properties of the molecules experimentally. This process is both time-intensive and costly. It is therefore beneficial to predict whether a specific molecule will be TADF active before it is prepared in the lab.")
    st.write("""Recent efforts in accelerated materials design have employed machine learning and deep learning to predict molecular properties directly from various input types, with the benefit of being computationally affordable for high-throughput screening when compared to conventional methods based on quantum chemistry calculations. In this project, a neural network is trained to predict the S1 and T1 energies of TADF molecules.
              The work is inspired by previous research carried out by Kim et al. DOI: [10.1002/bkcs.12516](https://doi.org/10.1002/bkcs.12516)
             
             """)

    st.write("# Dataset")
    st.write("The dataset used to train the model was built from the PubChem database by extracting molecules that contained at least one aromatic ring and up to 50 heavy atoms (boron, carbon, nitrogen, oxygen, phosphorus, and sulfur). This resulted in a set of 42 008 unique molecules. The authors then performed structural optimizations via density functional theory (DFT) calculations with the B3LYP functional and 6-31 g(d) basis set implemented in the Gaussian 16 package. The initial geometries for the 42k molecules were obtained from universal force field (UFF) optimizations using the RDKit python library. The S1 and T1 energies were then obtained using time-dependent DFT (TDDFT) with the above basis set and functional.")
    st.write("The first five rows of the dataset as well a histogram for each energy is shown below:")

    data = load_data()
    st.write(data.head())

    st.image("./Quantum Project/energy_histograms.png")


    st.write("# Model")
    st.write("""The model employed is a multilayer perceptron (MLP). MLPs are feed-forward neural networks consisting of multiple perceptron layers to make up an input layer, a number of hidden layers, and an output layer. Before training, the data was randomly split into 8:1:1 for training, validation, and test sets. The hyperparameters for the model are listed below:

             """)
    st.markdown("- Dimension of hidden layers: 1869 \n- Number of linear layers: 7 \n- Dropout: 0.0 \n-Number of linear layers of predictor: 1 \n- Learning rate: 10<sup>-3.5075</sup>", unsafe_allow_html=True)
    st.write("The activation function used for each layer is ReLU and the model was trained in batches (batch size = 32) over 100 epochs. The loss function that was minimised was smooth L1 and the model which minimised the validation set error was chosen. The training and validation loss is shown below:")
    st.image("./Quantum Project/validation_training_loss_comparison.png")
    results = """
                Loss on test set:  0.0685
                MSE on test set:  0.1415
                MAE on test set:  0.2706
                R2 score on test set:  0.6208
    """
    st.write("After training the model it was evaluated using the test set. Here are the results:")
    st.code(results, language="python")

    st.write("### Discussion")
    st.write("This project aimed to combine my knowledge of TADF molecules obtained from my undergraduate research project within the Zysman-Coleman group at University of St Andrews with what I have learned during my time at the Aspuru-Guzik group at the University of Toronto. The original goal was to deploy multiple different models using different molecular representations to determine which combination of model and representation showed the highest accuracy, although at the time of writing only the MLP model had completed training (CNN using one-hot-encoded SMILES as input currently in training).")
    st.write("The code behind this model can be found on GitHub [here](https://github.com/wejpurvis/TADF_matter_lab). Unfortunately, the mean average error obtained on the test set (0.27 eV) is greater than the maximum singlet-triplet energy gap (0.1 eV) required for the TADF mechanism to occur meaning that this particular model can not be used for efficient pre-screening of TADF candidate molecules.")
    st.write("An extension of this work would involve using different representations for the molecules as input to the MLP model, such as one-hot encoded vectors, use a CNN with SMILES or SELFIEs as input, and use Bayesian optimization to further fine-tune the hyperparameters of the MLP.")
    st.write("I would like to thank Sean Park for his help throughout this project.")