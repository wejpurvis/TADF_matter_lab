import streamlit as st

def show_info_page():
    st.write("# Background")

    st.write("Thermally activated delayed fluorescent (TADF) materials have attracted significant attention in organic light emitting diodes (OLEDs) due to their ability to harvest both singlet and triplet excitons for fluorescence through intersystem crossing (ISC) and reverse intersystem crossing (RISC). The TADF mechanism relies on a characteristically small singlet-triplet energy gap (~ 0.1 eV) that enables triplet excitons to undergo RISC to the singlet state resulting in a theoretical maximum internal quantum efficiency of 100%.")
    st.write("Development of novel TADF molecules employs molecular designs aimed at minimizing the singlet-triplet energy gap to enable RISC. This workflow usually involves modelling the candidate molecules using computational chemistry methods, synthesizing promising molecules, and validating the photophysical properties of the molecules experimentally. This process is both time-intensive and costly. It is therefore beneficial to predict whether a specific molecule will be TADF active before it is prepared in the lab.")
    st.write("""Recent efforts in accelerated materials design have employed machine learning and deep learning to predict molecular properties directly from various input types, with the benefit of being computationally affordable for high-throughput screening when compared to conventional methods based on quantum chemistry calculations. In this project, a neural network is trained to predict the S1 and T1 energies of TADF molecules.
              The work is inspired by previous research carried out by Kim et al. DOI: [10.1002/bkcs.12516](https://doi.org/10.1002/bkcs.12516)
             
             """)

    st.write("# Dataset")
    st.write("[Work in progress]")

    st.write("# Model")
    st.write("[Work in progress]")

    st.write("# Discussion")
    st.write("[Work in progress]")