from rdkit import Chem, rdBase
from rdkit.Chem import Draw, AllChem
from rdkit import Avalon
from rdkit.Avalon import pyAvalonTools
import streamlit as st

st.title('Fingerprintの可視化')

smiles = st.text_input(label='SMILES', value='')
if smiles:
    mol = Chem.MolFromSmiles(smiles)

    # bitI_avalon = {}
    # fp_avalon = pyAvalonTools.GetAvalonFP(mol, bitI_avalon)

    bitI_morgan = {}
    fp_morgan = AllChem.GetMorganFingerprintAsBitVect(mol, 2, bitInfo=bitI_morgan)

    st.header('morgan')
    st.write('bitの数: ',fp_morgan.GetNumBits(), 'flgの数: ',fp_morgan.GetNumOnBits()) ### 2048 86
    st.write(len(bitI_morgan)) 

    morgan_turples = ((mol, bit, bitI_morgan) for bit in list(bitI_morgan.keys()))
    st.image(Draw.DrawMorganBits(morgan_turples, molsPerRow=4, legends=['bit: '+str(x) for x in list(bitI_morgan.keys())]),use_column_width=True)

    # bitI_rdkit = {}
    # fp_rdkit = AllChem.RDKFingerprint(mol, bitInfo=bitI_rdkit)

    # st.write(fp_rdkit)
    # st.header('rdkit')
    # print('bitの数: ', len(fp_rdkit), 'flgの数: ', len(bitI_rdkit.keys()))
    # rdkit_turples = ((mol, bit, bitI_rdkit) for bit in list(bitI_rdkit.keys()))
    # st.image(Draw.DrawRDKitBits(rdkit_turples, molsPerRow=5, legends=['bit: '+str(x) for x in list(bitI_rdkit.keys())]))