import os

from tqdm import tqdm
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem


def generate_rdkit_conformation_v2(smiles, n_repeat=50):
    try:
        mol = Chem.MolFromSmiles(smiles)
        # mol = Chem.RemoveAllHs(mol)
        # mol = Chem.AddHs(mol)
        ps = AllChem.ETKDGv2()
        # rid = AllChem.EmbedMolecule(mol, ps)
        for repeat in range(n_repeat):
            rid = AllChem.EmbedMolecule(mol, ps)
            if rid == 0:
                break
        if rid == -1:
            # print("rid", pdb, rid)
            ps.useRandomCoords = True
            rid = AllChem.EmbedMolecule(mol, ps)
            if rid == -1:
                mol.Compute2DCoords()
            else:
                AllChem.MMFFOptimizeMolecule(mol, confId=0)
        else:
            AllChem.MMFFOptimizeMolecule(mol, confId=0)
    except Exception as e:
        print(e)
        mol = None
    # mol = Chem.RemoveAllHs(mol)
    return mol


def main():
    data_path = '/home/liuy/data/SynBio/Cooperation/jingke/data_to_predict.csv'
    save_dir = '/home/liuy/data/SynBio/Cooperation/jingke/feature/reaction/molecule_conformation'
    
    df_data = pd.read_csv(data_path)
    rxns = set(df_data['CANO_RXN_SMILES'])
    
    all_smiles_list = []
    for rxn in rxns:
        smiles_list = rxn.split('>>')[0].split(".") + rxn.split('>>')[1].split(".")
        all_smiles_list.extend(smiles_list)
    all_smiles_list = [smi.replace('*', 'C') for smi in all_smiles_list]
    all_mols = list(set(all_smiles_list))
    
    df_all_mols = pd.DataFrame({'SMILES': all_mols, 'ID': range(len(all_mols))})
    mol_index_path = os.path.join(save_dir, 'mol2id.csv')
    df_all_mols.to_csv(mol_index_path, index=False)
    
    failed_idx_list = []
    for smiles, idx in tqdm(df_all_mols[['SMILES', 'ID']].values):
        save_path = os.path.join(save_dir, f'{idx}.sdf')
        conformation = generate_rdkit_conformation_v2(smiles)
        if not conformation:
            failed_idx_list.append(idx)
            continue
        Chem.SDWriter(save_path).write(conformation)
    

if __name__ == '__main__':
    main()