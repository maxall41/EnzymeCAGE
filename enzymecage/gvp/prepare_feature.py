import os

from tqdm import tqdm
import mlcrate as mlc
from Bio.PDB import PDBParser
import torch
torch.set_num_threads(1)

from data import ProteinGraphDataset



three_to_one = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E', 'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 
                'ILE': 'I', 'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N', 'PRO': 'P', 'GLN': 'Q', 
                'ARG': 'R', 'SER': 'S', 'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}


def tranverse_folder(folder):
    all_files = []
    for root, dirs, files in os.walk(folder):
        for file in files:
            all_files.append(os.path.join(root, file))
    return all_files


def get_clean_res_list(res_list, verbose=False, ensure_ca_exist=False, bfactor_cutoff=None):
    clean_res_list = []
    for res in res_list:
        hetero, resid, insertion = res.full_id[-1]
        if hetero == ' ':
            if res.resname not in three_to_one:
                if verbose:
                    print(res, "has non-standard resname")
                continue
            if (not ensure_ca_exist) or ('CA' in res):
                if bfactor_cutoff is not None:
                    ca_bfactor = float(res['CA'].bfactor)
                    if ca_bfactor < bfactor_cutoff:
                        continue
                clean_res_list.append(res)
        else:
            if verbose:
                print(res, res.full_id, "is hetero")
    return clean_res_list


def get_protein_feature(res_list):
    # protein feature extraction code from https://github.com/drorlab/gvp-pytorch
    # ensure all res contains N, CA, C and O
    res_list = [res for res in res_list if (('N' in res) and ('CA' in res) and ('C' in res) and ('O' in res))]
    # construct the input for ProteinGraphDataset
    # which requires name, seq, and a list of shape N * 4 * 3
    structure = {}
    structure['name'] = "placeholder"
    structure['seq'] = "".join([three_to_one.get(res.resname) for res in res_list])
    coords = []
    for res in res_list:
        res_coords = []
        for atom in [res['N'], res['CA'], res['C'], res['O']]:
            res_coords.append(list(atom.coord))
        coords.append(res_coords)
    structure['coords'] = coords
    torch.set_num_threads(1)        # this reduce the overhead, and speed up the process for me.
    dataset = ProteinGraphDataset([structure])
    protein = dataset[0]
    x = (protein.x, protein.seq, protein.node_s, protein.node_v, protein.edge_index, protein.edge_s, protein.edge_v)
    return x


def batch_run(x):
    protein_dict = {}
    pdb, proteinFile, toFile = x
    parser = PDBParser(QUIET=True)
    s = parser.get_structure(pdb, proteinFile)
    res_list = get_clean_res_list(s.get_residues(), verbose=False, ensure_ca_exist=True)
    protein_dict[pdb] = get_protein_feature(res_list)
    torch.save(protein_dict, toFile)


def main():
    # pdb_folder = '/mnt/nas/ai-algorithm-data/liuyong/dataset/SynBio/enzyme-reaction-pairs/from_zt/3D-structures/active_site_new_no_ion/pdb_8A'
    # protein_embedding_folder = '/mnt/nas/ai-algorithm-data/liuyong/code/SynBio/enzyme-rxn-prediction/dataset/gvp/gvp_protein_embedding_pocket'
    # save_path = '/mnt/nas/ai-algorithm-data/liuyong/code/SynBio/enzyme-rxn-prediction/dataset/gvp/processed/protein_pocket_8A_new.pt'
    pdb_folder = '/mnt/nas/ai-algorithm-data/liuyong/dataset/SynBio/enzyme-reaction-pairs/from_zt/3D-structures/alphafold_structures'
    protein_embedding_folder = '/mnt/nas/ai-algorithm-data/liuyong/code/SynBio/enzyme-rxn-prediction/dataset/gvp/gvp_protein_embedding_full'
    save_path = '/mnt/nas/ai-algorithm-data/liuyong/code/SynBio/enzyme-rxn-prediction/dataset/gvp/processed/protein_full.pt'
    os.makedirs(protein_embedding_folder, exist_ok=True)
    pdbfiles = tranverse_folder(pdb_folder)

    input_ = []
    uniprot_id_list = []
    for filepath in pdbfiles:
        if not filepath.endswith('pdb'):
            continue
        
        uniprot_id = os.path.basename(filepath).replace('.pdb', '')
        uniprot_id_list.append(uniprot_id)
        toFile = f"{protein_embedding_folder}/{uniprot_id}.pt"
        x = (uniprot_id, filepath, toFile)
        input_.append(x)
    
    pool = mlc.SuperPool(64)
    pool.pool.restart()
    _ = pool.map(batch_run,input_)
    pool.exit()

    protein_dict = {}
    for uniprot_id in tqdm(uniprot_id_list):
        protein_dict.update(torch.load(f"{protein_embedding_folder}/{uniprot_id}.pt"))
    torch.save(protein_dict, save_path)


if __name__ == '__main__':
    main()
