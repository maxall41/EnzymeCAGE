import os
import argparse
from multiprocessing import Pool

from tqdm import tqdm


def read_txt(path):
    with open(path, "r") as f:
        data = [each.strip() for each in f.readlines()]
    return data


def tranverse_folder(folder):
    filepath_list = []
    for root, _, files in os.walk(folder):
        for file in files:
            filepath_list.append(os.path.join(root, file))
    return filepath_list


def main_multiprocess(input_dir, output_dir, postfix, pdb_fasta, pdb_redo_dir, n_blast=5, n_process=10):
    uid_list = []
    all_files = tranverse_folder(input_dir)
    for filename in all_files:
        if '.cif' in filename:
            uid_list.append(filename.split('.')[0])
    
    if len(uid_list) == 0:
        raise ValueError(f"No .cif files found in the input directory: {input_dir}")
    
    assert os.path.exists(pdb_fasta), f"PDB fasta file not found: {pdb_fasta}"
    assert os.path.exists(pdb_redo_dir), f"PDB redo database directory not found: {pdb_redo_dir}"

    command_list = []
    for uid in tqdm(uid_list):
        input_structure_path = f'{input_dir}/{uid}.cif'
        output_path = f'{output_dir}/{uid}_{postfix}'
        command = f'alphafill process {input_structure_path} {output_path} --pdb-fasta={pdb_fasta} --pdb-dir={pdb_redo_dir} --blast-report-limit={n_blast}'
        command_list.append(command)
    
    with Pool(n_process) as p:
        for _ in tqdm(p.imap(os.system, command_list), total=len(command_list)):
            pass


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Input directory containing CIF files')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory for AlphaFill transplantion results')
    parser.add_argument('--postfix', type=str, default='_transplant.cif', help='Postfix for output files')
    parser.add_argument('--pdb_fasta', type=str, default='../../../dataset/PDB-REDO/pdbredo_seqdb.txt')
    parser.add_argument('--pdb_redo_dir', type=str, default='../../../dataset/PDB-REDO/pdb-redo')
    args = parser.parse_args()
    
    main_multiprocess(args.input_dir, args.output_dir, args.postfix, args.pdb_fasta, args.pdb_redo_dir)
