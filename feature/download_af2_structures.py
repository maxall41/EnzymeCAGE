import os
import argparse
import requests
from functools import partial
from multiprocessing import Pool

import pandas as pd
from tqdm import tqdm

CNT_FAILED = 0
failed_uids = []


def download_alphafold_structure(uid, save_dir):
    output_file = os.path.join(save_dir, f"{uid}.cif")
    # output_file = f"/home/liuy/data/SynBio/Terpene/structures/af2/{uid}.pdb"

    if os.path.exists(output_file):
        return

    try:
        url = f"https://alphafold.ebi.ac.uk/files/AF-{uid}-F1-model_v4.cif"
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_file, 'wb') as file:
                file.write(response.content)
            # print(f"Structure for {uniprot_id} downloaded successfully.")
            return True
        else:
            print(f"No structure found for {uid}.")
            global CNT_FAILED, failed_uids
            CNT_FAILED += 1
            failed_uids.append(uid)
            return False
    except:
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--uid_col', type=str, default='uniprotID', help='The column name of Uniprot ID in the data file')
    parser.add_argument('--n_process', type=int, default=10)
    args = parser.parse_args()
    
    assert os.path.exists(args.data_path), f"Data file {args.data_path} does not exist."
    save_dir = os.path.join(os.path.dirname(args.data_path), 'af2_structures')
    os.makedirs(save_dir, exist_ok=True)
    
    df_data = pd.read_csv(args.data_path)
    if args.uid_col not in df_data.columns:
        raise ValueError(f"Column {args.uid_col} does not exist in the data file.")
    
    uniprot_ids = set(df_data[args.uid_col])
    uids_to_download = [each for each in uniprot_ids if isinstance(each, str)]
    running_func = partial(download_alphafold_structure, save_dir=save_dir)

    with Pool(args.n_process) as pool:
        for _ in tqdm(pool.imap(running_func, uids_to_download), total=len(uids_to_download)):
            pass

    n_success = len(uids_to_download) - CNT_FAILED
    print(f"Downloaded {n_success} structures successfully. Failed: {CNT_FAILED}")
    
    if failed_uids:
        df_failed = pd.DataFrame({args.uid_col: failed_uids})
        save_path = os.path.join(save_dir, 'failed_uids.csv')
        df_failed.to_csv(save_path, index=False)
        print(f"Failed proteins: {save_path}")


if __name__ == "__main__":
    main()
