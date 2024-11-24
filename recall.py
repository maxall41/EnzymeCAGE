import os
import sys
import pickle as pkl
from collections import defaultdict
from typing import *
from copy import deepcopy

import numpy as np
import pandas as pd
from tqdm import tqdm

from utils import fast_cosine_matrix, calc_rxn_center_fp, remove_nan, check_dir, ROOT_DIR, DATA_DIR


ALL_RXN_FEAT_DICT = {
    'rxnfp': pkl.load(open(f'{DATA_DIR}/feature/rxnfp/rxn2fp.pkl', 'rb')),
    'drfp': pkl.load(open(f'{DATA_DIR}/feature/drfp/rxn2fp.pkl', 'rb')),
}

ALL_ENZ_FEAT_DICT = {
    'esm2': pkl.load(open(f'{DATA_DIR}/feature/esm2_t33_650M_UR50D/protein_level/seq2feature.pkl', 'rb')),
    # 'gearnet': pkl.load(open('/mnt/nas/ai-algorithm-data/liuyong/dataset/SynBio/enzyme-reaction-pairs/from_zt/features/GearNet/uniprot2feature.pkl', 'rb')),
}

# All real reaction-enzyme pair data
df_enz_rxn_all = pd.read_csv(f'{DATA_DIR}/overall/rxn2seq_clean_v2_no_ion.csv')
rxns_with_tmpl = set(df_enz_rxn_all[df_enz_rxn_all['template_v3'].notnull()]['CANO_RXN_SMILES'])

# Load similarity matrix first
# drfp_simi_matrix_path = f'{DATA_DIR}/others/rxn_similarity/drfp_based.npz'
# morgan_simi_matrix_path = f'{DATA_DIR}/others/rxn_similarity/molecular_morganfp_based_all.npz'
# print('Loading drfp based reaction similarity matrix...')
# drfp_simi_matrix = np.load(drfp_simi_matrix_path, allow_pickle=True)
# all_rxns = drfp_simi_matrix['y'].tolist()
# drfp_simi_matrix = drfp_simi_matrix['x']

morgan_simi_matrix_path = f'{DATA_DIR}/others/rxn_similarity/molecular_morganfp_based_all.npz'
print('Loading morgan molecular fp based reaction similarity matrix...')
MORGAN_SIMI_MATRIX = np.load(morgan_simi_matrix_path, allow_pickle=True)
ALL_RXNS = MORGAN_SIMI_MATRIX['y'].tolist()
MORGAN_SIMI_MATRIX = MORGAN_SIMI_MATRIX['x']
RXN_TO_ID = {rxn: i for i, rxn in enumerate(ALL_RXNS)}


def init_mapping_info(df_all):
    rxn_to_ec = defaultdict(set)
    uniprot_to_seq = {}
    rxn_to_uniprot = defaultdict(set)
    ec_to_uniprot = defaultdict(set)
    ec_to_rxn = defaultdict(set)
    uniprot_to_cluster = {}
    cluster_to_uniprot = defaultdict(set)
    uniprot_to_ec = defaultdict(set)
    uniprot_to_rxns = defaultdict(set)

    rxn_list = df_all['CANO_RXN_SMILES'].values
    ec_number_list = df_all['ec number'].values
    uniprotID_list = df_all['uniprotID'].values
    sequence_list = df_all['sequence'].values
    cluster_list = df_all['active_site_cluster_id'].values
    for i in tqdm(range(len(df_all)), desc='Initializing mapping info'):
        rxn = rxn_list[i]
        ec_number = ec_number_list[i]
        uniprotID = uniprotID_list[i]
        sequence = sequence_list[i]
        cluster = cluster_list[i]

        #if rxn not in test_rxns:
        rxn_to_ec[rxn].add(ec_number)
        uniprot_to_seq[uniprotID] = sequence
        rxn_to_uniprot[rxn].add(uniprotID)
        
        if isinstance(ec_number, str):
            ec_to_uniprot[ec_number].add(uniprotID)
            ec_to_rxn[ec_number].add(rxn)
            
        if isinstance(uniprotID, str):
            uniprot_to_rxns[uniprotID].add(rxn)
            uniprot_to_ec[uniprotID].add(ec_number)
        
        uniprot_to_cluster[uniprotID] = cluster
        if not pd.isna(cluster):
            cluster_to_uniprot[cluster].add(uniprotID)
    return rxn_to_ec, uniprot_to_seq, rxn_to_uniprot, ec_to_uniprot, uniprot_to_cluster, cluster_to_uniprot, uniprot_to_ec, ec_to_rxn, uniprot_to_rxns

RXN_TO_EC, UNIPROT_TO_SEQ, RXN_TO_UNIPROT, EC_TO_UNIPROT, UNIPROT_TO_CLUSTER, CLUSTER_TO_UNIPROT, UNIPROT_TO_EC, EC_TO_RXNS, UNIPROT_TO_RXNS = init_mapping_info(df_enz_rxn_all) 





def find_most_similar_rxns(rxn: str, cand_rxns: List[str], topk=10, method='rxnfp', use_template: bool=False, template_rxn_map: dict=None, tmpl_features: tuple=None):
    """Find the most similar reactions for a given reaction

    Args:
        rxn (str): Input reaction SMILES. a.b>>c format.
        cand_rxns (List[str]): All candidate reactions. The same format as rxn.
        topk (int, optional): Number of result reactions to return. Defaults to 10.
        method (str, optional): How to match similar reactions. Defaults to 'rxnfp'. ['rxnfp', 'drfp', '']
        use_template (bool, optional): Whether to use template and template fp to match similar reactions. Defaults to False.
        template_rxn_map (dict, optional): A dict mapping template to reactions. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        _type_: _description_
    """
    cand_rxns = {rxn for rxn in cand_rxns if isinstance(rxn, str)}

    if use_template:
        assert template_rxn_map is not None and tmpl_features is not None

    if method not in ALL_RXN_FEAT_DICT:
        raise ValueError(f'Invalid method: {method}')
    
    if use_template:
        try:
            rxn_center = get_template(rxn)
        except Exception as e:
            print(e)
            rxn_center = None
            use_template = False
        if not isinstance(rxn_center, str):
            # Cannot get template from the input reaction, swith to no template mode
            use_template = False
            # return [], []

    if not use_template:
        # Only use reaction fingerprint to match similar reactions
        res_rxns, simi_scores = sort_similar_rxns(rxn, cand_rxns, method)
    else:
        # Step 1: Use template fingerprint to match the same type of reactions
        # Step 2: Use reaction fingerprint to sort the matched reactions acording to fp similarity
        
        # Step 1.1: Get the template and template fp of the input reaction
        reac_fp, prod_fp = calc_rxn_center_fp(rxn_center)

        # Step 1.2: Get template fps of all templates in template_rxn_map
        all_tmpls = list(template_rxn_map.keys())
        all_reac_fps, all_prod_fps = tmpl_features
        # Step 1.3: Get the most similar templates
        prod_sim_matrix = fast_cosine_matrix(prod_fp, all_prod_fps)
        reac_sim_matrix = fast_cosine_matrix(reac_fp, all_reac_fps)
        center_sim_matrix = (prod_sim_matrix + reac_sim_matrix) / 2
        template_result = list(zip(all_tmpls, center_sim_matrix))
        template_result = sorted(template_result, key=lambda x: x[1], reverse=True)

        # Step 1.4: Get the most similar reactions for each template
        res_rxns = []
        for each in template_result:
            template = each[0]
            similarity = each[1]
            related_rxns = template_rxn_map.get(template)
            if similarity == 1:
                res_rxns.extend(related_rxns)
            else:
                if len(res_rxns) >= topk:
                    break
                else:
                    res_rxns.extend(related_rxns)
        # Step 2: Use reaction fingerprint to sort the matched reactions acording to fp similarity
        res_rxns, simi_scores = sort_similar_rxns(rxn, res_rxns, method)
    
    res_rxns = res_rxns[:topk]
    simi_scores = simi_scores[:topk]

    return res_rxns, simi_scores


def enzyme_recall(df_test):
    """Recall candidate enzymes for each reaction in test set.

    Args:
        df_test (pd.DataFrame): test dataset.
        split_by (str, optional): method for train/valid/test split. Defaults to 'reaction'.
    """
    test_rxns = set(df_test[df_test['Label'] == 1]['CANO_RXN_SMILES'].drop_duplicates())
    test_rxns = {rxn for rxn in test_rxns if isinstance(rxn, str)}

    # test_rxns = test_rxns & rxns_with_tmpl
    print(f"Number of test reactions: {len(test_rxns)}")

    if split_by == 'reaction' or split_by == 'both':
        all_cand_rxns = set(df_enz_rxn_all['CANO_RXN_SMILES']) - test_rxns
        print(f"Number of all candidate reactions: {len(all_cand_rxns)}")

        all_rxns = df_enz_rxn_all['CANO_RXN_SMILES'].tolist()
        templates = df_enz_rxn_all['template_v3'].tolist()
        tmpl_to_rxns = defaultdict(set)
        for rxn, tmpl in zip(all_rxns, templates):
            if not isinstance(tmpl, str):
                continue
            if rxn not in test_rxns:
                tmpl_to_rxns[tmpl].add(rxn)
        
        all_tmpls = list(tmpl_to_rxns.keys())
        print(f"Number of all templates: {len(all_tmpls)}")
        all_reac_fps = []
        all_prod_fps = []
        for tmpl in tqdm(all_tmpls, desc='Calculate template fingerprints'):
            reac_fp, prod_fp = calc_rxn_center_fp(tmpl)
            all_reac_fps.append(reac_fp)
            all_prod_fps.append(prod_fp)
        all_reac_fps = np.array(all_reac_fps)
        all_prod_fps = np.array(all_prod_fps)
        tmpl_features = (all_reac_fps, all_prod_fps)

        test_rxns = list(test_rxns)
        similar_rxns_map = defaultdict(list)
        # similar_rxns_map_tmp = defaultdict(list)

        simi_rxn_map_cache_path = f'{ROOT_DIR}/dataset/cache/similar_rxn_map/{data_version}/split_by_{split_by}/{rxn_feat_type}/similar_rxn_map.pkl'
        check_dir(simi_rxn_map_cache_path)
        if os.path.exists(simi_rxn_map_cache_path):
            print(f'Find similar reaction map of {data_version} calculated by {rxn_feat_type}, cache loaded.')
            similar_rxns_map = pkl.load(open(simi_rxn_map_cache_path, 'rb'))
        else:
            for i, rxn in enumerate(tqdm(test_rxns, desc='Find similar reactions')):
                similar_rxns, simi_scores = find_most_similar_rxns(rxn, all_cand_rxns, topk=10, method=rxn_feat_type, use_template=False)
                # similar_rxns, _ = find_most_similar_rxns(rxn, all_cand_rxns, topk=10, method=rxn_feat_type, use_template=True, template_rxn_map=tmpl_to_rxns, tmpl_features=tmpl_features)
                similar_rxns_map[rxn] = similar_rxns

            with open(simi_rxn_map_cache_path, 'wb') as f:
                pkl.dump(similar_rxns_map, f)

        #     similar_rxns_map_tmp[rxn] = list(zip(similar_rxns, simi_scores))
        # similar_rxns_map_path = '/mnt/nas/ai-algorithm-data/liuyong/code/SynBio/enzyme-rxn-prediction/dataset/tmp/Gearnet_eval/split_by_reaction/drfp/similar_rxn_map.pkl'
        # with open(similar_rxns_map_path, 'wb') as f:
        #     pkl.dump(similar_rxns_map_tmp, f)

        all_ec_set = set(df_enz_rxn_all['ec number'])
        all_ec_set.remove(np.nan)
        cand_enz_dict = {}
        for rxn_to_infer in tqdm(test_rxns, desc='Recalling enzymes'):
            topk_most_similar_rxns = similar_rxns_map[rxn_to_infer]
            if not topk_most_similar_rxns:
                cand_enz_dict[rxn_to_infer] = set()
                continue
                
            cand_ec_set = set()
            for similar_rxn in topk_most_similar_rxns:
                cand_ec_set.update(RXN_TO_EC.get(similar_rxn))
            new_cand_ec_set = {each for each in cand_ec_set if isinstance(each, str)}
            
            enzyme_set = set()
            for ec in new_cand_ec_set:
                enzyme_set.update(EC_TO_UNIPROT.get(ec))
            
            all_clusters = set()
            for uniprot in enzyme_set:
                all_clusters.add(UNIPROT_TO_CLUSTER.get(uniprot))
            all_clusters = remove_nan(all_clusters)

            enzyme_set = set()
            for cluster in all_clusters:
                enzyme_set.update(CLUSTER_TO_UNIPROT.get(cluster))

            cand_enz_dict[rxn_to_infer] = enzyme_set
    
    n_candidates_mean = np.mean([len(v) for v in cand_enz_dict.values()])
    print(f"Mean number of candidates: {round(n_candidates_mean, 4)}")
    
    return cand_enz_dict, test_rxns


def calculate_recall(test_rxns: list, cand_enz_dict: dict):
    """Given candidate enzymes for each reaction, calculate recall. 
    If there is more than one true enzyme in the candidate enzyme set, we consider it as a hit.

    Args:
        test_rxns (list): list of test reactions.
        cand_enz_dict (dict): candidate enzymes for each reaction.

    Returns:
        float: recall
    """
    recall_list = []
    precise_recall_list = []
    for rxn in tqdm(test_rxns):
        cand_enz_set = cand_enz_dict.get(rxn)
        true_enzymes = RXN_TO_UNIPROT.get(rxn)
        if not cand_enz_set or not true_enzymes:
            recall_list.append(0)
            precise_recall_list.append(0)
            continue
        
        if len(cand_enz_set & true_enzymes) > 0:
            recall_list.append(1)
            precise_recall_list.append(len(cand_enz_set & true_enzymes) / len(true_enzymes))
        else:
            recall_list.append(0)
            precise_recall_list.append(0)
    
    recall = sum(recall_list) / len(recall_list)
    precise_recall = np.mean(precise_recall_list)
    return round(recall, 4), round(precise_recall, 4)


def eval_top_rank_result(df_test_inference: pd.DataFrame, test_rxns: list):
    """Evaluate the ranking result. 

    Args:
        df_test_inference (pd.DataFrame): testing data with prediction results.
    """
    rxn_col = 'reaction' if 'reaction' in df_test_inference.columns else 'CANO_RXN_SMILES'
    enz_col = 'enzyme' if 'enzyme' in df_test_inference.columns else 'uniprotID'

    inference_result = {}
    for rxn, df in tqdm(df_test_inference.groupby(rxn_col), desc='Reformat inference result'):
        enzymes = df[enz_col].values
        preds = df['pred'].values
        result = list(zip(enzymes, preds))
        result = sorted(result, key=lambda x: x[1], reverse=True)
        inference_result[rxn] = result
    
    best_rank_list = []
    for rxn, enzymes_ranked in tqdm(inference_result.items(), desc='Calculate best rank'):
        enzymes_ranked = list(map(lambda x: x[0], enzymes_ranked))
        true_enzymes = list(RXN_TO_UNIPROT.get(rxn))
        hit_ranks = []
        for enz in true_enzymes:
            if enz in enzymes_ranked:
                rank = enzymes_ranked.index(enz) + 1
                hit_ranks.append(rank)
            else:
                hit_ranks.append(-1)
    
        if max(hit_ranks) == -1:
            best_rank_list.append(-1)
        else:
            best_rank_list.append(min(filter(lambda x: x>0, hit_ranks)))
    
    eval_result = {}
    topk_list = [1, 3, 5, 10]
    for topk in topk_list:
        success_list = list(filter(lambda x: 0<x<=topk, best_rank_list))
        success_rate = len(success_list) / len(test_rxns)
        print(f"=========== Top-{topk} success rate: {round(success_rate, 4)} ===========")
        eval_result[f'top{topk}'] = success_rate
    
    return eval_result


def standardize_order(test_rxns, cand_enz_dict: dict):
    # Make sure to reproduce the same result
    test_rxns = sorted(list(test_rxns))
    for rxn, cand_enz in cand_enz_dict.items():
        cand_enz_dict[rxn] = sorted(list(cand_enz))
    return test_rxns, cand_enz_dict


def get_rxn_morgan_simi(r1, r2):
    idx1, idx2 = RXN_TO_ID[r1], RXN_TO_ID[r2]
    return MORGAN_SIMI_MATRIX[idx1][idx2]


def get_cached_similar_rxn(rxn, cand_rxns, topk=10):
    simi_list = []
    for cand_rxn in cand_rxns:
        simi = get_rxn_morgan_simi(rxn, cand_rxn)
        simi_list.append(simi)

    data = list(zip(cand_rxns, simi_list))
    data = sorted(data, key=lambda x: x[1], reverse=True)
    similar_rxns = list(map(lambda x: x[0], data))[:topk]
    simi_list = list(map(lambda x: x[1], data))[:topk]
    
    return similar_rxns, simi_list


def recalculate_similar_rxn():
    pass


def compute_correlation_score(target_rxn, enzyme, exclude_rxns=None):
    related_rxns = deepcopy(UNIPROT_TO_RXNS[enzyme])
    if exclude_rxns:
        exclude_rxns = set(exclude_rxns)
        related_rxns -= exclude_rxns
        
    if target_rxn in related_rxns:
        related_rxns.remove(target_rxn)
    related_rxns = list(related_rxns)
    
    max_simi = 0
    max_simi_rxn = None
    for r in related_rxns:
        similarity = get_rxn_morgan_simi(r, target_rxn) 
        if similarity > max_simi:
            max_simi = similarity
            max_simi_rxn = r
    
    return max_simi, max_simi_rxn
    

def filter_enzs_by_similarity(rxn, enzyme_set, topk=200, threshold=0.4, exclude_rxns=None):
    """
    exclude_rxns一般设置为测试集，也就是说enzyme_set与exclude_rxns之间的关系被去掉了，避免数据泄露
    """
    if not exclude_rxns:
        exclude_rxns = set([rxn])
    enzyme_list = list(enzyme_set)
    simi_list = []
    for enz in enzyme_list:
        max_simi, _ = compute_correlation_score(rxn, enz, exclude_rxns)
        simi_list.append(max_simi)
    info_list = list(zip(enzyme_list, simi_list))
    info_list = [each for each in info_list if each[1] >= threshold]
    info_list = sorted(info_list, key=lambda x: x[1], reverse=True)
    final_enzyme_list = list(map(lambda x: x[0], info_list))[:topk]
    
    return final_enzyme_list


def run_recall(df_test):
    df_data_zt = pd.read_csv('/home/liuy/data/SynBio/enzyme-reaction-pairs/overall/rxn2seq_clean_v2_no_ion.csv')
    
    rxn_col = 'CANO_RXN_SMILES' if 'CANO_RXN_SMILES' in df_test.columns else 'reaction'
    test_rxns = set(df_test[rxn_col])
    test_rxns = {rxn for rxn in test_rxns if isinstance(rxn, str)}
    # all_rxns = deepcopy(ALL_RXNS)
    # all_rxns = set(all_rxns) if isinstance(all_rxns, list) else all_rxns
    all_cand_rxns = set(df_data_zt['CANO_RXN_SMILES']) - test_rxns
    
    cnt_recall = 0
    n_enz_list = []
    cand_enz_dict = {}
    for rxn in tqdm(test_rxns):
        true_enzs = RXN_TO_UNIPROT.get(rxn)
        # all_cand_rxns = all_rxns - set([rxn])
        similar_rxns, _ = get_cached_similar_rxn(rxn, all_cand_rxns, topk=10)

        related_enzs = set()
        related_pocket_cluster = set()
        for simi_rxn in similar_rxns:

            enzs = RXN_TO_UNIPROT.get(simi_rxn, set())
            related_enzs.update(enzs)
            clusters = {UNIPROT_TO_CLUSTER.get(enz) for enz in enzs}
            related_pocket_cluster.update(clusters)

        related_pocket_cluster = {cid for cid in related_pocket_cluster if not pd.isna(cid)}
        for cluster in related_pocket_cluster:
            enzs = CLUSTER_TO_UNIPROT.get(cluster, set())
            related_enzs.update(enzs)
        
        if len(true_enzs & related_enzs) > 0:
            cnt_recall += 1
        
        n_enz_list.append(len(related_enzs))
        # cand_enz_dict[rxn] = set(filter_enzs_by_similarity(rxn, related_enzs, threshold=0.3, exclude_rxns=test_rxns))
        cand_enz_dict[rxn] = related_enzs

    print(f'mean num of candidate enzymes: {np.mean(n_enz_list)}')
    n_mean_enzymes = np.mean([len(each) for each in cand_enz_dict.values()])
    print(f'mean num of filtered enzymes: {n_mean_enzymes}')
    
    recall = calculate_recall(test_rxns, cand_enz_dict)
    print(f'recall: {recall}')
    
    df_list = []
    for rxn, enz_set in tqdm(cand_enz_dict.items()):
        true_enzs = RXN_TO_UNIPROT.get(rxn)
        enz_list = list(enz_set)
        df_tmp = pd.DataFrame({'enzyme': enz_list})
        df_tmp['reaction'] = rxn
        df_tmp['sequence'] = df_tmp['enzyme'].apply(lambda x: UNIPROT_TO_SEQ.get(x))
        df_tmp['Label'] = df_tmp['enzyme'].apply(lambda x: 1 if x in true_enzs else 0)
        df_tmp['rxn_similarity'] = df_tmp['enzyme'].apply(lambda enz: compute_correlation_score(rxn, enz)[0])
        df_tmp['simi_rxn'] = df_tmp['enzyme'].apply(lambda enz: compute_correlation_score(rxn, enz)[1])
        df_list.append(df_tmp)
    df_infer = pd.concat(df_list)
    
    return df_infer
    

def main():
    # df_test = pd.read_csv('/home/liuy/data/SynBio/enzyme-reaction-pairs/training/v12.3/new-rxns/test.csv')
    df_test = pd.read_csv('/home/liuy/code/SynBio/enzyme-rxn-prediction/case_study/synthesis-route/triglyceride/common_route/reactions.csv')
    
    df_infer = run_recall(df_test)
    
    # save_path = '/home/liuy/code/SynBio/enzyme-rxn-prediction/dataset/recruit_infer/V12.3/new-rxns/infer_optim_large.csv'
    save_path = '/home/liuy/code/SynBio/enzyme-rxn-prediction/case_study/synthesis-route/triglyceride/common_route/recruitment_candidates.csv'
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df_infer.to_csv(save_path, index=False)


def eval_pred_res():
    df_to_infer = pd.read_csv('/mnt/nas/ai-algorithm-data/liuyong/code/SynBio/enzyme-rxn-prediction/checkpoints/V9/split_by_both/esm2_drfp_dnn/eval_infer_result_selenzyme_ESP.csv')

    # df_test = pd.read_csv(f'{ROOT_DIR}/dataset/baseline/V8/split_{split_by}/csv/test.csv')
    df_test = pd.read_csv(f'{ROOT_DIR}/dataset/baseline/V9/final_test/final_test.csv')
    test_rxns = set(df_test[df_test['Label'] == 1]['CANO_RXN_SMILES'].drop_duplicates())
    eval_top_rank_result(df_to_infer, test_rxns)


if __name__ == "__main__":
    main()
    # eval_pred_res()
