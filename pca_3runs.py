import os
import numpy as np
import scipy.stats as stats
import pandas as pd
import multiprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.decomposition import PCA
from roi_rsa import merge_n_smooth_mask, transform_mask_MNI_to_T1, applyMask
from pca import neural_compression

"""Reproducing key results from Ahlheim et al., 2018
    - LOC shows task-specific functional dimensionality

We do not see expected results using Mack et al., 2020 routine.
One difference in implementation is that Ahlheim 2018 averaged 
the last 3 runs and then did cv-PCA.

Here, (before doing cv-PCA), we make minimum changes to Mack 2020
implementation to simply perform PCA on the average over last 3 runs.
"""    
    
def apply_PCA(roi, root_path, glm_path, roi_path, sub, task, dataType, conditions, smooth_beta):
    """
    Apply PCA onto an embedding matrix of (n_voxels, n_trials) of a given (roi, sub, task) averaged
    over runs, where the columns are beta weights of a given roi of a given trial.
    
    ROI extraction uses `applyMask()` from `roi_rsa.py`
    
    return:
    -------
        k: number of principle components that explains 90% variance.
    """
    # to collect run's embedding matrix, and 
    # return average.
    averaged_embedding_matrix = []
    
    for run in runs:
        # embedding mtx of a run (rp, roi_size)
        beta_weights_masked = []
        for rp in range(1, num_repetitions_per_run+1):
            # NOTE: not necessary as order does not matter.
            # we keep this because it is consistent with `roi_rsa.py`
            conditions_of_the_same_rp = conditions[rp-1::num_repetitions_per_run]
            assert len(conditions_of_the_same_rp) == 8
        
            for condition in conditions_of_the_same_rp:
                # given beta weights from a task & run & condition 
                # apply the transformed mask
                roi, maskROI, fmri_masked = applyMask(
                    roi=roi, 
                    root_path=root_path,
                    glm_path=glm_path,
                    roi_path=roi_path,
                    sub=sub, 
                    task=task, 
                    run=run, 
                    dataType=dataType, 
                    condition=condition,
                    smooth_beta=smooth_beta
                )
                beta_weights_masked.append(fmri_masked)
        
        # (n_voxels, n_trials), where n_trials = 32 in a run.
        beta_weights_masked = np.array(beta_weights_masked).T
        print(f'run{run}, beta_weights_masked.shape', beta_weights_masked.shape)
        
        # collect run's embedding matrix, and 
        averaged_embedding_matrix.append(beta_weights_masked)
    
    # PCA
    # (n_runs, n_voxels, n_trials) -> (n_voxels, n_trials)
    averaged_embedding_matrix = np.mean(
        np.array(averaged_embedding_matrix), axis=0
    )
    pca = PCA(n_components=averaged_embedding_matrix.shape[1], random_state=42)
    pca.fit(averaged_embedding_matrix)
    explained_variance_ratio = pca.explained_variance_ratio_
    
    # return the k PCs that explain at least 90% variance
    k = 0
    explained_variance_cumu_ = 0
    while k < averaged_embedding_matrix.shape[1]:
        if explained_variance_cumu_ >= 0.9:
            print(
                f'explained_variance_cumu_={explained_variance_cumu_}, k={k}'
            )
            return neural_compression(k=k)
        else:
            explained_variance_cumu_ += explained_variance_ratio[k]
            k += 1
    
    print(
        f'explained_variance_cumu_={explained_variance_cumu_}, k={k}'
    )
    return neural_compression(k=k)
        

def execute(roi, subs, tasks, num_processes):
    if not os.path.exists(f'Ahlheim_results/{roi}.npy'):
        with multiprocessing.Pool(num_processes) as pool:
            
            if 'HPC' in roi:
                roi_path = 'ROIs/HPC'
                    
            elif 'vmPFC' in roi:
                roi_path = 'ROIs/vmPFC'
                
            else:
                # V1,2,3,1-4, LOC, LHLOC,RHLOC
                roi_path = 'ROIs/ProbAtlas_v4/subj_vol_all'
            
            # Do it once for one ROI mask (MNI)
            # Will skip if already exists
            merge_n_smooth_mask(roi=roi, roi_path=roi_path, smooth_mask=False)
            
            # compute & collect compression
            type2metric = defaultdict(list)
            for task in tasks:
                for sub in subs:
                    # done once for each sub
                    transform_mask_MNI_to_T1(sub=sub, roi=roi, roi_path=roi_path, root_path=root_path)
                    
                    # per (sub, task, run) compression
                    res_obj = pool.apply_async(
                        apply_PCA, 
                        args=[
                            roi, root_path, glm_path, roi_path, 
                            sub, task, 
                            dataType, conditions, smooth_beta
                        ]
                    )
                                                            
                    if int(sub) % 2 == 0:
                        if task == 2:
                            problem_type = 1
                        elif task == 3:
                            problem_type = 2
                        else:
                            problem_type = 6
                            
                    # odd sub: Type1 is task3
                    else:
                        if task == 2:
                            problem_type = 2
                        elif task == 3:
                            problem_type = 1
                        else:
                            problem_type = 6
                    
                    # Notice res_obj.get() = compression
                    # To enable multiproc, we extract the actual
                    # compression score when plotting later.
                    type2metric[problem_type].append(res_obj)
            
            pool.close()
            pool.join()
            
    num_types = len(type2metric.keys())
    problem_types = sorted(list(type2metric.keys()))
    print(f'num_types={num_types}, {problem_types}')

    for z in range(num_types):
        problem_type = problem_types[z]
        # here we extract a list of res_obj and 
        # extract the actual compression scores.
        list_of_res_obj = type2metric[problem_type]
        # `metrics` is all scores over subs for one (problem_type)
        metrics = [res_obj.get() for res_obj in list_of_res_obj]
        mean = np.mean(metrics)
        std = np.std(metrics)
        print(f'Type=[{problem_type}], roi=[{roi}], mean=[{mean:.3f}], std=[{std:.3f}]')
        

if __name__ == '__main__':    
    root_path = '/home/ken/projects/brain_data'
    glm_path = 'glm_trial-estimate'
    roi = 'RHLOC'
    num_subs = 23
    num_types = 3
    dataType = 'beta'
    num_conditions = 64
    subs = [f'{i:02d}' for i in range(2, num_subs+2)]
    conditions = [f'{i:04d}' for i in range(1, num_conditions+1)]
    tasks = [1, 2, 3]
    runs = [2, 3, 4]  # will average over.
    num_runs = len(runs)
    num_repetitions_per_run = 4
    smooth_beta = 2
    num_processes = 70
    if dataType == 'beta':
        # ignore `_rp*_fb` conditions, the remaining are `_rp*` conditions.
        conditions = [f'{i:04d}' for i in range(1, num_conditions, 2)]
    
    execute(
        roi=roi, 
        subs=subs, 
        tasks=tasks, 
        num_processes=num_processes
    )
    
    