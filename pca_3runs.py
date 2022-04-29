import os
import numpy as np
from scipy import linalg
import scipy.stats as stats
import pandas as pd
import multiprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.decomposition import PCA
from roi_rsa import merge_n_smooth_mask, transform_mask_MNI_to_T1, applyMask

"""Reproducing key results from Ahlheim et al., 2018
    - LOC shows task-specific functional dimensionality
    
Some key differences to the original implementation:
1. There is no cv-PCA at the moment. Instead, we average 
    the last 3 runs matrices and selecting dimensions based 
    on 90% variance explained.
2. Ahlheim 2018 uses GLM from Mack 2016 (run-level), but here 
    our run-level GLM does not include impulse.
3. Ahlheim 2018 does PCA centering by row (voxel) but Mack 2020
    does PCA centering by column. Here we try both.
"""

def neural_compression(k, n):
    """
    k: number of PCs that explains 90% variance
    n: number of conditions within a block (i.e. run)
    """
    return 1 - (k/n) 
    
    
def apply_PCA(roi, root_path, glm_path, roi_path, sub, task, dataType, conditions, smooth_beta, centering_by):
    """
    Apply PCA onto an embedding matrix of (n_voxels, n_conditions) of a given (roi, sub, task) averaged
    over runs, where the columns are beta weights of a given roi of a given condition.
    
    ROI extraction uses `applyMask()` from `roi_rsa.py`
    
    return:
    -------
        k: number of principle components that explains 90% variance.
    """
    # to collect run's embedding matrix, and 
    # return average.
    averaged_embedding_matrix = []
    
    for run in runs:
        # embedding mtx of a run (conditions, roi_size)
        beta_weights_masked = []
        for condition in conditions:
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
        
        # (n_voxels, n_conditions), where n_conditions = 8 in a run.
        beta_weights_masked = np.array(beta_weights_masked).T
        print(f'run{run}, beta_weights_masked.shape', beta_weights_masked.shape)
        
        # collect run's embedding matrix, and 
        averaged_embedding_matrix.append(beta_weights_masked)
    
    # PCA
    # (n_runs, n_voxels, n_conditions) -> (n_voxels, n_conditions)
    averaged_embedding_matrix = np.mean(
        np.array(averaged_embedding_matrix), axis=0
    )
    
    if centering_by == 'row':
        # mean-center (by row)
        # NOTE: sklearn PCA default is by column.
        row_mean = np.mean(averaged_embedding_matrix, axis=1).reshape(-1, 1)
        averaged_embedding_matrix -= row_mean
        
        # SVD
        U, S, Vt = linalg.svd(averaged_embedding_matrix, full_matrices=False)
        explained_variance_ = (S ** 2) / (averaged_embedding_matrix.shape[0] - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio = explained_variance_ / total_var
    
    elif centering_by == 'col':
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
            return neural_compression(
                k=k, n=averaged_embedding_matrix.shape[1]
            )
        else:
            explained_variance_cumu_ += explained_variance_ratio[k]
            k += 1
    
    print(
        f'explained_variance_cumu_={explained_variance_cumu_}, k={k}'
    )
    return neural_compression(
        k=k, n=averaged_embedding_matrix.shape[1]
    )


def execute(roi, subs, tasks, num_processes, centering_by):
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
                            dataType, conditions, smooth_beta, centering_by
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

    fig, ax = plt.subplots()
    for z in range(num_types):
        problem_type = problem_types[z]
        # here we extract a list of res_obj and 
        # extract the actual compression scores.
        list_of_res_obj = type2metric[problem_type]
        # `metrics` is all scores over subs for one (problem_type)
        metrics = [res_obj.get() for res_obj in list_of_res_obj]
        mean = np.mean(metrics)
        std = np.std(metrics)
        print(f'Type=[{problem_type}], roi=[{roi}], mean=[{mean:.3f}], std=[{std:.3f}], centerBy=[{centering_by}]')
        
        ax.errorbar(
            x=z+1,
            y=mean,
            yerr=std,
            label=f'Type {problem_type}',
            fmt='o'
        )
        
    ax.set_ylabel('Compression')
    ax.set_xlabel('Problem Type')
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([1, 2, 6])
    plt.title(f'ROI: {roi}, centering by {centering_by}')
    plt.legend()
    plt.savefig(f'Ahlheim_results/{roi}_centeringBy{centering_by}.png')
        

if __name__ == '__main__':    
    root_path = '/home/ken/projects/brain_data'
    glm_path = 'glm_run-estimate'
    roi = 'RHLOC'
    num_subs = 23
    num_types = 3
    dataType = 'beta'
    num_conditions = 16
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    num_subs = len(subs)
    conditions = [f'{i:04d}' for i in range(1, num_conditions+1)]
    tasks = [1, 2, 3]
    runs = [2, 3, 4]  # will average over.
    num_runs = len(runs)
    smooth_beta = 2
    num_processes = 70
    centering_by = 'row'
    if dataType == 'beta':
        # ignore `*_fb` conditions
        conditions = [f'{i:04d}' for i in range(1, num_conditions, 2)]
        num_conditions = len(conditions)
    
    execute(
        roi=roi, 
        subs=subs, 
        tasks=tasks, 
        num_processes=num_processes,
        centering_by=centering_by
    )