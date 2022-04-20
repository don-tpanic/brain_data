import numpy as np
import pandas as pd
import multiprocessing
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.decomposition import PCA
from roi_rsa import applyMask

"""Reproducing key results from Mack et al., 2020
1. Figure2b results: neural compression against learning blocks across 
    problem types (complexities)
2. See [sustain_plus] for brain-model prediction (neural compression - attn compression)
"""

def neural_compression(k, n=32):
    """
    k: number of PCs that explains 90% variance
    n: number of trials within a block (i.e. run)
    """
    return 1 - (k/n)
    
    
def apply_PCA(roi, root_path, glm_path, roi_path, sub, task, run, dataType, conditions, smooth_beta):
    """
    Apply PCA onto an embedding matrix of (n_voxels, n_trials) of a given (roi, sub, task, run),
    where the columns are beta weights of a given roi of a given trial.
    
    ROI extraction uses `applyMask()` from `roi_rsa.py`
    
    return:
    -------
        k: number of principle components that explains 90% variance.
    """
    # embedding mtx (rp, roi_size)
    beta_weights_masked = []
    for rp in range(1, num_repetitions_per_run+1):
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
    
    # (n_voxels, n_trials)
    beta_weights_masked = np.array(beta_weights_masked).T
    print('beta_weights_masked.shape', beta_weights_masked.shape)
    
    # PCA
    pca = PCA(n_components=beta_weights_masked.shape[1], random_state=42)
    pca.fit(beta_weights_masked)
    explained_variance_ratio = pca.explained_variance_ratio_
    
    # return the k PCs that explain at least 90% variance
    k = 0
    explained_variance_cumu_ = 0
    while k < beta_weights_masked.shape[1]:
        if explained_variance_cumu_ >= 0.9:
            print(
                f'explained_variance_cumu_={explained_variance_cumu_}, k={k}'
            )
            return k
        else:
            explained_variance_cumu_ += explained_variance_ratio[k]
            k += 1
    
    print(
        f'explained_variance_cumu_={explained_variance_cumu_}, k={k}'
    )
    compression = neural_compression(k=k)
    return compression
        

def compression_execute(roi, subs, runs, tasks, num_processes):
    """
    Top-level execute that apply PCA, get top k, 
    compute compression score and plot for all subs, runs, tasks.
    """
    with multiprocessing.Pool(num_processes) as pool:

        if 'HPC' not in roi:
                roi_path = 'ROIs/ProbAtlas_v4/subj_vol_all'
        else:
            roi_path = 'ROIs/HPC'
        
        # compute & collect compression
        run2type2metric = defaultdict(lambda: defaultdict(list))
        for run in runs:
            for task in tasks:
                for sub in subs:
                    # per (sub, task, run) compression
                    res_obj = pool.apply_async(
                        apply_PCA, 
                        args=[
                            roi, root_path, glm_path, roi_path, 
                            sub, task, run, 
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
                    run2type2metric[run][problem_type].append(res_obj)
        
        pool.close()
        pool.join()
                    
    # plot compression
    # ref: https://stackoverflow.com/questions/68629457/seaborn-grouped-violin-plot-without-pandas
    fig, ax = plt.subplots()
    x = []    # each sub's run
    y = []    # each sub problem_type's compression
    hue = []  # each sub problem_type
    for run in runs:
        print(f'--------- run {run} ---------')
        type2metric = run2type2metric[run]
        num_types = len(type2metric.keys())
        problem_types = sorted(list(type2metric.keys()))
        print(f'num_types={num_types}')
        
        for z in range(num_types):
            problem_type = problem_types[z]
            # here we extract a list of res_obj and 
            # extract the actual compression scores.
            list_of_res_obj = type2metric[problem_type]
            metrics = [res_obj.get() for res_obj in list_of_res_obj]
            # TODO: how did Mike mean-center?
            # metrics = list(
            #     (metrics - np.mean(metrics)) / (np.max(metrics) - np.min(metrics))
            # )
            x.extend([f'{run}'] * num_subs)
            y.extend(metrics)
            hue.extend([f'Type {problem_type}'] * num_subs)
            
    print(x, '\n\n')
    print(y)
    assert len(x) == len(y), f"Unequal length of x, y, x={len(x)}, y={len(y)}"
    palette = {'Type 1': 'pink', 'Type 2': 'green', 'Type 6': 'blue'}
    sns.violinplot(x=x, y=y, hue=hue, palette=palette)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels([1, 2, 3, 4])
    ax.set_xlabel('Learning Blocks')
    ax.set_ylabel('vmPFC Compression')
    plt.savefig('test.png')


if __name__ == '__main__':    
    root_path = '/home/ken/projects/brain_data'
    glm_path = 'glm_trial-estimate'
    roi = 'LHHPC'
    num_subs = 23
    dataType = 'beta'
    num_conditions = 64
    subs = [f'{i:02d}' for i in range(2, num_subs+2)]
    conditions = [f'{i:04d}' for i in range(1, num_conditions+1)]
    tasks = [1, 2, 3]
    runs = [1, 2, 3, 4]
    num_runs = len(runs)
    num_repetitions_per_run = 4
    smooth_beta = 2
    num_processes = 70
    if dataType == 'beta':
        # ignore `_rp*_fb` conditions, the remaining are `_rp*` conditions.
        conditions = [f'{i:04d}' for i in range(1, num_conditions, 2)]
    
    compression_execute(
        roi=roi, 
        subs=subs, 
        runs=runs, 
        tasks=tasks, 
        num_processes=num_processes
    )
    
    