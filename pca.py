import os
import multiprocessing
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy import linalg
import scipy.stats as stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

from roi_rsa import merge_n_smooth_mask, transform_mask_MNI_to_T1, applyMask

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
    
    
def apply_PCA(roi, root_path, glm_path, roi_path, sub, task, run, dataType, conditions, smooth_beta, centering_by):
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
    print('beta_weights_masked.shape', beta_weights_masked.shape)
    
    # PCA
    if centering_by == 'row':
        # mean-center (by row)
        # NOTE: sklearn PCA default is by column.
        row_mean = np.mean(beta_weights_masked, axis=1).reshape(-1, 1)
        beta_weights_masked -= row_mean
        
        # SVD
        U, S, Vt = linalg.svd(beta_weights_masked, full_matrices=False)
        explained_variance_ = (S ** 2) / (beta_weights_masked.shape[0] - 1)
        total_var = explained_variance_.sum()
        explained_variance_ratio = explained_variance_ / total_var
    
    elif centering_by == 'col':
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
            return neural_compression(k=k)
        else:
            explained_variance_cumu_ += explained_variance_ratio[k]
            k += 1
    
    print(
        f'explained_variance_cumu_={explained_variance_cumu_}, k={k}'
    )
    return neural_compression(k=k)
        

def compression_execute(roi, subs, runs, tasks, num_processes, centering_by):
    """
    Top-level execute that apply PCA, get top k, 
    compute compression score and plot for all subs, runs, tasks.
    """
    if not os.path.exists(f'compression_results/{roi}_centeringBy{centering_by}.npy'):
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
            run2type2metric = defaultdict(lambda: defaultdict(list))
            for run in runs:
                for task in tasks:
                    for sub in subs:
                        # done once for each sub
                        transform_mask_MNI_to_T1(sub=sub, roi=roi, roi_path=roi_path, root_path=root_path)
                        
                        # per (sub, task, run) compression
                        res_obj = pool.apply_async(
                            apply_PCA, 
                            args=[
                                roi, root_path, glm_path, roi_path, 
                                sub, task, run, 
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
                        run2type2metric[run][problem_type].append(res_obj)
            
            pool.close()
            pool.join()
        
        # save & plot compression results
        # ref: https://stackoverflow.com/questions/68629457/seaborn-grouped-violin-plot-without-pandas
        x = []    # each sub's run
        y = []    # each sub problem_type's compression
        hue = []  # each sub problem_type
        means = []
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
                # `metrics` is all scores over subs for one (problem_type, run)
                metrics = [res_obj.get() for res_obj in list_of_res_obj]
                # metrics = list(metrics - np.mean(metrics))
                means.append(np.mean(metrics))
                x.extend([f'{run}'] * num_subs)
                y.extend(metrics)
                hue.extend([f'Type {problem_type}'] * num_subs)

        compression_results = {}
        compression_results['x'] = x
        compression_results['y'] = y
        compression_results['hue'] = hue
        compression_results['means'] = means
        np.save(f'compression_results/{roi}_centeringBy{centering_by}.npy', compression_results)
    
    else:
        print('[NOTE] Loading saved results, make sure it does not need update.')
        # load presaved results dictionary.
        compression_results = np.load(f'compression_results/{roi}_centeringBy{centering_by}.npy', allow_pickle=True).ravel()[0]

    # plot violinplots / stripplots
    fig, ax = plt.subplots()
    x = compression_results['x']
    y = compression_results['y']
    hue = compression_results['hue']
    means = compression_results['means']
    palette = {'Type 1': 'pink', 'Type 2': 'green', 'Type 6': 'blue'}
    ax = sns.stripplot(x=x, y=y, hue=hue, palette=palette, dodge=True, alpha=0.8, jitter=0.3, size=4)
    
    # plot mean/median
    num_bars = int(len(y) / (num_subs))
    positions = []
    margin = 0.24
    problem_types = [1, 2, 6]
    for per_run_center in ax.get_xticks():
        positions.append(per_run_center-margin)
        positions.append(per_run_center)
        positions.append(per_run_center+margin)
    
    labels = []
    final_run_data = []  # for t-test
    # global_index: 0-11
    for global_index in range(num_bars):
        # run: 1-4
        run = global_index // len(problem_types) + 1
        # within_run_index: 0-2
        within_run_index = global_index % len(problem_types)        
        problem_type = problem_types[within_run_index]
        
        # data
        per_type_data = y[ global_index * num_subs : (global_index+1) * num_subs ]
        position = [positions[global_index]]
        
        q1, md, q3 = np.percentile(per_type_data, [25,50,75])
        mean = np.mean(per_type_data)
        std = np.std(per_type_data)
        median_obj = ax.scatter(position, md, marker='s', color='red', s=33, zorder=3)
        mean_obj = ax.scatter(position, mean, marker='^', color='k', s=33, zorder=3)
        
        # print out stats
        print(f'Type=[{problem_type}], run=[{run}], mean=[{mean:.3f}], std=[{std:.3f}], centerBy=[{centering_by}]')
        if within_run_index == 2:
            print('-'*60)
        
        if global_index in range(num_bars)[-3:]:
            # print(global_index)
            final_run_data.append(per_type_data)
    
    # independent t-test on the last run's 3 problem_types:
    for i in range(len(final_run_data)):
        for j in range(len(final_run_data)):
            if i >= j:
                continue
            print(f'Type {problem_types[i]} vs {problem_types[j]}')
            print(stats.ttest_ind(final_run_data[i], final_run_data[j]))
    
    # hacky way getting legend
    median_obj = ax.scatter(position, md, marker='s', color='red', s=33, zorder=3, label='median')
    mean_obj = ax.scatter(position, mean, marker='^', color='k', s=33, zorder=3, label='mean')
    plt.legend()
    ax.set_xlabel('Learning Blocks')
    ax.set_ylabel(f'{roi} Compression')
    plt.title(f'ROI: {roi}, centering by {centering_by}')
    plt.savefig(f'compression_results/{roi}_centeringBy{centering_by}.png')


def mixed_effects_analysis(roi, centering_by):
    """
    Perform a two-way ANOVA analysis as an alternative of 
    the bayesian mixed effect analysis in Mack et al., 2020.
    
    Independent variable: 
        problem_type, learning_block, interaction
    Dependent variable:
        compression score
    """
    import pingouin as pg
    if not os.path.exists(f"compression_results/{roi}_centeringBy{centering_by}.csv"):
        subjects = ['subject']
        types = ['problem_type']
        learning_blocks = ['learning_block']
        compression_scores = ['compression_score']

        compression_results = np.load(
            f'compression_results/{roi}_centeringBy{centering_by}.npy', 
            allow_pickle=True).ravel()[0]
        y = compression_results['y']
        num_bars = int(len(y) / (num_subs))
        problem_types = [1, 2, 6]
        
        # global_index: 0-11
        for global_index in range(num_bars):
            # run: 1-4 i.e. learning block
            run = global_index // len(problem_types) + 1
            # within_run_index: 0-2
            within_run_index = global_index % len(problem_types)        
            problem_type = problem_types[within_run_index]
            print(f'run={run}, type={problem_type}')
            
            # data
            per_type_data = y[ global_index * num_subs : (global_index+1) * num_subs ]
            
            for s in range(num_subs):
                sub = subs[s]
                subjects.append(sub)
                types.append(problem_type)
                learning_blocks.append(run)
                compression_scores.append(per_type_data[s])
            
        subjects = np.array(subjects)
        types = np.array(types)
        learning_blocks = np.array(learning_blocks)
        compression_scores = np.array(compression_scores)
        
        df = np.vstack((
            subjects, 
            types, 
            learning_blocks, 
            compression_scores
        )).T
        pd.DataFrame(df).to_csv(
            f"compression_results/{roi}_centeringBy{centering_by}.csv", 
            index=False, 
            header=False
        )
        
    df = pd.read_csv(f"compression_results/{roi}_centeringBy{centering_by}.csv")
        
    # sns.stripplot(
    #     x='learning_block',
    #     y='compression_score',
    #     hue='problem_type',
    #     data=df,
    #     dodge=True
    # )
    # plt.savefig('test.png')
    
    # two-way ANOVA:
    res = pg.rm_anova(
        dv='compression_score',
        within=['problem_type', 'learning_block'],
        subject='subject',
        data=df, 
    )
    print(res)
        
                   
if __name__ == '__main__':    
    root_path = '/home/ken/projects/brain_data'
    glm_path = 'glm_trial-estimate'
    roi = 'LHHPC'
    num_subs = 23
    num_types = 3
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
    centering_by = 'row'
    if dataType == 'beta':
        # ignore `_rp*_fb` conditions, the remaining are `_rp*` conditions.
        conditions = [f'{i:04d}' for i in range(1, num_conditions, 2)]
    
    # compression_execute(
    #     roi=roi, 
    #     subs=subs, 
    #     runs=runs, 
    #     tasks=tasks, 
    #     num_processes=num_processes,
    #     centering_by=centering_by
    # )
    
    mixed_effects_analysis(roi=roi, centering_by=centering_by)

    