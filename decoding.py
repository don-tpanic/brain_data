import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import dill
import itertools
import multiprocessing
from functools import partial
from collections import defaultdict

import numpy as np
import seaborn as sns
import scipy.stats as stats
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.utils import shuffle
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_validate

from roi_rsa import applyMask


stimuli = ['000', '001', '010', '011', '100', '101', '110', '111']

def stimuli2conditions(conditions, num_repetitions_per_run):
    """While we want to train classifier based on stimuli, the conditions 
    at rep-level are unique for each repetition of the same stimulus. Therefore,
    to group the same stimulus from different repetitions as data-points of 
    the same class, we need to make sure the conditions are grouped by stimulus.
    
    In result, we return a mapper whose keys are the stimuli; each key corresponds 
    to the unique conditions of the same stimulus.
    
    i.e.
        mapper={
            '000': ['0001', '0003', '0005', '0007'], 
            '001': ['0009', '0011', '0013', '0015'], 
            '010': ['0017', '0019', '0021', '0023'], 
            '011': ['0025', '0027', '0029', '0031'], 
            '100': ['0033', '0035', '0037', '0039'], 
            '101': ['0041', '0043', '0045', '0047'], 
            '110': ['0049', '0051', '0053', '0055'], 
            '111': ['0057', '0059', '0061', '0063']
        }
    
    inputs:
    -------
        conditions: each stimulus * 4 reps (1run), excluded _fb
    """
    mapper = defaultdict(list)
    for i in range(len(stimuli)):
        stimulus = stimuli[i]
        # one stimulus has 4 conditions (i.e. repetitions)
        for j in range(
            i*num_repetitions_per_run, 
            (i+1)*num_repetitions_per_run):
            mapper[stimulus].append(conditions[j])
    return mapper


def per_stimuli_pair_train_and_eval(
        runs, stimulus1, stimulus2, 
        roi,
        root_path,
        glm_path,
        roi_path,
        sub, 
        task, 
        dataType, 
        smooth_beta,
        mapper
    ):
    """Single train and eval on a pair of stimuli.
    1. Data-points across runs for a single pair are collected
    2. Cross-validation is applied. Notice, we always test on pairs
       from the same run. Therefore, a trick to create the folds of 
       cv is simply to use the run as spliter.
    
    inputs:
    -------
        X: beta weights of a pair of stimuli across all repetitions.
        Y: conditions of a pair of stimuli across all repetitions.
        
        # e.g. 
        # Y=['000' '000' '000' '000' '001' '001' '001' '001' '000' '000' '000' '000'
        #  '001' '001' '001' '001' '000' '000' '000' '000' '001' '001' '001' '001'
        #  '000' '000' '000' '000' '001' '001' '001' '001']
    
    return:
    -------
        val_acc: average validation accuracy of this single pair of stimuli.
    """
    X = []
    Y = []
    run_info = []  # helpful for cross-validation
    for run in runs:
        for stimulus in [stimulus1, stimulus2]:                            
            for condition in mapper[stimulus]:                
                # get per-condition beta weights
                _, _, fmri_masked = applyMask(
                    roi,
                    root_path,
                    glm_path,
                    roi_path,
                    sub, 
                    task, 
                    run, 
                    dataType, 
                    condition,
                    smooth_beta
                )
                X.append(fmri_masked)
                Y.append(stimulus)
                run_info.append(run)              

    # convert to ndarray so cv masking works correctly
    X = np.array(X)
    Y = np.array(Y)
    run_info = np.array(run_info)
    
    # convert Y to classifier-compatible labels.
    le = preprocessing.LabelEncoder()
    Y = le.fit(Y).transform(Y)
    
    # cross-validation
    # runs are fold_ids
    test_score = []
    for fold_id in runs:
        val_mask = (run_info == fold_id)
        train_mask = ~val_mask
        
        X_train = X[train_mask, :]
        Y_train = Y[train_mask]
        X_val = X[val_mask, :]
        Y_val = Y[val_mask]
        
        # fit and eval for one fold        
        classifier = LinearSVC(C=0.1)
        classifier.fit(X=X_train, y=Y_train)
        test_score.append(
            classifier.score(X=X_val, y=Y_val))
    
    print(test_score)  
    val_acc = np.mean(test_score)
    print(f'[Check] sub{sub}, {stimulus1}-{stimulus2}, val_acc={val_acc}')
    return val_acc


def decoding_error_execute(
        rois, 
        conditions, 
        num_repetitions_per_run, 
        num_runs,
        num_processes=72
    ):
    """
    1. Given each problem type and subject, we iterate pairs of stimuli,
    for each stimuli pair, we group all repetitions （of all runs)
    of the same stimulus as from one class.
    2. For each stimuli pair, we train a classifier with 
        cross-validation and return a single accuracy for that pair. 
    3. We finally return all 1-accuracies for each problem type.
    """
    results_path = 'decoding_results'
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    
    for roi in rois:
        if 'HPC' not in roi:
            roi_path = 'ROIs/ProbAtlas_v4/subj_vol_all'
        else:
            roi_path = 'ROIs/HPC'
            
        if not os.path.exists(f'{results_path}/decoding_error_{num_runs}runs_{roi}.npy'):
            if 'HPC' not in roi:
                roi_path = 'ROIs/ProbAtlas_v4/subj_vol_all'
            else:
                roi_path = 'ROIs/HPC'
                
            mapper = stimuli2conditions(
                conditions=conditions, 
                num_repetitions_per_run=num_repetitions_per_run
            )
            
            with multiprocessing.Pool(num_processes) as pool:
                decoding_error = defaultdict(lambda: defaultdict(list))
                for problem_type in problem_types:
                    for sub in subs:
                        for stimulus1, stimulus2 in itertools.combinations(stimuli, r=2):  
                            if int(sub) % 2 == 0:
                                if problem_type == 1:
                                    task = 2
                                elif problem_type == 2:
                                    task = 3
                                else:
                                    task = 1
                            else:
                                if problem_type == 1:
                                    task = 3
                                elif problem_type == 2:
                                    task = 2
                                else:
                                    task = 1
                            
                            res_obj = pool.apply_async(
                                per_stimuli_pair_train_and_eval, 
                                args=[
                                    runs, stimulus1, stimulus2, 
                                    roi,
                                    root_path,
                                    glm_path,
                                    roi_path,
                                    sub, 
                                    task, 
                                    dataType, 
                                    smooth_beta,
                                    mapper
                                ]
                            )
                            # res_obj.get() is val_acc of (sub, pair)
                            decoding_error[problem_type][sub].append(res_obj)
                pool.close()
                pool.join()
        
            decoding_error_collector = defaultdict(list)
            for problem_type in problem_types:
                for sub in subs:
                    # all pairs of (type, sub)
                    per_type_results_obj = decoding_error[problem_type][sub]
                    per_type_results = [res_obj.get() for res_obj in per_type_results_obj]
                    # only need average over pairs and convert to 1-acc
                    decoding_error_collector[problem_type].append(1-np.mean(per_type_results))
            np.save(f'{results_path}/decoding_error_{num_runs}runs_{roi}.npy', decoding_error_collector)
        
        else:
            decoding_error_collector = np.load(
                f'{results_path}/decoding_error_{num_runs}runs_{roi}.npy', 
                allow_pickle=True).ravel()[0]
        
        # plotting
        visualize_decoding_error(decoding_error_collector, num_runs, roi)
        
        # print key stats
        for problem_type in problem_types:
            print(
                f'ROI={roi}, Type={problem_type}, '\
                f'err={np.mean(decoding_error_collector[problem_type]):.3f}, '\
                f'sem={stats.sem(decoding_error_collector[problem_type]):.3f}'
            )
        
        # stats significance fitting linear regression
        average_coef, t, p, _ = decoding_error_regression(
            decoding_error_collector,
            num_subs=num_subs, 
            problem_types=problem_types
        )
        print(f'-'*37)


def visualize_decoding_error(decoding_error_collector, num_runs, roi):
    fig, ax = plt.subplots()
    x = []    # each sub's problem_type
    y = []    # each sub problem_type's decoding error
    for problem_type in problem_types:
        per_run_metrics = decoding_error_collector[problem_type]
        x.extend([f'Type {problem_type}'] * num_subs)
        y.extend(per_run_metrics)
    
    palette = {'Type 1': 'pink', 'Type 2': 'green', 'Type 6': 'blue'}
    ax = sns.boxplot(x=x, y=y, palette=palette)
    ax.set_xlabel('Problem Type')
    ax.set_ylabel(f'{roi} Neural Stimulus Reconstruction Loss\n(1 - decoding accuracy)')
    plt.tight_layout()
    plt.savefig(f'decoding_results/decoding_error_{num_runs}runs_{roi}.png')


def decoding_error_regression(
        decoding_error_collector, 
        num_subs, 
        problem_types
    ):
    """Fitting linear regression models to per subject decoding 
    1-accuracies over problem_types. This way, we can read off the 
    regression coefficient on whether there is a down trend of 
    decoding 1-accuracies as task difficulty increases in order to 
    test statistic significance of our finding that the harder 
    the problem, the better the decoding (hence lower recon loss).
    
    Impl:
    -----
        `decoding_error_collector` are saved in format:
            {
             'Type1': [sub02_err, sub03_err, ..],
             'Type2}: ...
            }
        
        To fit linear regression per subject across types, we 
        convert the format to a matrix where each row is a subject,
        and each column is a problem_type.
    """
    import pingouin as pg
    from scipy import stats
    
    group_results_by_subject = np.ones((num_subs, len(problem_types)))
    for z in range(len(problem_types)):
        problem_type = problem_types[z]
        # [sub02_err, sub03_err, ...]
        per_type_all_subjects = decoding_error_collector[problem_type]
        for s in range(num_subs):
            group_results_by_subject[s, z] = per_type_all_subjects[s]
    
    all_coefs = []
    for s in range(num_subs):
        X_sub = problem_types
        # [sub02_type1_err, sub02_type2_err, ...]
        y_sub = group_results_by_subject[s, :]
        coef = pg.linear_regression(X=X_sub, y=y_sub, coef_only=True)
        # print(f'sub{subs[s]}, {y_sub}, coef={coef[-1]:.3f}')
        all_coefs.append(coef[-1])
    
    average_coef = np.mean(all_coefs)
    print(f'average_coef={average_coef:.3f}')
    t, p = stats.ttest_1samp(all_coefs, popmean=0)
    print(f't={t:.3f}, one-tailed p={p/2:.3f}')
    return average_coef, t, p/2, all_coefs
        

def paired_ttest_between_rois(roi1, roi2, num_runs):
    """An extra significance testing by pair ttesting
    the linear regression coefficients of two different
    ROIs. The purpose is to see whether LOC is significantly
    different from HPC in terms of decoding pattern across
    problem types.
    """
    results_path = 'decoding_results'
    decoding_error_collector_roi1 = np.load(
        f'{results_path}/decoding_error_{num_runs}runs_{roi1}.npy', 
        allow_pickle=True).ravel()[0]

    decoding_error_collector_roi2 = np.load(
        f'{results_path}/decoding_error_{num_runs}runs_{roi2}.npy', 
        allow_pickle=True).ravel()[0]
    
    _, _, _, all_coefs_roi1 = decoding_error_regression(
        decoding_error_collector=decoding_error_collector_roi1, 
        num_subs=num_subs, 
        problem_types=problem_types
    )
    
    _, _, _, all_coefs_roi2 = decoding_error_regression(
        decoding_error_collector=decoding_error_collector_roi2, 
        num_subs=num_subs, 
        problem_types=problem_types
    )
    
    t, p = stats.ttest_rel(all_coefs_roi1, all_coefs_roi2)
    print(f'{roi1} vs {roi2} t={t:.3f}, one-tailed p={p/2:.3f}')
    
     
##### overtime analysis ####
def per_stimuli_pair_train_and_eval_overtime(
        runs, stimulus1, stimulus2, 
        roi,
        root_path,
        glm_path,
        roi_path,
        sub, 
        task, 
        dataType, 
        smooth_beta,
        mapper
    ):
    """
    To obtain overtime results, we need to collect 
    decoding accuracy for each fold separately. 
    
    Impl:
    -----
        We store per fold (i.e. run) results using 
        a dictionary whose keys are the run ids and 
        values are the decoding accuracies.
    
    return:
    -------
        e.g. 
            test_score={
                run2: val_acc,
                run3: val_acc,
                ...
            }
    """
    X = []
    Y = []
    run_info = []  # helpful for cross-validation
    for run in runs:
        for stimulus in [stimulus1, stimulus2]:                            
            for condition in mapper[stimulus]:                
                # get per-condition beta weights
                _, _, fmri_masked = applyMask(
                    roi,
                    root_path,
                    glm_path,
                    roi_path,
                    sub, 
                    task, 
                    run, 
                    dataType, 
                    condition,
                    smooth_beta
                )
                X.append(fmri_masked)
                Y.append(stimulus)
                run_info.append(run)              

    # convert to ndarray so cv masking works correctly
    X = np.array(X)
    Y = np.array(Y)
    run_info = np.array(run_info)
    
    # convert Y to classifier-compatible labels.
    le = preprocessing.LabelEncoder()
    Y = le.fit(Y).transform(Y)
    
    # cross-validation
    # runs are fold_ids
    test_score = {}
    for fold_id in runs:
        val_mask = (run_info == fold_id)
        train_mask = ~val_mask
        
        X_train = X[train_mask, :]
        Y_train = Y[train_mask]
        X_val = X[val_mask, :]
        Y_val = Y[val_mask]
        
        # fit and eval for one fold        
        classifier = LinearSVC(C=0.1)
        classifier.fit(X=X_train, y=Y_train)
        test_score[fold_id] = classifier.score(X=X_val, y=Y_val)
    
    print(f'test_score={test_score}')
    return test_score


def decoding_error_overtime_execute(
        rois, 
        conditions, 
        num_repetitions_per_run, 
        num_runs,
        num_processes=72
    ):
    """
    overtime meaning that we collect decoding errors 
    split by runs and do not average over runs.
    """
    results_path = 'decoding_results_overtime'
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    
    for roi in rois:
        if 'HPC' not in roi:
            roi_path = 'ROIs/ProbAtlas_v4/subj_vol_all'
        else:
            roi_path = 'ROIs/HPC'
            
        if not os.path.exists(f'{results_path}/decoding_error_{num_runs}runs_{roi}.pkl'):
            mapper = stimuli2conditions(
                conditions=conditions, 
                num_repetitions_per_run=num_repetitions_per_run
            )
            
            with multiprocessing.Pool(num_processes) as pool:    
                decoding_error = defaultdict(lambda: defaultdict(list))
                for problem_type in problem_types:
                    for sub in subs:
                        for stimulus1, stimulus2 in itertools.combinations(stimuli, r=2):  
                            if int(sub) % 2 == 0:
                                if problem_type == 1:
                                    task = 2
                                elif problem_type == 2:
                                    task = 3
                                else:
                                    task = 1
                            else:
                                if problem_type == 1:
                                    task = 3
                                elif problem_type == 2:
                                    task = 2
                                else:
                                    task = 1
                            
                            res_obj = pool.apply_async(
                                per_stimuli_pair_train_and_eval_overtime, 
                                args=[
                                    runs, stimulus1, stimulus2, 
                                    roi,
                                    root_path,
                                    glm_path,
                                    roi_path,
                                    sub, 
                                    task, 
                                    dataType, 
                                    smooth_beta,
                                    mapper
                                ]
                            )
                            # per (type, subject, pair) of all runs
                            # ...[sub] = [a list of dictionaries] where 
                            # there are 28 dictionaries each is a pair of stimuli,
                            # and each dictionary is of all runs' val_acc.
                            decoding_error[problem_type][sub].append(res_obj)
                pool.close()
                pool.join()

            decoding_error_collector = defaultdict(lambda: defaultdict(list))
            for problem_type in problem_types:
                for sub in subs:
                    # per (type, subject, pair) of all runs
                    per_type_results_obj = decoding_error[problem_type][sub]
                    # a list of dictionaries where each dict is all runs' val_acc
                    per_type_results = [res_obj.get() for res_obj in per_type_results_obj]         
                    # iterate all dicts and gather val_acc per run_id
                    
                    # the temporary dictionary will store all pairs' val_acc
                    # grouped by run. So we can expect each key corresponds 
                    # to a list of 28 pairs' val_acc for that run.
                    temp = defaultdict(list)
                    for per_pair_dict in per_type_results:
                        for run_id in per_pair_dict.keys():
                            val_acc = per_pair_dict[run_id]
                            # collect all pairs by run_id and average over pairs
                            # so that for each (sub, run_id) there is one value
                            temp[run_id].append(1-val_acc)
                            
                    # since we want evetually each subject has 1 single value for each run,
                    # here we need to go over the runs of temp and compute the average 
                    # over pairs of every run.
                    for run_id in runs:
                        assert len(temp[run_id]) == len(list(itertools.combinations(stimuli, r=2)))
                        decoding_error_collector[problem_type][run_id].append(np.mean(temp[run_id]))
                                                
            # https://stackoverflow.com/questions/16439301/cant-pickle-defaultdict/16439531#16439531 
            with open(f'{results_path}/decoding_error_{num_runs}runs_{roi}.pkl', 'wb') as f:
                dill.dump(decoding_error_collector, f)
        
        else:
            with open(f'{results_path}/decoding_error_{num_runs}runs_{roi}.pkl', 'rb') as f:
                decoding_error_collector = dill.load(f)
        
        visualize_decoding_error_overtime(decoding_error_collector, num_runs, roi)
        
        for problem_type in problem_types:
            for run in runs:
                print(
                    f'ROI {roi}, Type {problem_type}, run {run}, err={np.mean(decoding_error_collector[problem_type][run]):.3f}, '\
                    f'sem={stats.sem(decoding_error_collector[problem_type][run]):.3f}'
                )
            print('\n\n')


def visualize_decoding_error_overtime(decoding_error_collector, num_runs, roi):
    fig, ax = plt.subplots()
    x = []    # each sub's problem_type
    y = []    # each sub problem_type's decoding error
    hue = []  # each sub run
    for problem_type in problem_types:
        for run in runs:
            per_run_metrics = decoding_error_collector[problem_type][run]
            x.extend([f'Type {problem_type}'] * num_subs)
            y.extend(per_run_metrics)
            hue.extend([f'run {run}'] * num_subs)
    
    if num_runs == 3:
        palette = {'run 2': 'pink', 'run 3': 'green', 'run 4': 'blue'}
    elif num_runs == 4:
        palette = {'run 1': 'red', 'run 2': 'pink', 'run 3': 'green', 'run 4': 'blue'}
        
    ax = sns.boxplot(x=x, y=y, hue=hue, palette=palette)
    ax.set_xlabel('Learning Blocks')
    ax.set_ylabel(f'{roi} Neural Stimulus Reconstruction Loss\n(1 - decoding accuracy)')
    plt.tight_layout()
    plt.savefig(f'decoding_results_overtime/decoding_error_{num_runs}runs_{roi}.png')
    
    
if __name__ == '__main__':
    root_path = '/home/ken/projects/brain_data'
    glm_path = 'glm_trial-estimate'
    rois = ['LHHPC', 'RHHPC', 'LOC']
    num_subs = 23
    dataType = 'beta'
    num_conditions = 64  # exc. bias term (8*4rp + 8_fb*4rp)
    problem_types = [1, 2, 6]
    runs = [2, 3, 4]
    smooth_beta = 2
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    num_subs = len(subs)
    num_repetitions_per_run = 4
    
    if dataType == 'beta':
        # This is to skip conditions that are _fb
        conditions = [f'{i:04d}' for i in range(1, num_conditions, 2)]
        num_conditions = len(conditions)

    decoding_error_execute(
        rois=rois, conditions=conditions, 
        num_repetitions_per_run=num_repetitions_per_run,
        num_runs=len(runs),
        num_processes=72
    )
    
    # paired_ttest_between_rois(
    #     roi1='LOC',
    #     roi2='LHHPC',
    #     num_runs=len(runs)
    # )
    
    # decoding_error_overtime_execute(
    #     rois=rois, conditions=conditions, 
    #     num_repetitions_per_run=num_repetitions_per_run,
    #     num_runs=len(runs),
    #     num_processes=72
    # )    
