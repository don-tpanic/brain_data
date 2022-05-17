import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import itertools
import multiprocessing
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
                
                # print(
                #     f'[Check] sub{sub}, '\
                #     f'run={run}, '\
                #     f'stimulus={stimulus}, '\
                #     f'condition={condition}'
                # )
                
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


def decoding_accuracy_execute(
        roi, 
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
    3. We finally return all accuracies for each problem type.
    """
    results_path = 'decoding_results'
    if not os.path.exists(results_path):
        os.mkdir(results_path)
    
    if not os.path.exists(f'{results_path}/decoding_accuracy_{num_runs}runs_{roi}.npy'):
        if 'HPC' not in roi:
            roi_path = 'ROIs/ProbAtlas_v4/subj_vol_all'
        else:
            roi_path = 'ROIs/HPC'
            
        mapper = stimuli2conditions(
            conditions=conditions, 
            num_repetitions_per_run=num_repetitions_per_run
        )
        
        with multiprocessing.Pool(num_processes) as pool:
            decoding_accuracy = defaultdict(lambda: defaultdict(list))
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
                        decoding_accuracy[problem_type][sub].append(res_obj)
            pool.close()
            pool.join()
    
        decoding_accuracy_collector = defaultdict(list)
        for problem_type in problem_types:
            for sub in subs:
                # all pairs of (type, sub)
                per_type_results_obj = decoding_accuracy[problem_type][sub]
                per_type_results = [res_obj.get() for res_obj in per_type_results_obj]
                # only need average of all pairs
                decoding_accuracy_collector[problem_type].append(np.mean(per_type_results))
        np.save(f'{results_path}/decoding_accuracy_{num_runs}runs_{roi}.npy', decoding_accuracy_collector)
    
    else:
        decoding_accuracy_collector = np.load(
            f'{results_path}/decoding_accuracy_{num_runs}runs_{roi}.npy', 
            allow_pickle=True).ravel()[0]
    
    print(
        f'Type 1 acc={np.mean(decoding_accuracy_collector[1]):.3f}, '\
        f'sem={stats.sem(decoding_accuracy_collector[1]):.3f}'
    )
    print(
        f'Type 2 acc={np.mean(decoding_accuracy_collector[2]):.3f}, '\
        f'sem={stats.sem(decoding_accuracy_collector[2]):.3f}'
    )
    print(
        f'Type 6 acc={np.mean(decoding_accuracy_collector[6]):.3f}, '\
        f'sem={stats.sem(decoding_accuracy_collector[6]):.3f}'
    )  


def decoding_accuracy_regression(roi, num_runs, num_subs, problem_types):
    """Fitting linear regression models to per subject decoding 
    accuracies over problem_types. This way, we can read off the 
    regression coefficient on whether there is an up trend of 
    decoding accuracies as task difficulty increases in order to 
    test statistic significance of our finding that the harder 
    the problem, the better the decoding (hence lower recon loss).
    
    Impl:
    -----
        `decoding_accuracy_collector` are saved in format:
            {
             'Type1': [sub02_acc, sub03_acc, ..],
             'Type2}: ...
            }
        
        To fit linear regression per subject across types, we 
        convert the format to a matrix where each row is a subject,
        and each column is a problem_type.
    """
    import pingouin as pg
    from scipy import stats
    
    results_path = 'decoding_results'
    decoding_accuracy_collector = np.load(
        f'{results_path}/decoding_accuracy_{num_runs}runs_{roi}.npy', 
        allow_pickle=True).ravel()[0]
    
    group_results_by_subject = np.ones((num_subs, len(problem_types)))
    for z in range(len(problem_types)):
        problem_type = problem_types[z]
        # [sub02_acc, sub03_acc, ...]
        per_type_all_subjects = decoding_accuracy_collector[problem_type]
        for s in range(num_subs):
            group_results_by_subject[s, z] = per_type_all_subjects[s]
    
    all_coefs = []
    for s in range(num_subs):
        X_sub = problem_types
        # [sub02_type1_acc, sub02_type2_acc, ...]
        y_sub = group_results_by_subject[s, :]
        coef = pg.linear_regression(X=X_sub, y=y_sub, coef_only=True)
        # print(f'sub{subs[s]}, {y_sub}, coef={coef[-1]:.3f}')
        all_coefs.append(coef[-1])
    
    print(f'average coef={np.mean(all_coefs):.3f}')
    t, p = stats.ttest_1samp(all_coefs, popmean=0)
    print(f't={t:.3f}, one-tailed p={p/2:.3f}')
        
        
if __name__ == '__main__':
    root_path = '/home/ken/projects/brain_data'
    glm_path = 'glm_trial-estimate'
    roi = 'LOC'
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

    decoding_accuracy_execute(
        roi=roi, conditions=conditions, 
        num_repetitions_per_run=num_repetitions_per_run,
        num_runs=len(runs),
        num_processes=72
    )
    
    decoding_accuracy_regression(
        roi=roi, 
        num_runs=len(runs), 
        num_subs=num_subs, 
        problem_types=problem_types
    )
    
