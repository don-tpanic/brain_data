import numpy as np
import pandas as pd
from utils import Mappings


def dcnnCoding2subjectCoding(
        stimulus, sub, 
        sub2assignment_n_scheme, coding_scheme
    ):
    """
    In order to extract stimulus-specific activations (for RSA later), 
    we need to first establish the mapping between stimulus coding of 
    DCNN (which is fixed due to finetuning) and stimulus coding of 
    subjects (which differs for each subject due to random scheme).
    
    Since the ultimate purpose is to find brain activation of each stimulus
    (in terms of DCNN coding), we need to find the coding used by each subject
    of every stimulus.
    
    e.g. In terms of DCNN coding, stimulus 000 (thin leg, thick antenna, pincer mandible)
    so what is the coding for this subject? i.e. thin leg=? thick antenna=? pincer mandible=?
    For different subs, this coding is different, depending on '12' or '21' random scheme: 
            
    e.g. if incoming=101, 
            with assignment 213 and scheme 12, 12, 12
            101 -> 011 -> 011
            
            with assignment 312 and scheme 12, 21, 12
            101 -> 110 -> 100
    """
    sub_stimulus = [i for i in stimulus]
    print(f'\n\n--------------------------------')
    print(f'[Check] DCNN stimulus {sub_stimulus}')
    
    # assignment
    assignment_n_scheme = sub2assignment_n_scheme[sub]
    new_stimulus_0 = sub_stimulus[assignment_n_scheme[0]-1]
    new_stimulus_1 = sub_stimulus[assignment_n_scheme[1]-1]
    new_stimulus_2 = sub_stimulus[assignment_n_scheme[2]-1]
    sub_stimulus[0] = new_stimulus_0
    sub_stimulus[1] = new_stimulus_1
    sub_stimulus[2] = new_stimulus_2
    # print(f'[Check] sub{sub}, assignment stimulus {sub_stimulus}')
    
    # scheme
    dim1_scheme = assignment_n_scheme[3]
    dim2_scheme = assignment_n_scheme[4]
    dim3_scheme = assignment_n_scheme[5]
    sub_stimulus[0] = coding_scheme[dim1_scheme][sub_stimulus[0]]
    sub_stimulus[1] = coding_scheme[dim2_scheme][sub_stimulus[1]]
    sub_stimulus[2] = coding_scheme[dim3_scheme][sub_stimulus[2]]
    print(f'[Check] sub{sub}, scheme stimulus {sub_stimulus}')
    return sub_stimulus


def group_trials_by_stimulus(study, run, sub, trials_to_group, sub_stimulus):
    """
    For each subject, of a given task, we need to 
    group the trials into columns of the same stimulus.
    This is for us later to construct a design matrix for GLM.
    
    Given a study and a run (behaviour/), we locate the trials 
    where the target stimulus is presented to the subject.
    
    We can then according to trials, extract stimulus onset info
    from (trialtiming/) which will then be convolved by HDF
    """
    fpath = f'Mack-Data/behaviour/subject_{sub}/{sub}_study{study}_run{run}.txt'
    print(f'[Check] fpath = {fpath}')
    
    trials = pd.read_csv(fpath, header=None).to_numpy()
    
    for trial in trials:
        trial = trial[0].split('\t')
        
        if trial[3:6] == sub_stimulus:
            trials_to_group.append(trial[0])
    
    print(f'[Check] trials_to_group = {trials_to_group}')
    return trials_to_group


def group_stimulus_onset_by_trials(study, run, sub, stimulus_onset, trials_to_group):
    """
    After `group_trials_by_stimulus`, we have all trials of a run where 
    the target stimulus is presented. Given these trials, we can look 
    for their corresponding stimulus onset from 'trialtiming/'
    """
    fpath = f'Mack-Data/trialtiming/{sub}_study{study}_run{run}.txt'
    trials = pd.read_csv(fpath, header=None).to_numpy()
    
    for trial in trials:
        trial = trial[0].split('\t')
        
        if trial[1] in trials_to_group:
            stimulus_onset.append(trial[4])
    
    print(f'[Check] stimulus_onset = {stimulus_onset}')
    return stimulus_onset


def create_design_mtx(stimulus_onset):
    """
    Given stimulus onset info of a run & sub & study, create
    one-hot columns into design matrix before convolved by 
    HRF.
    """
    # TODO: column length 194s * 2?
    regressors = np.zeros((194 * 2, len(stimulus_onset))) 
    for i, time in enumerate(stimulus_onset):
        print(time, i)
        regressors[int(time), i] = 1
    
    print(regressors[:, 2])
    print(regressors[:, 3])
    


def execute():
    # stimuli = ['000', '001', '010', '011', '100', '101', '110', '111']
    stimuli = ['101']
    sub2assignment_n_scheme = Mappings().sub2assignment_n_scheme
    coding_scheme = Mappings().coding_scheme
    subs = []
    for i in range(2, 25):
        if len(f'{i}') == 1:
            subs.append(f'0{i}')
        else:
            subs.append(f'{i}')
            
    runs = [1, 2, 3, 4]
    studies = [1, 2, 3]
    
    for stimulus in stimuli:
        for sub in subs:
            sub_stimulus = dcnnCoding2subjectCoding(
                stimulus, sub, sub2assignment_n_scheme, coding_scheme
            )
            for study in studies:
                for run in runs:
                    trials_to_group = []
                    stimulus_onset = []
                
                    # get from 'behaviour/' which has info about 
                    # trial - stimulus pair
                    trials_to_group = group_trials_by_stimulus(
                        study, run, sub, trials_to_group, sub_stimulus
                    )
                    
                    # given the trials of the stimulus, we can get 
                    # stimulus onset from 'trialtiming/'
                    stimulus_onset = group_stimulus_onset_by_trials(
                        study, run, sub, stimulus_onset, trials_to_group
                    )
                    
                    # given the stimulus onset, create one-hot columns
                    # in a design matrix.
                    create_design_mtx(stimulus_onset)
                    exit()
                
                # TODO: one design mtx is one subject and one study?
                # TODO: one design mtx is one run or all later runs?
                    
            
if __name__ == '__main__':
    execute()