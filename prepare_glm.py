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
                    
    
def prepare_events_table(sub, task, run):
    """
    Given a subject, task and run, produce a 
    table which contains:
    
    onset   duration    weight  stimulus (i.e. trial_type)
    -----   --------    ------  --------
    
    - Both `onset` and `duration` are from 'trialtiming/'
    - weight is 1
    - trial_type is from 'behaviour/'
    """
    trialtiming_path = f'Mack-Data/trialtiming/{sub}_study{task}_run{run}.txt'
    behaviour_path = f'Mack-Data/behaviour/subject_{sub}/{sub}_study{task}_run{run}.txt'
    trialtiming = pd.read_csv(trialtiming_path, header=None).to_numpy()
    behaviour = pd.read_csv(behaviour_path, header=None).to_numpy()

    onsets = ['onset']
    durations = ['duration']
    weights = ['weight']
    stimuli = ['stimulus']
        
    for i in range(len(trialtiming)):
        trialtiming_i = trialtiming[i][0].split('\t')
        behaviour_i = behaviour[i][0].split('\t')
        
        # stimulus events
        stimulus_onset = int(trialtiming_i[4])
        stimulus_duration = 3.5
        stimulus = ''.join(behaviour_i[3:6])   # convert ['1', '0', '1'] to '101'
        onsets.append(stimulus_onset)
        durations.append(stimulus_duration)
        stimuli.append(stimulus)
        
        # feedback events
        feedback_onset = int(trialtiming_i[5])
        feedback_duration = 2.0
        stimulus_fb = f'{stimulus}_fb'
        onsets.append(feedback_onset)
        durations.append(feedback_duration)
        stimuli.append(stimulus_fb)

    onsets = np.array(onsets)
    durations = np.array(durations)
    stimuli = np.array(stimuli)
    df = np.vstack((onsets, durations, stimuli)).T
    pd.DataFrame(df).to_csv(
        f"sub-{sub}_task-{task}_run-{run}_events.tsv", 
        sep='\t', index=False, header=False
    )
    print(f'[Check] Saved tsv.')

   
def prepare_motion_correction_params(sub, task, run):
    """
    For fitting GLM, we will need to extract
    mc params from the preprocessed data.
    """
    confounds_path = f'Mack-Data/derivatives/sub-{sub}/func/' \
        f'sub-{sub}_task-{task}_run-{run}_desc-confounds_timeseries.tsv'
    mc_params = pd.read_table(
        confounds_path, 
        usecols=[
            'trans_x', 'trans_y', 'trans_z', 
            'rot_x', 'rot_y', 'rot_z'
        ]
    )

    pd.DataFrame(mc_params).to_csv(
        f"sub-{sub}_task-{task}_run-{run}_mc_params.tsv", 
        sep='\t', index=False, header=False
    )
    print(f'[Check] Saved mc_params.')

         
if __name__ == '__main__':
    sub='02'
    task='1'
    run='1'
    prepare_events_table(sub=sub, task=task, run=run)
    # prepare_motion_correction_params(sub=sub, task=task, run=run)