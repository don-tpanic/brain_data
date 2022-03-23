import os
import numpy as np
import pandas as pd
from collections import defaultdict


class Mappings(object):
    def __init__(self, num_subs=23):
        # fixed mapping between binary coding and physical dims
        # it's fixed because DCNN finetuned is unique.
        dcnn_mapping = {
            0: {'0': 'thin leg', '1': 'thick leg'},
            1: {'0': 'thick antenna', '1': 'thin antenna'},
            2: {'0': 'pincer mandible', '1': 'shovel mandible'}
        }

        # '12' and '21' are coding scheme
        # 12: is the original coding scheme that is 
        # the same as the provided stimulus set coding.
        # 21: is the flipped coding scheme that is 
        # the opposite as the provided stimulus set coding.
        coding_scheme = {
            12: {'0': '0','1': '1'},
            21: {'0': '1','1': '0'}
        }

        behaviour_columns = {
            0: 'trial',
            1: 'task',
            2: 'run',
            3: 'dim1',
            4: 'dim2',
            5: 'dim3',
            6: 'answer',
            7: 'response',
            8: 'RT',
            9: 'accuracy'
        }

        trial_columns = {
            0: 'subject',
            1: 'trial',
            2: 'task',
            3: 'run',
            4: 'stimulus onset',
            5: 'feedback onset'
        }
        
        sub2assignment_n_scheme = {
            '02': [2,1,3,12,12,12],
            '03': [3,1,2,12,12,12],
            '04': [1,2,3,21,21,12],
            '05': [3,2,1,12,12,21],
            '06': [3,1,2,21,12,21],
            '07': [3,1,2,12,21,12],
            '08': [3,2,1,21,12,12],
            '09': [1,2,3,12,21,21],
            '10': [2,3,1,12,12,12],
            '11': [1,3,2,21,12,21],
            '12': [3,2,1,12,12,21],
            '13': [1,2,3,21,21,21],
            '14': [2,3,1,12,12,21],
            '15': [1,2,3,12,12,21],
            '16': [2,3,1,12,21,21],
            '17': [3,1,2,12,12,21],
            '18': [2,1,3,21,21,12],
            '19': [2,1,3,21,12,12],
            '20': [3,1,2,12,12,12],
            '21': [1,3,2,21,21,12],
            '22': [1,3,2,21,12,21],
            '23': [2,3,1,12,21,12],
            '24': [2,1,3,12,21,21],
        }
        
        self.dcnn_mapping = dcnn_mapping
        self.coding_scheme = coding_scheme
        self.behaviour_columns = behaviour_columns
        self.trial_columns = trial_columns
        self.sub2assignment_n_scheme = sub2assignment_n_scheme


def convert_dcnnCoding_to_subjectCoding(sub):
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
            
    e.g. if DCNN=101, 
            with assignment 213 and scheme 12, 12, 12
            101 -> 011 -> 011
            
            with assignment 312 and scheme 12, 21, 12
            101 -> 110 -> 100
            
    return:
    -------
        Given all DCNN stimuli, return the conversion ordering for a given subject.
        E.g. for subject a, the orders of the stimuli should be [6, 1, 7, 5, 4, 2, 3, 0]
        where 6 corresponds to 000 in DCNN coding but 110 in subject coding.
    """
    sub2assignment_n_scheme = Mappings().sub2assignment_n_scheme
    coding_scheme = Mappings().coding_scheme
    
    conversion_ordering = []
    stimulus2order_mapping = {
        '000': 0, '001': 1, '010': 2, '011': 3,
        '100': 4, '101': 5, '110': 6, '111': 7,
    }
    for dcnn_stimulus in ['000', '001', '010', '011', '100', '101', '110', '111']:
        sub_stimulus = [i for i in dcnn_stimulus]
        print(f'\n\n--------------------------------')
        print(f'[Check] DCNN stimulus {sub_stimulus}')
        
        # assignment (flip three dims)
        assignment_n_scheme = sub2assignment_n_scheme[sub]
        new_stimulus_0 = sub_stimulus[assignment_n_scheme[0]-1]
        new_stimulus_1 = sub_stimulus[assignment_n_scheme[1]-1]
        new_stimulus_2 = sub_stimulus[assignment_n_scheme[2]-1]
        sub_stimulus[0] = new_stimulus_0
        sub_stimulus[1] = new_stimulus_1
        sub_stimulus[2] = new_stimulus_2
        # print(f'[Check] sub{sub}, assignment stimulus {sub_stimulus}')
        
        # scheme (flip binary codings)
        dim1_scheme = assignment_n_scheme[3]
        dim2_scheme = assignment_n_scheme[4]
        dim3_scheme = assignment_n_scheme[5]
        sub_stimulus[0] = coding_scheme[dim1_scheme][sub_stimulus[0]]
        sub_stimulus[1] = coding_scheme[dim2_scheme][sub_stimulus[1]]
        sub_stimulus[2] = coding_scheme[dim3_scheme][sub_stimulus[2]]
        # print(f'[Check] sub{sub}, scheme stimulus {sub_stimulus}')
        
        conversion_ordering.append(
            stimulus2order_mapping[
                ''.join(sub_stimulus)
            ]
        )
        
    return np.array(conversion_ordering)
                    
                    
def reorder_RDM_entries_into_chunks():
    """
    For each subject, the groud true label for each stimulus coding is different.
    When visualising RDMs of conditions, we want to make sure that
    rows and columns are grouped (in chunk) by their labels. This asks
    for a mapping from each subject's stimulus coding to their labels in each
    task.
    
    Impl:
    -----
        We create a dictionary like:
        
        mapping = {sub1: 
                    {task1: [orderOf(000), orderOf(001), ...], 
                     task2: [orderOf(000), orderOf(001), ...], 
                    ...
        
        Notice, within each task (across runs), the labels are the same so we can
        simply use run1.
        
        Notice, there is one extra conversion inside to get the orderOf(..). 
        That is, after we get a list of labels correspond to 000, 001, ..., 111, which look 
        like [1, 2, 2, 1, 2, ...], we argsort them to get indices which we will return as 
        `conversion_ordering` that is going to reorder the RDM entries into desired chunks.
    
    return:
    -------
        `mapping` explained above
    """
    mapping = defaultdict(lambda: defaultdict(list))
    num_subs = 23
    tasks = [1, 2, 3]
    subs = [f'{i:02d}' for i in range(2, num_subs+1)]
    stimuli = ['000', '001', '010', '011', '100', '101', '110', '111']
    behaviour_path = f'Mack-Data/behaviour'
    for sub in subs:
        for task in tasks:
            
            behaviour = pd.read_csv(
                f'{behaviour_path}/subject_{sub}/{sub}_study{task}_run1.txt', 
                header=None
            ).to_numpy()
            
            i = 0
            temp_mapping = dict()
            # search exactly all stimuli once and stop.
            while len(temp_mapping.keys()) != len(stimuli):
                behaviour_i = behaviour[i][0].split('\t')
                stimulus = ''.join(behaviour_i[3:6])
                # label = behaviour_i[7]  # 7 - true response (some missing)
                label = behaviour_i[6]  # 6 - subj answer (later correct?)
                # print(f'stimulus = {stimulus}, label = {label}')
                temp_mapping[stimulus] = int(label)
                i += 1
            
            # print(temp_mapping.keys())
            # print(temp_mapping.values())
            
            # this is to reorder the stimuli as 000, 001, ..., 111
            # so the corresponding list of labels match the order.
            labels = []
            for stimulus in stimuli:
                labels.append(temp_mapping[stimulus])
                
            # sort the labels and get indices in asc order
            grouped_labels_indices = np.argsort(labels)
            mapping[sub][task].extend(grouped_labels_indices)

    # mapping[sub][task] = a list of indices that will be used to sort the RDM entries.
    return mapping
    
                             
def prepare_events_table(sub, task, run, save_dir):
    """
    Given a subject, task and run, produce a 
    table which contains:
    
    onset   duration    weight  stimulus (i.e. trial_type)
    -----   --------    ------  --------
    
    - Both `onset` and `duration` are from 'trialtiming/'
    - weight is 1
    - trial_type is from 'behaviour/'
    """
    output_dir = f'{save_dir}/events'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
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
        f"{output_dir}/sub-{sub}_task-{task}_run-{run}_events.tsv", 
        sep='\t', index=False, header=False
    )
    print(f'[Check] Saved events tsv.')

   
def prepare_motion_correction_params(sub, task, run, save_dir):
    """
    For fitting GLM, we will need to extract
    mc params from the preprocessed data.
    """
    output_dir = f'{save_dir}/mc_params'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
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
        f"{output_dir}/sub-{sub}_task-{task}_run-{run}_mc_params.tsv", 
        sep='\t', index=False, header=False
    )
    print(f'[Check] Saved mc_params.')

         
if __name__ == '__main__':
    # convert_dcnnCoding_to_subjectCoding(dcnn_stimulus='101', sub='23')
    
    mapping = reorder_RDM_entries_into_chunks()
    print(mapping['02'][1])
    print(mapping['03'][2])
    