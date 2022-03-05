from utils import Mappings


"""
"""

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
    # e.g. stimulus 000 - thin leg, thick antenna, pincer mandible
    # ask: what is the coding for this subject?
    # i.e. thin leg=? thick antenna=? pincer mandible=?
    # for different subs, this coding is different, depending on
    # '12' or '21' scheme
    
    # if incoming=101, 
    #   with assignment 213 and scheme 12, 12 12
    #   101 -> 011 -> 011
    #   
    #   with assignment 312 and scheme 12, 21, 12
    #   101 -> 110 -> 100
    for sub in subs:
        
        stimulus_copy = [i for i in stimulus]
        print(f'\n\n--------------------------------')
        print(f'[Check] DCNN stimulus {stimulus_copy}')
        
        # assignment
        assignment_n_scheme = sub2assignment_n_scheme[sub]
        new_stimulus_0 = stimulus_copy[assignment_n_scheme[0]-1]
        new_stimulus_1 = stimulus_copy[assignment_n_scheme[1]-1]
        new_stimulus_2 = stimulus_copy[assignment_n_scheme[2]-1]
        stimulus_copy[0] = new_stimulus_0
        stimulus_copy[1] = new_stimulus_1
        stimulus_copy[2] = new_stimulus_2
        # print(f'[Check] sub{sub}, assignment stimulus {stimulus_copy}')
        
        # scheme
        dim1_scheme = assignment_n_scheme[3]
        dim2_scheme = assignment_n_scheme[4]
        dim3_scheme = assignment_n_scheme[5]
        stimulus_copy[0] = coding_scheme[dim1_scheme][stimulus_copy[0]]
        stimulus_copy[1] = coding_scheme[dim2_scheme][stimulus_copy[1]]
        stimulus_copy[2] = coding_scheme[dim3_scheme][stimulus_copy[2]]
        print(f'[Check] sub{sub}, scheme stimulus {stimulus_copy}')