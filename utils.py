import os 
import numpy as np
import pandas as pd


"""
Prepare design matrix that has grouped by task and stimulus.

Each column of a dmtx, is the stimulus onset info of a trial of a task
"""

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
        

if __name__ == '__main__':
    Mappings()
        


