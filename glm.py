import os
import json
import numpy as np
import pandas as pd
from scipy.io import loadmat
import matplotlib.pyplot as plt
from IPython.display import Image

from nipype import Node, Workflow
from nipype.interfaces.matlab import MatlabCommand
from nipype.interfaces.base import Bunch
from nipype.algorithms.modelgen import SpecifySPMModel
from nipype.interfaces.spm import Level1Design
from nipype.interfaces.spm import EstimateModel
from nipype.interfaces.spm import EstimateContrast
from nipype.interfaces.spm import Normalize12
from nipype import SelectFiles
from nipype.algorithms.misc import Gunzip
from nipype.interfaces.io import DataSink

import nibabel as nb
from nilearn.plotting import plot_anat
from nilearn.plotting import plot_glass_brain

import utils

"""
Code for fitting GLM on preprocessed fMRI BOLD data
from `brain_data/Mack-Data/derivatives`

ref: https://miykael.github.io/nipype_tutorial/notebooks/handson_analysis.html
"""

def GLM(sub, task, run, n_procs):
    """
    Run GLM of a given (sub, task, run)
    """
    # TR of functional images
    TR = 2.0

    # Specify which SPM to use
    MatlabCommand.set_default_paths('/opt/spm12-r7771/spm12_mcr/spm12')
    analysis1st = Workflow(name='work_1st', base_dir=base_dir)

    # Create/load events and motion correction files.
    events_file = f'{root_path}/{base_dir}/events/sub-{sub}_task-{task}_run-{run}_events.tsv'
    mc_params_file = f'{root_path}/{base_dir}/mc_params/sub-{sub}_task-{task}_run-{run}_mc_params.tsv'
    if not os.path.exists(events_file):
        utils.prepare_events_table(sub, task, run, save_dir=base_dir)
        utils.prepare_motion_correction_params(sub, task, run, save_dir=base_dir)
        
    trialinfo = pd.read_table(events_file, dtype={'stimulus': str})
    trialinfo.head()
    conditions = []
    onsets = []
    durations = []
    
    for group in trialinfo.groupby('stimulus'):
        conditions.append(group[0])
        onsets.append(list(group[1].onset))
        durations.append(group[1].duration.tolist())
        
    subject_info = [
        Bunch(
            conditions=conditions,
            onsets=onsets,
            durations=durations,
        )
    ]

    # Initiate the SpecifySPMModel node here
    modelspec = Node(
        SpecifySPMModel(
            concatenate_runs=False,
            input_units='secs',
            output_units='secs',
            time_repetition=TR,
            high_pass_filter_cutoff=128,
            subject_info=subject_info
        ),
        name="modelspec"
    )

    # Condition names
    stimuli = ['000', '001', '010', '011', '100', '101', '110', '111']
    num_repetitions = 4
    feedback_or_not = 2
    
    condition_names = []
    for stimulus in stimuli:
        for rp in range(1, num_repetitions+1):
            for i in range(feedback_or_not):
                if i == 0:
                    condition_names.append(f'{stimulus}_rp{rp}')
                else:
                    condition_names.append(f'{stimulus}_rp{rp}_fb')
    
    num_conditions = len(condition_names)      
    # print(len(condition_names))   # 64

    # Contrasts
    contrast_list = []
    # step=2 to skip contrast for feedback onset
    for i in range(0, num_conditions, 2):
        mask = np.zeros(num_conditions, dtype=int)
        mask[i] = 1
        cont_i = [condition_names[i], 'T', condition_names, list(mask)]
        contrast_list.append(cont_i)
    
    # print(contrast_list[1])
    
    # Initiate the Level1Design node here
    level1design = Node(
        Level1Design(
            bases={'hrf': {'derivs': [0, 0]}},
            timing_units='secs',
            interscan_interval=TR,
            model_serial_correlations='FAST'
        ),
        name="level1design"
    )
    
    # Now that we have the Model Specification and 1st-Level Design node, we can connect them to each other:
    # Connect the two nodes here
    analysis1st.connect(
        [
            (modelspec, level1design, 
                [
                    ('session_info', 'session_info')
                ]
            )
        ]
    )
    
    # Now we need to estimate the model. I recommend that you'll use a Classical: 1 method to estimate the model.
    # Initiate the EstimateModel node here
    level1estimate = Node(
        EstimateModel(
            estimation_method={'Classical': 1}
        ),
        name="level1estimate"
    )

    # Connect the two nodes here
    analysis1st.connect(
        [
            (level1design, level1estimate, 
                [
                    ('spm_mat_file', 'spm_mat_file')
                ]
            )
        ]
    )

    # Initiate the EstimateContrast node here
    level1conest = Node(
        EstimateContrast(contrasts=contrast_list),
        name="level1conest"
    )

    # Connect the two nodes here
    analysis1st.connect(
        [
            (level1estimate, level1conest, 
                [
                    ('spm_mat_file','spm_mat_file'), 
                    ('beta_images', 'beta_images'),
                    ('residual_image', 'residual_image')
                ]
            )
        ]
    )

    # String template with {}-based strings
    templates = {
        'anat': '/home/ken/projects/brain_data/Mack-Data/derivatives/' \
                'sub-{sub}/anat/sub-{sub}_desc-preproc_T1w.nii.gz',
        'func': '/home/ken/projects/brain_data/Mack-Data/derivatives/' \
                'sub-{sub}/func/sub-{sub}_task-{task}_run-{run}_space-T1w_desc-preproc_bold.nii.gz',
        'mc_param': '/home/ken/projects/brain_data/glm/mc_params/sub-{sub}_task-{task}_run-{run}_mc_params.tsv',
    }    
    
    # Create SelectFiles node
    sf = Node(
        SelectFiles(
            templates, 
            sort_filelist=True
        ),
        name='selectfiles'
    )

    # list of subject identifiers
    subject_list = [f'{sub}']
    task_list = [f'{task}']
    run_list = [f'{run}']
    sf.iterables = [
        ('sub', subject_list), 
        ('task', task_list), 
        ('run', run_list)
    ]

    # Initiate the two Gunzip node here
    gunzip_anat = Node(Gunzip(), name='gunzip_anat')
    gunzip_func = Node(Gunzip(), name='gunzip_func')

    # Connect SelectFiles node to the other nodes here
    analysis1st.connect(
        [
            (sf, gunzip_anat, [('anat', 'in_file')]),
            (sf, gunzip_func, [('func', 'in_file')]),
            (gunzip_func, modelspec, [('out_file', 'functional_runs')]),
            (sf, modelspec, [('mc_param', 'realignment_parameters')])
        ]
    )

    # Initiate DataSink node here
    # Initiate the datasink node
    output_folder = 'datasink'
    datasink = Node(
        DataSink(
            base_directory=base_dir,
            container=output_folder
        ),
        name="datasink"
    )
    ## Use the following substitutions for the DataSink output
    substitutions = [('_run_', 'output_run_')]
    datasink.inputs.substitutions = substitutions

    analysis1st.connect(
        [
            (level1estimate, datasink,
                [
                    ('beta_images', 'model.@beta'),
                ]
            ),
            (level1conest, datasink, 
                [
                    ('spm_mat_file', '1stLevel.@spm_mat'),
                    ('spmT_images', '1stLevel.@T'),
                    ('spmF_images', '1stLevel.@F'),
                ]
            ),
        ]
    )

    # Create 1st-level analysis output graph
    analysis1st.write_graph(graph2use='colored', format='png', simple_form=True)

    # Visualize the graph
    Image(filename=f'{base_dir}/work_1st/graph.png')
    analysis1st.run('MultiProc', plugin_args={'n_procs': n_procs})


def visualize_glm(sub, task, run, dataType, condition, plot, threshold):
    output_dir = f'output_run_{run}_sub_{sub}_task_{task}'
    
    if plot == 'dmtx':
        # Visualize results
        out_path = f'{root_path}/{base_dir}/work_1st/{output_dir}/datasink/' \
                f'{base_dir}/datasink/1stLevel/{output_dir}'
        # Using scipy's loadmat function we can access SPM.mat
        spmmat = loadmat(
            f'{out_path}/SPM.mat',
            struct_as_record=False
        )
        # designMatrix -> (194, 9)
        designMatrix = spmmat['SPM'][0][0].xX[0][0].X    
        names = [i[0] for i in spmmat['SPM'][0][0].xX[0][0].name[0]]
        normed_design = designMatrix / np.abs(designMatrix).max(axis=0)
        fig, ax = plt.subplots(figsize=(8, 8))
        plt.imshow(normed_design, aspect='auto', cmap='gray', interpolation='none')
        ax.set_ylabel('Volume id')
        ax.set_xticks(np.arange(len(names)))
        ax.set_xticklabels(names, rotation=90)
        plt.tight_layout()
        plt.savefig('dmtx.png')
        plt.close()

    elif plot == 'anat':
        fig, ax = plt.subplots()
        plot_anat(
            f'{root_path}/Mack-Data/dropbox/sub-{sub}/anat/sub-{sub}_T1w.nii.gz',
            axes=ax)
        plt.savefig(f'anat_sub-{sub}.png')
    
    elif plot == 'contrast':
        from nilearn.image import mean_img

        print('sub = ', sub)

        across_runs = []
        for run in runs:        
            output_dir = f'output_run_{run}_sub_{sub}_task_{task}'
            
            if dataType == 'spmT':
                img = nb.load(
                    f'{root_path}/{base_dir}/work_1st/{output_dir}/datasink/' \
                    f'{base_dir}/datasink/1stLevel/{output_dir}/{dataType}_{condition}.nii')
            elif dataType == 'beta':
                img = nb.load(
                    f'{root_path}/{base_dir}/work_1st/{output_dir}/datasink/' \
                    f'{base_dir}/datasink/model/{output_dir}/{dataType}_{condition}.nii')
            
            data = np.array(img.dataobj)
            across_runs.append(data)
            print(data.shape)
        
        across_runs = np.array(across_runs)
        average_runs = np.mean(across_runs, axis=0)
        # print(across_runs.shape)  # (run, 82, 109, 87)
        # print(average_runs.shape) # (1,   82, 109, 87)
        
        # load the data
        ni_img = nb.Nifti1Image(average_runs, img.affine)           
        fig, ax = plt.subplots()
        if len(runs) == 1:
            title = f'{dataType}_{condition}-run{run}'
        else:
            title = f'{dataType}_{condition}-average{runs}'
        plot_glass_brain(
            stat_map_img=ni_img,
            colorbar=True, 
            display_mode='lyrz', 
            black_bg=True, 
            threshold=threshold,
            title=title,
            axes=ax)
        plt.savefig(f'{title}.png')
        
    plt.close()


def execute(subs, tasks, runs, n_procs):
    """
    Run GLM through all combo
    """
    for sub in subs:
        for task in tasks:
            for run in runs:
                GLM(sub, task, run, n_procs)


if __name__ == '__main__':
    root_path = '/home/ken/projects/brain_data'
    base_dir = 'glm_trial-estimate'
    num_subs = 1
    subs = [f'{i:02d}' for i in range(2, num_subs+2)]
    tasks = [1]
    runs = [1]
    n_procs = 2
    print(f'subs={subs}')
    print(f'tasks={tasks}')
    print(f'runs={runs}')
    print(f'n_procs={n_procs}')
    execute(subs, tasks, runs, n_procs)
    
    # sub = '02'
    # task = '1'
    # run = '1'
    # dataType = 'beta'
    # condition = '0002'
    # plot = 'dmtx'
    # threshold = 10
    # visualize_glm(
    #     sub=sub, task=task, run=run,
    #     dataType=dataType, condition=condition, 
    #     plot=plot, threshold=threshold
    # )