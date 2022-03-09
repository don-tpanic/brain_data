import json
import pandas as pd

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

from IPython.display import Image


def run_glm(sub, task, run):
    
    # TR of functional images
    with open('Mack-Data/dropbox/sub-02/func/sub-02_task-1_run-01_bold.json', 'rt') as fp:
        task_info = json.load(fp)
    TR = task_info['RepetitionTime']

    # Specify which SPM to use
    MatlabCommand.set_default_paths('/opt/spm12-r7771/spm12_mcr/spm12')
    analysis1st = Workflow(name='work_1st', base_dir='output_try2')

    trialinfo = pd.read_table(
        f'/home/ken/projects/brain_data/{sub}_study{task}_run{run}.tsv', 
        dtype={'stimulus': str}
    )
    
    trialinfo.head()
    conditions = []
    onsets = []
    durations = []
    
    for group in trialinfo.groupby('stimulus'):
        conditions.append(group[0])
        onsets.append(list(group[1].onset))
        durations.append(group[1].duration.tolist())
        
    subject_info = [Bunch(conditions=conditions,
                            onsets=onsets,
                            durations=durations,
                            #amplitudes=None,
                            #tmod=None,
                            #pmod=None,
                            #regressor_names=None,
                            #regressors=None
                            )]

    # Initiate the SpecifySPMModel node here
    modelspec = Node(SpecifySPMModel(concatenate_runs=False,
                                     input_units='secs',
                                     output_units='secs',
                                     time_repetition=TR,
                                     high_pass_filter_cutoff=128,
                                     subject_info=subject_info),
                     name="modelspec")

    # Condition names
    condition_names = ['000', '001', '010', '011', '100', '101', '110', '111']
    # Contrasts
    cont01 = ['000',         'T', condition_names, [1, 0, 0, 0, 0, 0, 0, 0]]
    cont02 = ['001',         'T', condition_names, [0, 1, 0, 0, 0, 0, 0, 0]]
    cont03 = ['010',         'T', condition_names, [0, 0, 1, 0, 0, 0, 0, 0]]
    cont04 = ['011',         'T', condition_names, [0, 0, 0, 1, 0, 0, 0, 0]]
    cont05 = ['100',         'T', condition_names, [0, 0, 0, 0, 1, 0, 0, 0]]
    cont06 = ['101',         'T', condition_names, [0, 0, 0, 0, 0, 1, 0, 0]]
    cont07 = ['110',         'T', condition_names, [0, 0, 0, 0, 0, 0, 1, 0]]
    cont08 = ['111',         'T', condition_names, [0, 0, 0, 0, 0, 0, 0, 1]]
    # contrast_list = [cont01, cont02, cont03, cont04, cont05, cont06, cont07, cont08]
    contrast_list = [cont01]

    # Initiate the Level1Design node here
    level1design = Node(Level1Design(bases={'hrf': {'derivs': [0, 0]}},
                                     timing_units='secs',
                                     interscan_interval=TR,
                                     model_serial_correlations='FAST'),
                        name="level1design")
    # Now that we have the Model Specification and 1st-Level Design node, we can connect them to each other:

    # Connect the two nodes here
    analysis1st.connect([(modelspec, level1design, [('session_info',
                                                     'session_info')])])
    
    # Now we need to estimate the model. I recommend that you'll use a Classical: 1 method to estimate the model.
    # Initiate the EstimateModel node here
    level1estimate = Node(EstimateModel(estimation_method={'Classical': 1}),
                          name="level1estimate")

    # Connect the two nodes here
    analysis1st.connect([(level1design, level1estimate, [('spm_mat_file',
                                                          'spm_mat_file')])])

    # Initiate the EstimateContrast node here
    level1conest = Node(EstimateContrast(contrasts=contrast_list),
                        name="level1conest")

    # Connect the two nodes here
    analysis1st.connect([(level1estimate, level1conest, [('spm_mat_file',
                                                          'spm_mat_file'),
                                                         ('beta_images',
                                                          'beta_images'),
                                                         ('residual_image',
                                                          'residual_image')])])

    # Location of the template
    template = '/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii'
    # Initiate the Normalize12 node here
    normalize = Node(Normalize12(jobtype='estwrite',
                                 tpm=template,
                                 write_voxel_sizes=[4, 4, 4]
                                ),
                     name="normalize")
    # Now we can connect the estimated contrasts to normalization node.

    # Connect the nodes here
    analysis1st.connect([(level1conest, normalize, [('con_images',
                                                     'apply_to_files')])
                         ])

    # String template with {}-based strings
    templates = {'anat': '/home/ken/projects/brain_data/Mack-Data/dropbox/sub-{sub}/anat/sub-{sub}_T1w.nii.gz',
                 'func': '/home/ken/projects/brain_data/Mack-Data/derivatives/sub-{sub}/func/sub-{sub}_task-{task}_run-{run}_space-T1w_desc-preproc_bold.nii.gz',
                #  'mc_param': '/output/datasink_handson/preproc/sub-{subj_id}.par',
                #  'outliers': '/output/datasink_handson/preproc/art.sub-{subj_id}_outliers.txt'
                }

    # Create SelectFiles node
    sf = Node(SelectFiles(templates, sort_filelist=True),
              name='selectfiles')

    # list of subject identifiers
    subject_list = ['02']
    task_list = ['1']
    run_list = ['1']
    sf.iterables = [
        ('sub', subject_list), 
        ('task', task_list), 
        ('run', run_list)
    ]

    # Initiate the two Gunzip node here
    gunzip_anat = Node(Gunzip(), name='gunzip_anat')
    gunzip_func = Node(Gunzip(), name='gunzip_func')

    # Connect SelectFiles node to the other nodes here
    analysis1st.connect([(sf, gunzip_anat, [('anat', 'in_file')]),
                         (sf, gunzip_func, [('func', 'in_file')]),
                         (gunzip_anat, normalize, [('out_file', 'image_to_align')]),
                         (gunzip_func, modelspec, [('out_file', 'functional_runs')]),
                        #  (sf, modelspec, [('mc_param', 'realignment_parameters'),
                        #                   ('outliers', 'outlier_files'),
                        #                   ])
                        ])

    # Initiate DataSink node here
    # Initiate the datasink node
    output_folder = 'datasink_handson'
    datasink = Node(DataSink(base_directory='output_try2',
                             container=output_folder),
                    name="datasink")
    ## Use the following substitutions for the DataSink output
    substitutions = [('_sub_', 'sub-')]
    datasink.inputs.substitutions = substitutions

    analysis1st.connect([(level1conest, datasink, [('spm_mat_file', '1stLevel.@spm_mat'),
                                                   ('spmT_images', '1stLevel.@T'),
                                                   ('spmF_images', '1stLevel.@F'),
                                                  ]),
                         (normalize, datasink, [('normalized_files', 'normalized.@files'),
                                                ('normalized_image', 'normalized.@image'),
                                               ]),
                        ])

    # Create 1st-level analysis output graph
    analysis1st.write_graph(graph2use='colored', format='png', simple_form=True)

    # Visualize the graph
    Image(filename='output_try2/work_1st/graph.png')
    analysis1st.run('MultiProc', plugin_args={'n_procs': 8})


def visualize_glm(sub):
    # Visualize results
    out_path = f'/home/ken/projects/brain_data/output_try2/work_1st/sub-{sub}/datasink/output_try2/datasink_handson/1stLevel/sub-{sub}/'

    import numpy as np
    from matplotlib import pyplot as plt
    from scipy.io import loadmat

    # Using scipy's loadmat function we can access SPM.mat
    spmmat = loadmat(f'{out_path}/SPM.mat',
                    struct_as_record=False)

    designMatrix = spmmat['SPM'][0][0].xX[0][0].X
    names = [i[0] for i in spmmat['SPM'][0][0].xX[0][0].name[0]]
    normed_design = designMatrix / np.abs(designMatrix).max(axis=0)
    fig, ax = plt.subplots(figsize=(8, 8))
    plt.imshow(normed_design, aspect='auto', cmap='gray', interpolation='none')
    ax.set_ylabel('Volume id')
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=90)
    plt.savefig('dmtx.png')
    plt.close()

    # Visualize
    import nibabel as nb
    from nilearn.plotting import plot_anat
    from nilearn.plotting import plot_glass_brain
    # from IPython.display import Image

    fig, ax = plt.subplots()
    
    # Load GM probability map of TPM.nii
    img = nb.load('/opt/spm12-r7771/spm12_mcr/spm12/tpm/TPM.nii')
    GM_template = nb.Nifti1Image(
        img.get_data()[..., 0], img.affine, img.header
    )

    # Plot normalized subject anatomy
    display = plot_anat(
        f'/home/ken/projects/brain_data/output_try2/work_1st/sub-{sub}/datasink/output_try2/datasink_handson/normalized/sub-{sub}/wsub-{sub}_T1w.nii',
        axes=ax
    )

    # Overlay in edges GM map
    display.add_edges(GM_template)
    plt.savefig('brain_anat.png')
    plt.close()
    
    
    fig, ax = plt.subplots()
    plot_glass_brain(
        f'/home/ken/projects/brain_data/output_try2/work_1st/sub-{sub}/datasink/output_try2/datasink_handson/normalized/sub-{sub}/wcon_0001.nii',
        colorbar=True, 
        display_mode='lyrz', 
        black_bg=True, 
        threshold=15,
        title=f'subject {sub} - F-contrast: Activation',
        axes=ax
    )
    
    plt.savefig('brain.png')
    

if __name__ == '__main__':
    sub = '02'
    task = '1'
    run = '1'
    run_glm(sub=sub, task=task, run=run)
    # visualize_glm(sub='02')