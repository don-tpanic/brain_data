# Get the Node and Workflow object
from nipype import Node, Workflow

# Specify which SPM to use
from nipype.interfaces.matlab import MatlabCommand
MatlabCommand.set_default_paths('/opt/spm12-r7771/spm12_mcr/spm12')

analysis1st = Workflow(name='work_1st', base_dir='output_try2')


def subjectinfo(subject_id):
    import pandas as pd
    from nipype.interfaces.base import Bunch
    
    trialinfo = pd.read_table('/home/ken/projects/brain_data/02_study1_run1.tsv', dtype={'stimulus': str})
    
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
    return subject_info  # this output will later be returned to infosource
subject_info = subjectinfo(subject_id='02')


from nipype.algorithms.modelgen import SpecifySPMModel
# Initiate the SpecifySPMModel node here
modelspec = Node(SpecifySPMModel(concatenate_runs=False,
                                 input_units='secs',
                                 output_units='secs',
                                 time_repetition=2.5,
                                 high_pass_filter_cutoff=128,
                                 subject_info=subject_info),
                 name="modelspec")




# Condition names
condition_names = ['000', '001', '010', '011', '100', '101', '110', '111']

# Contrasts
cont01 = ['000',         'T', condition_names, [1, 0, 0, 0, 0, 0, 0, 0]]
# cont02 = ['001',         'T', condition_names, [0, 1, 0, 0, 0, 0, 0, 0]]
# cont03 = ['010',         'T', condition_names, [0, 0, 1, 0, 0, 0, 0, 0]]
# cont04 = ['011',         'T', condition_names, [0, 0, 0, 1, 0, 0, 0, 0]]
# cont05 = ['100',         'T', condition_names, [0, 0, 0, 0, 1, 0, 0, 0]]
# cont06 = ['101',         'T', condition_names, [0, 0, 0, 0, 0, 1, 0, 0]]
# cont07 = ['110',         'T', condition_names, [0, 0, 0, 0, 0, 0, 1, 0]]
# cont08 = ['111',         'T', condition_names, [0, 0, 0, 0, 0, 0, 0, 1]]
# contrast_list = [cont01, cont02, cont03, cont04, cont05, cont06, cont07, cont08]
contrast_list = [cont01]


from nipype.interfaces.spm import Level1Design
# Initiate the Level1Design node here
level1design = Node(Level1Design(bases={'hrf': {'derivs': [0, 0]}},
                                 timing_units='secs',
                                 interscan_interval=2.5,
                                 model_serial_correlations='AR(1)'),
                    name="level1design")
# Now that we have the Model Specification and 1st-Level Design node, we can connect them to each other:

# Connect the two nodes here
analysis1st.connect([(modelspec, level1design, [('session_info',
                                                 'session_info')])])
# Now we need to estimate the model. I recommend that you'll use a Classical: 1 method to estimate the model.
from nipype.interfaces.spm import EstimateModel
# Initiate the EstimateModel node here
level1estimate = Node(EstimateModel(estimation_method={'Classical': 1}),
                      name="level1estimate")

# Connect the two nodes here
analysis1st.connect([(level1design, level1estimate, [('spm_mat_file',
                                                      'spm_mat_file')])])

from nipype.interfaces.spm import EstimateContrast
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

from nipype.interfaces.spm import Normalize12

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

# Import the SelectFiles
from nipype import SelectFiles

# String template with {}-based strings
templates = {'anat': '/home/ken/projects/brain_data/Mack-Data/dropbox/sub-{subj_id}/anat/sub-{subj_id}_T1w.nii.gz',
             'func': '/home/ken/projects/brain_data/Mack-Data/derivatives/sub-{subj_id}/func/sub-{subj_id}_task-1_run-1_space-T1w_desc-preproc_bold.nii.gz',
            #  'mc_param': '/output/datasink_handson/preproc/sub-{subj_id}.par',
            #  'outliers': '/output/datasink_handson/preproc/art.sub-{subj_id}_outliers.txt'
            }

# Create SelectFiles node
sf = Node(SelectFiles(templates, sort_filelist=True),
          name='selectfiles')

# list of subject identifiers
subject_list = ['02']
sf.iterables = [('subj_id', subject_list)]

from nipype.algorithms.misc import Gunzip
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

from nipype.interfaces.io import DataSink
# Initiate DataSink node here
# Initiate the datasink node
output_folder = 'datasink_handson'
datasink = Node(DataSink(base_directory='output_try2',
                         container=output_folder),
                name="datasink")
## Use the following substitutions for the DataSink output
substitutions = [('_subj_id_', 'sub-')]
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
from IPython.display import Image
Image(filename='output_try2/work_1st/graph.png')

analysis1st.run('MultiProc', plugin_args={'n_procs': 8})


from nilearn.plotting import plot_glass_brain
out_path = '/home/ken/projects/brain_data/output_try2/work_1st/sub-02/datasink/output_try2/datasink_handson/1stLevel/sub-02/'
plot_glass_brain(out_path + 'spmT_0001.nii', display_mode='lyrz',
                 black_bg=True, colorbar=True, title='average (FDR corrected)')