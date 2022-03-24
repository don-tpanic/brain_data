import os
import scipy
import numpy as np
import scipy.stats as stats
from subprocess import call
import matplotlib.pyplot as plt
from matplotlib import colorbar, colors
from matplotlib.colors import ListedColormap
from sklearn.metrics import pairwise_distances

import nibabel as nb
import nilearn as nl
import nipype.interfaces.ants as ants
from nipype.interfaces.fsl import MultiImageMaths
from nilearn.masking import apply_mask
from nilearn import plotting, image

from utils import convert_dcnnCoding_to_subjectCoding, reorder_RDM_entries_into_chunks

"""
1. Extract ROI-level beta weights and compile into RDMs.
2. Perform RSA
"""

def run_ants_command(roi, roi_nums, smooth):
    """
    Helper function that automatically grabs files to prepare 
    for the final ants command.
    """
    maths = MultiImageMaths()
    
    if roi_nums is not None:
        all_files = []
        for roi_num in roi_nums:
            all_files.append(f'{roi_path}/perc_VTPM_vol_roi{roi_num}_lh.nii.gz')
            all_files.append(f'{roi_path}/perc_VTPM_vol_roi{roi_num}_rh.nii.gz')
        
        maths.inputs.op_string = ''
        for _ in range(len(roi_nums) * 2 - 1):
            maths.inputs.op_string += '-add %s '
            
        if smooth:
            maths.inputs.op_string += f'-s {smooth}'
        maths.inputs.op_string += ' -bin '
        
        maths.inputs.in_file = all_files[0]
        maths.inputs.operand_files = all_files[1:]
    
    # HPC
    else:
        if 'LH' in roi:
            h = 'l'
        else:
            h = 'r'
                        
        maths.inputs.in_file = f'{roi_path}/HIPP_BODY_{h}h.nii.gz'
        maths.inputs.op_string = '-add %s -add %s -add %s ' 
        if smooth:
            maths.inputs.op_string += f'-s {smooth}'
        maths.inputs.op_string += ' -bin '
        
        maths.inputs.operand_files = [
            f'{roi_path}/HIPP_BODY_{h}h.nii.gz',
            f'{roi_path}/HIPP_HEAD_{h}h.nii.gz',
            f'{roi_path}/HIPP_TAIL_{h}h.nii.gz'
        ]
        
    maths.inputs.out_file = f'{roi_path}/mask-{roi}.nii.gz'
    runCmd = '/usr/bin/fsl5.0-' + maths.cmdline
    print(f'runCmd = {runCmd}')
    call(runCmd, shell=True)
    

def merge_n_smooth_mask(roi, smooth):
    """
    First processing of the standard ROI masks is to
    merge some left&right masks and smooth them.
    """
    if 'HPC' not in roi:
        roi_number_mapping = {
            'V1': [1, 2],
            'V2': [3, 4],
            'V3': [5, 6],
            'V4': [7],
            'V1-3': [1, 2, 3, 4, 5, 6],
            'LOC': [14, 15]
        }
        roi_nums = roi_number_mapping[roi]
    
    else:
        roi_nums = None
    
    run_ants_command(roi=roi, roi_nums=roi_nums, smooth=smooth)
        
    
def transform_mask_MNI_to_T1(sub, roi):
    """
    Given a subject and a ROI, 
    transform ROI mask from MNI space to subject's T1 space.
    
    This is based on the fact that the ROI masks provided are already
    in standard MNI space.
    """
    print(f'[Check] transform roi mask to subject{sub} T1 space')
    at = ants.ApplyTransforms()
    
    reference_image_path = f'{root_path}/Mack-Data/dropbox/sub-{sub}/anat'
    transform_path = f'{root_path}/Mack-Data/derivatives/sub-{sub}/anat'
    assert os.path.exists(reference_image_path)
    assert os.path.exists(transform_path)
    
    at.inputs.dimension = 3
    at.inputs.input_image = f'{roi_path}/mask-{roi}.nii.gz'
    at.inputs.reference_image = f'{reference_image_path}/sub-{sub}_T1w.nii.gz'
    at.inputs.output_image = f'{roi_path}/mask-{roi}_T1.nii.gz'
    at.inputs.interpolation = 'NearestNeighbor'
    at.inputs.default_value = 0
    at.inputs.transforms = [f'{transform_path}/sub-{sub}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5']
    at.inputs.invert_transform_flags = [False]
    runCmd = at.cmdline
    call(runCmd, shell=True)
     

def applyMask(
        roi, sub, task, run, dataType='beta', condition='0001'):
    """
    Apply ROI mask (T1 space) to subject's whole brain beta weights.
    
    return:
    -------
        per (ROI, subject, task, run, condition) beta weights
    """
    output_path = f'output_run_{run}_sub_{sub}_task_{task}'
    
    if dataType == 'beta':
        data_path = f'{root_path}/{glm_path}/work_1st/{output_path}/datasink/' \
            f'{glm_path}/datasink/model/{output_path}/{dataType}_{condition}.nii'
    elif dataType == 'spmT':
        data_path = f'{root_path}/{glm_path}/work_1st/{output_path}/datasink/' \
            f'{glm_path}/datasink/1stLevel/{output_path}/{dataType}_{condition}.nii'
    
    imgs = nb.load(data_path)
    # print(f'[Check] beta weight file: {data_path}')
    
    maskROI = nb.load(
        f'{roi_path}/mask-{roi}_T1.nii.gz'
    )
    maskROI = nl.image.resample_img(
        maskROI, 
        target_affine=imgs.affine,
        target_shape=imgs.shape[:3], 
        interpolation='nearest'
    )
    # print('maskROI.shape = ', maskROI.shape)
    fmri_masked = apply_mask(
        imgs, maskROI, smoothing_fwhm=None
    )
    # print(fmri_masked.shape)  # per ROI & subject & condition beta weights
    return roi, maskROI, fmri_masked


def compute_RDM(embedding_mtx, sub, task, run, roi, distance):
    """
    Compute and save RDM of given beta weights and sort 
    the conditions based on specified ordering.
    
    There are two possible orderings:
    1. Based on DCNN codings, using `convert_dcnnCoding_to_subjectCoding`
    2. Based on class labels, using `reorder_RDM_entries_into_chunks`
    """
    if distance == 'euclidean':
        RDM = pairwise_distances(embedding_mtx, metric='euclidean')
        
    elif distance == 'pearson':
        RDM = pairwise_distances(embedding_mtx, metric='correlation')
    
    # NOTE(ken): subject level conditions have different physical meanings
    # due to random shuffle. Hence here we do the conversion based on
    # DCNN coding which has fixed and unique meaning.
    # TODO: when to use at all?
    # conversion_ordering = convert_dcnnCoding_to_subjectCoding(sub)
    
    # TODO: for viz, should use this mapping instead?
    conversion_ordering = reorder_RDM_entries_into_chunks()[sub][task]
    print(f'[Check] conversion_ordering={conversion_ordering}')
    
    # reorder both cols and rows based on ordering.
    RDM = RDM[conversion_ordering, :][:, conversion_ordering]
    # RDM = RDM[np.triu_indices(RDM.shape[0])]
    
    save_path = f'{rdm_path}/sub-{sub}_task-{task}_run-{run}_roi-{roi}_{distance}.npy'
    np.save(save_path, RDM)
    print(f'[Check] RDM shape = {RDM.shape}')
    print(f'[Check] Saved RDM: {save_path}')
    return RDM
    

def visualize_mask(sub, rois, maskROIs, smooth, threshold=0.00005):
    """

    """
    T1_path = f'{root_path}/Mack-Data/dropbox/sub-{sub}/anat/sub-{sub}_T1w.nii.gz'

    fig, (img_ax, cbar_ax) = plt.subplots(
        1,
        2,
        gridspec_kw={"width_ratios": [10.0, 0.1], "wspace": 0.0},
        figsize=(10, 2),
    )
    
    combined_mask = maskROIs[0].dataobj
    for maskROI in maskROIs[1:]:
        combined_mask += maskROI.dataobj
    combined_mask = image.new_img_like(maskROIs[0], combined_mask)
    
    cmap = ListedColormap(['red', 'green', 'blue', 'yellow', 'white'])
    
    plotting.plot_roi(
        combined_mask,
        colorbar=False, 
        threshold=threshold, 
        bg_img=T1_path,
        title=f'',
        axes=img_ax,
        cmap=cmap, 
    )
    
    norm = colors.Normalize(vmin=0, vmax=len(maskROIs))
    cbar = colorbar.ColorbarBase(
        cbar_ax,
        ticks=[0.5, 1.5, 2.5, 3.5, 4.5],
        norm=norm,
        orientation="vertical",
        cmap=cmap,
        spacing="proportional",
    )
    cbar_ax.set_yticklabels(['V1', 'V2', 'V3', 'V4', 'LOC'])
        
    img_ax.set_title(f'smooth = {smooth}')
    print(f'[Check] Saved roiMask_smooth={smooth}.png')
    plt.savefig(f'roiMask_smooth={smooth}.png')
    plt.close()


def visualize_RSM(RDM, sub, task, run, roi, distance):
    """
    Visualize RSM instead of RDM due to that's what was done in
    Mack et al.
    """
    RSM = np.ones((RDM.shape[0], RDM.shape[0])) - RDM
    
    fig, ax = plt.subplots()
    for i in range(RSM.shape[0]):
        for j in range(RSM.shape[0]):
            text = ax.text(
                j, i, np.round(RSM[i, j], 1),
                ha="center", va="center", color="w"
            )
    
    ax.set_title(f'distance = {distance}')
    plt.imshow(RSM)
    plt.savefig(
        f'RDMs/sub-{sub}_task-{task}_run-{run}_roi-{roi}_distance-{distance}.png'
    )


def compute_RSA(RDM1, RDM2):
    """
    RSA between a subject pairs given (task, run, roi, distance)
    """
    r, _ = stats.spearmanr(RDM1, RDM2)
    print(r)
    return r


def roi_execute(rois, subs, tasks, runs, dataType, conditions, smooth, viz_mask, viz_RSM):
    """
    This is a top-level execution routine that does the following in order:
    1. `merge_n_smooth_mask`: 
        - merge given ROI masks by left+right hemisphere and apply smoothing.
        - this step is subject general as it's in the MNI space.
        - the merged and smoothed masks are saved in `ROIs/`
        
    2. `transform_mask_MNI_to_T1` 
        - transform the saved masks from MNI space to subject's T1 space.
        - this step is subject specific.
        - the transformed masks are saved in `ROIs/*_T1.nii.gz`

    3. `applyMask`
        - extract beta weights based on ROI masks.
        - this step is done for tasks, runs and conditions, one at a time.
    
    4. `compute_RDM`
        - convert all stimuli beta weights (i.e. embedding matrix) into 
            RDM based on provided distance metric.
        - notice RDM entries need reordered.
        - optionally, RDMs can be plotted as heatmaps.
    """
    maskROIs = []
    
    for roi in rois:
        global roi_path
        if 'HPC' not in roi:
            roi_path = 'ROIs/ProbAtlas_v4/subj_vol_all'
        else:
            roi_path = 'ROIs/HPC'
        merge_n_smooth_mask(roi=roi, smooth=smooth)
        
        for sub in subs:
            transform_mask_MNI_to_T1(sub=sub, roi=roi)
            
            for task in tasks:
                for run in runs:
                    beta_weights_masked = []
                    for condition in conditions:
                        # given beta weights from a task & run & condition 
                        # apply the transformed mask
                        roi, maskROI, fmri_masked = applyMask(
                            roi=roi, 
                            sub=sub, 
                            task=task, 
                            run=run, 
                            dataType=dataType, 
                            condition=condition
                        )
                        
                        # visualize masks as a check
                        maskROIs.append(maskROI)
                        beta_weights_masked.append(fmri_masked)
                    beta_weights_masked = np.array(beta_weights_masked)
                    print(f'beta_weights_masked.shape = {beta_weights_masked.shape}')
                    print(f'[Check] End of sub{sub}, task{task}, run{run}')
                    
                    for distance in distances:
                        RDM = compute_RDM(
                            embedding_mtx=beta_weights_masked, 
                            sub=sub, 
                            task=task, 
                            run=run, 
                            roi=roi, 
                            distance=distance
                        )
                        if viz_RSM:
                            visualize_RSM(
                                RDM=RDM, 
                                sub=sub, 
                                task=task, 
                                run=run, 
                                roi=roi, 
                                distance=distance
                            )
                                    

def rsa_execute(subs, tasks, runs, rois, distances):
    for roi in rois:
        for task in tasks:
            for run in runs:
                for distance in distances:
                    
                    RDMs = []
                    for sub in subs:
                        RDM = np.load(
                           f'{rdm_path}/sub-{sub}_task-{task}_run-{run}_roi-{roi}_{distance}.npy'
                        )
                        
                    RDMs.append(RDM)
                    # compute correlation between pairs of subjects
                    for i in range(len(subs)):
                        for j in range(len(subs)):
                            if i >= j:
                                continue
                            else:
                                r = compute_RSA(RDMs[i], RDMs[j])

                    
if __name__ == '__main__':
    root_path = '/home/ken/projects/brain_data'
    glm_path = 'glm'
    rdm_path = 'RDMs'
    # rois = ['V1', 'V2', 'V3', 'V4', 'LOC']
    rois = ['LHHPC', 'RHHPC']
    num_subs = 23
    num_conditions = 8
    subs = [f'{i:02d}' for i in range(2, num_subs+1)]
    conditions = [f'{i:04d}' for i in range(1, num_conditions+1)]
    tasks = [2, 3]
    runs = [4]
    distances = ['pearson']
    
    roi_execute(
        rois=rois, 
        subs=subs, 
        tasks=tasks, 
        runs=runs, 
        dataType='beta',
        conditions=conditions,
        smooth=0.2,
        viz_mask=False,
        viz_RSM=True
    )
    
    # rsa_execute(
    #     subs=subs, 
    #     tasks=tasks, 
    #     runs=runs, 
    #     rois=rois, 
    #     distances=distances,
    #     visualize=True
    # )
    
    