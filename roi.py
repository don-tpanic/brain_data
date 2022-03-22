import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances

import nipype.interfaces.ants as ants
from subprocess import call
import nibabel as nb
import nilearn as nl
from nilearn.masking import apply_mask
from nilearn.plotting import plot_glass_brain, plot_stat_map

"""
Extract ROI-level beta weights and compile into RDMs.
"""

def transform(sub, roi):
    """
    Given a subject and a ROI, 
    transform ROI mask from MNI space to subject's T1 space.
    """
    print(f'[Check] transform roi mask to subject{sub} T1 space')
    at = ants.ApplyTransforms()
    
    reference_image_path = f'{root_path}/Mack-Data/dropbox/sub-{sub}/anat'
    transform_path = f'{root_path}/Mack-Data/derivatives/sub-{sub}/anat'
    assert os.path.exists(reference_image_path)
    assert os.path.exists(transform_path)
    
    at.inputs.dimension = 3
    
    # TODO: change ROI dir
    at.inputs.input_image = f'{roi_path}/derivatives_spm_sub-CSI1_sub-CSI1_mask-{roi}.nii.gz'
    at.inputs.reference_image = f'{reference_image_path}/sub-{sub}_T1w.nii.gz'
    at.inputs.output_image = f'{roi_path}/derivatives_spm_sub-CSI1_sub-CSI1_mask-{roi}_output.nii.gz'
    at.inputs.interpolation = 'NearestNeighbor'
    at.inputs.default_value = 0
    at.inputs.transforms = [f'{transform_path}/sub-{sub}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5']
    at.inputs.invert_transform_flags = [False]
    runCmd = at.cmdline
    call(runCmd, shell=True)
     

def applyMask(
        roi, sub, task, run, dataType='beta', condition='0001'):
    """
    Apply ROI mask to subject's whole brain beta weights.
    
    return:
    -------
        per (ROI, subject, task, run, condition) beta weights
    """
    output_path = f'output_run_{run}_sub_{sub}_task_{task}'
    data_path = f'{root_path}/{glm_path}/work_1st/{output_path}/datasink/' \
        f'{glm_path}/datasink/model/{output_path}/{dataType}_{condition}.nii'
    imgs = nb.load(data_path)
    print(f'[Check] beta weight file: {data_path}')
    
    # TODO: change ROI dir
    maskROI = nb.load(
        f'{roi_path}/derivatives_spm_sub-CSI1_sub-CSI1_mask-{roi}_output.nii.gz'
    )
    maskROI = nl.image.resample_img(
        maskROI, 
        target_affine=imgs.affine,
        target_shape=imgs.shape[:3], 
        interpolation='nearest'
    )
    print('maskROI.shape = ', maskROI.shape)
    fmri_masked = apply_mask(
        imgs, maskROI, smoothing_fwhm=None
    )
    print(fmri_masked.shape)  # per ROI & subject & condition beta weights
    return roi, maskROI, fmri_masked


def compute_RDM(embedding_mtx, sub, task, run, roi, distance):
    """
    Compute RDM of given beta weights
    """    
    if distance == 'euclidean':
        RDM = euclidean_distances(embedding_mtx)
    elif distance == 'cosine':
        NotImplementedError()
    
    save_path = f'{rdm_path}/sub-{sub}_task-{task}_run-{run}_roi-{roi}_{distance}.npy'
    np.save(save_path, RDM)
    print(f'[Check] RDM shape = {RDM.shape}')
    print(f'[Check] Saved RDM: {save_path}')
    

def visualize_mask(sub, roi, maskROI, threshold):
    """
    Given a final roi mask, visualize as an
    activation plot with `plot_glass_brain`
    """
    T1_path = f'{root_path}/Mack-Data/dropbox/sub-{sub}/anat/sub-{sub}_T1w.nii.gz'

    fig, ax = plt.subplots()
    plot_stat_map(
        maskROI, 
        colorbar=True, 
        threshold=0.00005, 
        bg_img=T1_path,
        title=f'roi: {roi}',
        axes=ax
    )
    
    plt.savefig(f'roiMask.png')
    plt.close()


def execute(rois, subs, tasks, runs, conditions):
    """
    1. `transform`: Transform mask to T1 space
    2. `applyMask`: extract beta weights within a ROI
    3. `compute_RDM`: save RDM
    """
    for roi in rois:
        for sub in subs:
            
            # TODO: somewhere here ROI needs a mapping due to naming.
            transform(sub=sub, roi=roi)  # get subject-specific roi mask
            
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
                            dataType='beta', 
                            condition=condition
                        )
                        beta_weights_masked.append(fmri_masked)
                    
                    beta_weights_masked = np.array(beta_weights_masked)
                    print(f'beta_weights_masked.shape = {beta_weights_masked.shape}')
                    
                    for distance in distances:
                        compute_RDM(
                            embedding_mtx=beta_weights_masked, 
                            sub=sub, 
                            task=task, 
                            run=run, 
                            roi=roi, 
                            distance=distance
                        )
                    
                    
if __name__ == '__main__':
    root_path = '/home/ken/projects/brain_data'
    glm_path = 'glm'
    roi_path = 'ROIs'
    rdm_path = 'RDMs'
    rois = ['LHLOC']
    subs = []
    for i in range(2, 4):
        if len(f'{i}') == 1:
            subs.append(f'0{i}')
        else:
            subs.append(f'{i}')
    tasks = [1]
    runs = [1]
    conditions = ['0001', '0002']
    distances = ['euclidean']
    execute(
        rois=rois, 
        subs=subs, 
        tasks=tasks, 
        runs=runs, 
        conditions=conditions
    )    
    
    # FIXME: diff between anat space, T1 and MNI?