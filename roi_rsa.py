import os
import scipy
import time
import numpy as np
import multiprocessing
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

def run_ants_command(roi, roi_path, roi_nums, smooth_mask=False):
    """
    Helper function that automatically grabs files to prepare 
    for the final ants command.
    """
    maths = MultiImageMaths()
    
    # V1,2,3,1-3,4,LOC (left+right)
    if roi_nums is not None:
        all_files = []
        for roi_num in roi_nums:
            all_files.append(f'{roi_path}/perc_VTPM_vol_roi{roi_num}_lh.nii.gz')
            all_files.append(f'{roi_path}/perc_VTPM_vol_roi{roi_num}_rh.nii.gz')
        
        maths.inputs.op_string = ''
        for _ in range(len(roi_nums) * 2 - 1):
            maths.inputs.op_string += '-add %s '
            
        if smooth_mask:
            maths.inputs.op_string += f'-s {smooth_mask}'
        maths.inputs.op_string += ' -bin '
        
        maths.inputs.in_file = all_files[0]
        maths.inputs.operand_files = all_files[1:]
    
    # HPC
    elif 'HHPC' in roi:
        if 'LH' in roi:
            h = 'l'
        else:
            h = 'r'
                        
        maths.inputs.in_file = f'{roi_path}/HIPP_BODY_{h}h.nii.gz'
        maths.inputs.op_string = '-add %s -add %s ' 
        if smooth_mask:
            maths.inputs.op_string += f'-s {smooth_mask}'
        maths.inputs.op_string += ' -bin '
        
        maths.inputs.operand_files = [
            f'{roi_path}/HIPP_HEAD_{h}h.nii.gz',
            f'{roi_path}/HIPP_TAIL_{h}h.nii.gz'
        ]
    
    # LHLOC, RHLOC
    elif 'HLOC' in roi:
        if 'LH' in roi:
            h = 'l'
        else:
            h = 'r'
        
        # WARNING: hardcoded roi_num
        maths.inputs.in_file = f'{roi_path}/perc_VTPM_vol_roi14_{h}h.nii.gz'
        maths.inputs.op_string = '-add %s ' 
        if smooth_mask:
            maths.inputs.op_string += f'-s {smooth_mask}'
        maths.inputs.op_string += ' -bin '
        
        maths.inputs.operand_files = [
            f'{roi_path}/perc_VTPM_vol_roi15_{h}h.nii.gz'
        ]
    
    maths.inputs.out_file = f'{roi_path}/mask-{roi}.nii.gz'
    runCmd = '/usr/bin/fsl5.0-' + maths.cmdline
    print(f'runCmd = {runCmd}')
    call(runCmd, shell=True)
    

def merge_n_smooth_mask(roi, roi_path, smooth_mask):
    """
    First processing of the standard ROI masks is to
    merge some left&right masks and smooth them.
    """
    print(f'[Check] running `merge_n_smooth_mask`')
    if not os.path.exists(f'{roi_path}/mask-{roi}.nii.gz'):
        
        if roi in ['LHHPC', 'RHHPC', 'LHLOC', 'RHLOC']:
            # If the above ROI, there is no left+right merge
            roi_nums = None
        else:
            # Will need to merge left+right
            roi_number_mapping = {
                'V1': [1, 2],
                'V2': [3, 4],
                'V3': [5, 6],
                'V4': [7],
                'V1-3': [1, 2, 3, 4, 5, 6],
                'LOC': [14, 15]
            }
            roi_nums = roi_number_mapping[roi]
        
        run_ants_command(
            roi=roi, 
            roi_path=roi_path, 
            roi_nums=roi_nums, 
            smooth_mask=smooth_mask
        )
    
    else:
        print(f'[Check] mask-{roi} already done, skip')        
   
    
def transform_mask_MNI_to_T1(sub, roi, roi_path, root_path):
    """
    Given a subject and a ROI, 
    transform ROI mask from MNI space to subject's T1 space.
    
    This is based on the fact that the ROI masks provided are already
    in standard MNI space.
    """
    print(f'[Check] running `transform_mask_MNI_to_T1`')
    if not os.path.exists(f'{roi_path}/mask-{roi}_T1_sub-{sub}.nii.gz'):
        print(f'[Check] transform roi mask to subject{sub} T1 space')
        at = ants.ApplyTransforms()
        
        reference_image_path = f'{root_path}/Mack-Data/dropbox/sub-{sub}/anat'
        transform_path = f'{root_path}/Mack-Data/derivatives/sub-{sub}/anat'
        assert os.path.exists(reference_image_path)
        assert os.path.exists(transform_path)
        
        at.inputs.dimension = 3
        at.inputs.input_image = f'{roi_path}/mask-{roi}.nii.gz'
        at.inputs.reference_image = f'{reference_image_path}/sub-{sub}_T1w.nii.gz'
        at.inputs.output_image = f'{roi_path}/mask-{roi}_T1_sub-{sub}.nii.gz'
        at.inputs.interpolation = 'NearestNeighbor'
        at.inputs.default_value = 0
        at.inputs.transforms = [f'{transform_path}/sub-{sub}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5']
        at.inputs.invert_transform_flags = [False]
        runCmd = at.cmdline
        call(runCmd, shell=True)
    else:
        print(f'[Check] mask-{roi}_T1 Already done, skip')
     

def applyMask(roi, root_path, glm_path, roi_path, sub, task, run, dataType, condition, smooth_beta):
    """
    Apply ROI mask (T1 space) to subject's whole brain beta weights.
    
    return:
    -------
        per (ROI, subject, task, run, condition) beta weights
    """
    print(f'[Check] apply mask, condition={condition}')
    output_path = f'output_run_{run}_sub_{sub}_task_{task}'
    
    if dataType == 'beta':
        data_path = f'{root_path}/{glm_path}/work_1st/{output_path}/datasink/' \
            f'{glm_path}/datasink/model/{output_path}/{dataType}_{condition}.nii'
    elif dataType == 'spmT':
        data_path = f'{root_path}/{glm_path}/work_1st/{output_path}/datasink/' \
            f'{glm_path}/datasink/1stLevel/{output_path}/{dataType}_{condition}.nii'
    
    imgs = nb.load(data_path)
    maskROI = nb.load(f'{roi_path}/mask-{roi}_T1_sub-{sub}.nii.gz')
    maskROI = nl.image.resample_img(
        maskROI, 
        target_affine=imgs.affine,
        target_shape=imgs.shape[:3], 
        interpolation='nearest'
    )
    # print('maskROI.shape = ', maskROI.shape)
    fmri_masked = apply_mask(
        imgs, maskROI, smoothing_fwhm=smooth_beta
    )
    # print('fmri_masked.shape, ', fmri_masked.shape)  # per ROI & subject & condition beta weights
    return roi, maskROI, fmri_masked


def return_RDM(embedding_mtx, sub, task, run, roi, distance):
    """
    Compute and save RDM or just load of given beta weights and sort 
    the conditions based on specified ordering.
    """
    if not os.path.exists(rdm_path):
        os.mkdir(rdm_path)
    
    RDM_fpath = f'{rdm_path}/sub-{sub}_task-{task}_run-{run}_roi-{roi}_{distance}_{dataType}.npy'
    
    if len(embedding_mtx) != 0:
        print(f'[Check] Computing RDM..')
        if distance == 'euclidean':
            RDM = pairwise_distances(embedding_mtx, metric='euclidean')
            
        elif distance == 'pearson':
            RDM = pairwise_distances(embedding_mtx, metric='correlation')
        
        # rearrange so entries are grouped by two categories.
        conversion_ordering = reorder_mapper[sub][task]
        print(f'[Check] sub{sub}, task{task}, conversion_ordering={conversion_ordering}')
        
        # reorder both cols and rows based on ordering.
        RDM = RDM[conversion_ordering, :][:, conversion_ordering]
        np.save(RDM_fpath, RDM)
        print(f'[Check] Saved: {RDM_fpath}')
        assert RDM.shape == (embedding_mtx.shape[0], embedding_mtx.shape[0])
        
    else:
        print(f'[Check] Already exists, {RDM_fpath}')
    

def applyMask_returnRDM(roi, root_path, glm_path, roi_path, sub, task, run, dataType, conditions, smooth_beta, distance):
    """
    Combines `applyMask` and `returnRDM` in one function,
    this is done so to enable multiprocessing.
    """
    # If a specific RDM has been saved,
    # ignore apply mask and compute RDM, 
    # just load it from disk.
    RDM_fpath = f'{rdm_path}/sub-{sub}_task-{task}_run-{run}_roi-{roi}_{distance}_{dataType}.npy'
    beta_weights_masked = []
    if not os.path.exists(RDM_fpath):
        for condition in conditions:
            # given beta weights from a task & run & condition 
            # apply the transformed mask
            roi, maskROI, fmri_masked = applyMask(
                roi=roi, 
                root_path=root_path,
                glm_path=glm_path,
                roi_path=roi_path,
                sub=sub, 
                task=task, 
                run=run, 
                dataType=dataType, 
                condition=condition,
                smooth_beta=smooth_beta
            )
            beta_weights_masked.append(fmri_masked)
        beta_weights_masked = np.array(beta_weights_masked)
        print('beta_weights_masked.shape', beta_weights_masked.shape)

    # Either way, return RDM
    return_RDM(
        embedding_mtx=beta_weights_masked, 
        sub=sub, 
        task=task, 
        run=run, 
        roi=roi, 
        distance=distance
    )


def kendall_a(a, b):
    """Kendalls tau-a
    Arguments:
        a {array} -- [description]
        b {[array]} -- [description]
    Returns:
        tau -- Kendalls tau-a
        
    E.g.
        x1 = np.random.random(10000)
        x2 = np.random.random(10000)
        print(kendall_a(x1,x2))
    """
    a, b = np.array(a), np.array(b)
    assert a.size == b.size, 'Both arrays need to be the same size'
    K = 0
    n = a.size
    for k in range(n):
        pairRelations_a = np.sign(a[k]-a[k+1:])
        pairRelations_b = np.sign(b[k]-b[k+1:])
        K = K + np.sum(pairRelations_a * pairRelations_b)
    return K/(n*(n-1)/2)


def compute_RSA(RDM_1, RDM_2, method):
    """
    Compute spearman correlation between 
    two RDMs' upper trigular entries
    """
    RDM_1_triu = RDM_1[np.triu_indices(RDM_1.shape[0])]
    RDM_2_triu = RDM_2[np.triu_indices(RDM_2.shape[0])]
    
    if method == 'spearman':
        rho, _ = stats.spearmanr(RDM_1_triu, RDM_2_triu)
    elif method == 'kendall_a':
        rho = kendall_a(RDM_1_triu, RDM_2_triu)
    
    return rho
    
    
def roi_execute(
        rois, subs, tasks, runs, 
        dataType, conditions, distances, 
        smooth_mask, smooth_beta, 
        num_processes
    ):
    """This is a top-level execution routine that does the following in order:
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
    
    4. `return_RDM`
        - convert all stimuli beta weights (i.e. embedding matrix) into 
            RDM based on provided distance metric.
        - notice RDM entries need somehow reordered.
    """
    with multiprocessing.Pool(num_processes) as pool:
        for roi in rois:
           
            if 'HPC' not in roi:
                roi_path = 'ROIs/ProbAtlas_v4/subj_vol_all'
            else:
                roi_path = 'ROIs/HPC'
            merge_n_smooth_mask(roi=roi, roi_path=roi_path, smooth_mask=smooth_mask)
            
            for sub in subs:
                transform_mask_MNI_to_T1(sub=sub, roi=roi, roi_path=roi_path, root_path=root_path)
                
                for task in tasks:
                    for run in runs:
                        for distance in distances:
                            
                            # Create a single process to produce 
                            # 1 RDM.
                            res_obj = pool.apply_async(
                                applyMask_returnRDM, 
                                args=[
                                    roi, root_path, glm_path, roi_path, sub, task, run, dataType, 
                                    conditions, smooth_beta, distance
                                ]
                            )
        print(res_obj.get())
        pool.close()
        pool.join()
                                                            

def visualize_RDM(subs, roi, problem_type, distance):
    """
    Visualize RSM instead of RDM due to that's what was done in
    Mack et al.
    
    Will average over later runs like Mack et al.
    """
    runs = [3, 4]
    RDM_sum = np.zeros((8, 8))
    for sub in subs:
        for run in runs:
            # even sub: Type1 is task2
            if int(sub) % 2 == 0:
                if problem_type == 1:
                    task = 2
            # odd sub: Type1 is task3
            else:
                if problem_type == 1:
                    task = 3
                    
            RDM = np.load(
                f'{rdm_path}/sub-{sub}_task-{task}_run-{run}_roi-{roi}_{distance}.npy'
            )
            RDM_sum += RDM
    
    RDM_avg = RDM_sum / (len(subs) * len(runs))
    
    fig, ax = plt.subplots()
    for i in range(RDM_avg.shape[0]):
        for j in range(RDM_avg.shape[0]):
            text = ax.text(
                j, i, np.round(RDM_avg[i, j], 1),
                ha="center", va="center", color="w"
            )
    
    ax.set_title(f'distance: {distance}')
    plt.imshow(RDM_avg)
    plt.savefig(
        f'RDMs/average_problem_type-{problem_type}_roi-{roi}_distance-{distance}.png'
    )
    print(f'[Check] plotted.')


def correlate_against_ideal_RDM(rois, distance, problem_type, num_shuffles, method, dataType, seed=999):
    """
    Correlate each subject's RDM to the ideal RDM 
    of a given type. To better examine the significance of 
    the correlations, we build in a shuffle mechanism that
    randomly permutes the entries of the RDM to determine if 
    the correlation we get during un-shuffled is real.
    """
    # exc. the _fb conditions
    ideal_RDM = np.ones((num_conditions, num_conditions))
    ideal_RDM[:4, :4] = 0
    ideal_RDM[4:, 4:] = 0
    # ideal_RDM = np.zeros((num_conditions, num_conditions))
    # ideal_RDM[:4, :4] = 1
    # ideal_RDM[4:, 4:] = 1
            
    for roi in rois:
        for run in runs:
            
            np.random.seed(seed)
            all_rho = []  # one per subject-run of a task
            for shuffle in range(num_shuffles):   
                for sub in subs:

                    # even sub: Type1 is task2, Type2 is task3
                    if int(sub) % 2 == 0:
                        if problem_type == 1:
                            task = 2
                        elif problem_type == 2:
                            task = 3
                        else:
                            task = 1
                            
                    # odd sub: Type1 is task3, Type2 is task2
                    else:
                        if problem_type == 1:
                            task = 3
                        elif problem_type == 2:
                            task = 2
                        else:
                            task = 1
                            
                    sub_RDM = np.load(
                        f'{rdm_path}/sub-{sub}_task-{task}_run-{run}_roi-{roi}_{distance}_{dataType}.npy'
                    )
                    
                    if num_shuffles > 1:
                        shuffle_indices = np.random.choice(
                            range(sub_RDM.shape[0]), 
                            size=sub_RDM.shape[0],
                            replace=False
                        )
                        # print(f'shuffle_indices={shuffle_indices}')
                        sub_RDM = sub_RDM[shuffle_indices, :]

                    # compute correlation to the ideal RDM
                    rho = compute_RSA(sub_RDM, ideal_RDM, method=method)
                    all_rho.append(rho)
            print(
                f'Dist=[{distance}], Type=[{problem_type}], roi=[{roi}], run=[{run}], ' \
                f'avg_rho=[{np.mean(all_rho):.2f}], ' \
                f'std=[{np.std(all_rho):.2f}], ' \
                f't-stats=[{stats.ttest_1samp(a=all_rho, popmean=0)[0]:.2f}], ' \
                f'pvalue=[{stats.ttest_1samp(a=all_rho, popmean=0)[1]:.2f}]' \
            )    
        print('------------------------------------------------------------------------')

    
def correlate_against_ideal_RDM_regression(rois, distance, problem_type, method, dataType):
    """
    Correlate each subject's RDM to the ideal RDM 
    of a given type. For each (problem_type, run, sub) 
    there is a correlation score (rho). 
    
    We will then fit linear regression model for each subject
    across runs to obtain a co-efficient. This co-efficient 
    indicates how correlation change over runs and should ideally be 
    positive and significant over subjects.
    """
    import pingouin as pg

    # exc. the _fb conditions
    ideal_RDM = np.ones((num_conditions, num_conditions))
    ideal_RDM[:4, :4] = 0
    ideal_RDM[4:, 4:] = 0
            
    for roi in rois:
        coefs = []
        all_rho = np.empty((len(runs), num_subs))
        
        for r in range(len(runs)):
            run = runs[r]
            for s in range(num_subs):
                sub = subs[s]
                
                # even sub: Type1 is task2, Type2 is task3
                if int(sub) % 2 == 0:
                    if problem_type == 1:
                        task = 2
                    elif problem_type == 2:
                        task = 3
                    else:
                        task = 1
                        
                # odd sub: Type1 is task3, Type2 is task2
                else:
                    if problem_type == 1:
                        task = 3
                    elif problem_type == 2:
                        task = 2
                    else:
                        task = 1
                        
                sub_RDM = np.load(
                    f'{rdm_path}/sub-{sub}_task-{task}_run-{run}_roi-{roi}_{distance}_{dataType}.npy'
                )

                # compute correlation to the ideal RDM
                rho = compute_RSA(sub_RDM, ideal_RDM, method=method)
                all_rho[r, s] = rho
        
        # fit linear regression models for each
        # subject and obtain coefficient.
        for s in range(num_subs):
            X_sub = [1, 2, 3, 4]
            y_sub = all_rho[:, s]
            coef = pg.linear_regression(X=X_sub, y=y_sub, coef_only=True)
            # coefs.append(coef)
            print(f'sub{subs[s]}, coef={coef[-1]:.3f}')

       
if __name__ == '__main__':
    root_path = '/home/ken/projects/brain_data'
    glm_path = 'glm_run-estimate_Mack2016'
    rdm_path = 'subject_RDMs_Mack2016'
    rois = ['LHHPC']
    num_subs = 23
    dataType = 'beta'
    num_conditions = 24  # stimulus + _fb + _resp
    tasks = [1, 2, 3]
    runs = [1, 2, 3, 4]
    distances = ['euclidean', 'pearson']   
    subs = [f'{i:02d}' for i in range(2, num_subs+2) if i!=9]
    num_subs = len(subs)
    
    if dataType == 'beta':
        conditions = [f'{i:04d}' for i in range(1, num_conditions+1, 3)]
        print(f'[Check] conditions=\n{conditions}')
        num_conditions = len(conditions)
        
    reorder_mapper = reorder_RDM_entries_into_chunks()
    
    # roi_execute(
    #     rois=rois, 
    #     subs=subs, 
    #     tasks=tasks, 
    #     runs=runs, 
    #     dataType=dataType,
    #     conditions=conditions,
    #     distances=distances,
    #     smooth_mask=0.2,
    #     smooth_beta=2,
    #     num_processes=70
    # )
    
    # correlate_against_ideal_RDM(
    #     rois=rois, 
    #     distance='pearson',
    #     problem_type=1,
    #     seed=999, 
    #     num_shuffles=1,
    #     method='spearman',
    #     dataType='beta'
    # )    
    
    correlate_against_ideal_RDM_regression(
        rois=rois, 
        distance='pearson',
        problem_type=1,
        method='spearman',
        dataType='beta'
    )    