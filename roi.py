import os
import matplotlib.pyplot as plt

import nipype.interfaces.ants as ants
from subprocess import call
import nibabel as nb
import nilearn as nl
from nilearn.masking import apply_mask
from nilearn.plotting import plot_glass_brain


def transform():
    """
    Transform ROI mask from MNI space to subject's T1 space.
    """
    roiNames = ['LHLOC']
    at = ants.ApplyTransforms() # define function
    
    for sub in range(2, 3):
        sub = f'{sub:02d}'

        reference_image_path = f'{root_path}/Mack-Data/dropbox/sub-{sub}/anat'
        transform_path = f'{root_path}/Mack-Data/derivatives/sub-{sub}/anat'
        assert os.path.exists(reference_image_path)
        assert os.path.exists(transform_path)
        
        for roi in roiNames:
            print(f'Transforming ROIs from MNI to T1 space: sub-{sub}, {roi}')
            at.inputs.dimension = 3
            at.inputs.input_image = f'derivatives_spm_sub-CSI1_sub-CSI1_mask-LHLOC.nii.gz'
            at.inputs.reference_image = f'{reference_image_path}/sub-{sub}_T1w.nii.gz'
            at.inputs.output_image = f'derivatives_spm_sub-CSI1_sub-CSI1_mask-LHLOC_output.nii.gz'
            at.inputs.interpolation = 'NearestNeighbor'
            at.inputs.default_value = 0
            at.inputs.transforms = [f'{transform_path}/sub-{sub}_from-MNI152NLin2009cAsym_to-T1w_mode-image_xfm.h5']
            at.inputs.invert_transform_flags = [False]
            print('[Check] running command: ', at.cmdline)
            runCmd = at.cmdline
            call(runCmd, shell=True) # run in cmd line via python. to check output, use subprocess.check_output: from subprocess import check_output
        

def applyMask(
        roiName='LHLOC', sub='02', task='1', run='1', dataType='beta', condition='0001'):
    """
    Apply ROI mask to subject's whole brain beta weights.
    """
    output_dir = f'output_run_{run}_sub_{sub}_task_{task}'

    imgs = nb.load(
        f'{root_path}/{base_dir}/work_1st/{output_dir}/datasink/' \
        f'{base_dir}/datasink/model/{output_dir}/{dataType}_{condition}.nii'
    )
    maskROI = nb.load(
        f'derivatives_spm_sub-CSI1_sub-CSI1_mask-{roiName}_output.nii.gz'
    )
    maskROI = nl.image.resample_img(
        maskROI, 
        target_affine=imgs.affine,
        target_shape=imgs.shape[:3], 
        interpolation='nearest'
    )
    fmri_masked = apply_mask(
        imgs, maskROI, smoothing_fwhm=None
    )
    print(fmri_masked.shape)
    return roiName, maskROI
    

def visualize_mask(roiName, maskROI, threshold):
    """
    Given a final roi mask, visualize as an
    activation plot with `plot_glass_brain`
    """
    fig, ax = plt.subplots()
    plot_glass_brain(
        stat_map_img=maskROI,
        colorbar=True, 
        display_mode='lyrz', 
        black_bg=True, 
        threshold=0,
        title=f'roi mask - {roiName}',
        axes=ax)
    plt.savefig(f'roiMask.png')
    plt.close()


def execute():
    pass


if __name__ == '__main__':
    root_path = '/home/ken/projects/brain_data'
    base_dir = 'glm'
    # transform()
    roiName, maskROI = applyMask()
    visualize_mask(roiName=roiName, maskROI=maskROI, threshold=10)
    