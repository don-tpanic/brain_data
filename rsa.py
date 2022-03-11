import numpy as np
import nibabel as nb
import scipy.stats as stats
from sklearn.metrics.pairwise import cosine_distances
from sklearn.metrics.pairwise import euclidean_distances


def compute_ROI(sub, task, run):
    """
    Return beta weights from speficied ROIs
    """
    output_path = f'output_run_{run}_sub_{sub}_task_{task}'
    beta_path = f'{root_path}/{base_dir}/work_1st/{output_path}' \
                f'/datasink/{base_dir}/datasink/model/{output_path}'

    beta_weights_masked = []

    # TODO: for testing we return whole brain beta weights
    beta_0001 = np.array(
        nb.load(f'{beta_path}/beta_0001.nii').dataobj
    ).flatten()
    beta_0002 = np.array(
        nb.load(f'{beta_path}/beta_0002.nii').dataobj
    ).flatten()    
    # TODO: what to do with NaN?
    
    beta_weights_masked = [beta_0001, beta_0002]
    return beta_weights_masked


def compute_RDM(sub, task, run, beta_weights_masked, distance):
    """
    Compute RDM of given beta weights
    """
    num_stimuli = len(beta_weights_masked)
    flatten_length = len(beta_weights_masked[0])
    embedding_mtx = np.empty((num_stimuli, flatten_length))
    
    for row_idx in range(num_stimuli):
        # Convert nan to 0
        temp = beta_weights_masked[row_idx]
        temp[np.isnan(temp)] = 0.
        embedding_mtx[row_idx, :] = temp
    
    if distance == 'euclidean':
        RDM = euclidean_distances(embedding_mtx)
    
    save_path = f'RDMs/sub-{sub}_task-{task}_run-{run}_{distance}.npy'
    np.save(save_path, RDM)
    print(f'[Check] Saved RDM: sub-{sub}_task-{task}_run-{run}_{distance}')
    

def compute_RSA(subs, tasks, runs, distance):
    """
    """
    task = 1
    run = 1
    sub1 = np.load(f'RDMs/sub-02_task-{task}_run-{run}_{distance}.npy')
    sub2 = np.load(f'RDMs/sub-03_task-{task}_run-{run}_{distance}.npy')
    
    # FIXME: RSA cannot be directly computed because
    # for each subject, the physical meaning of the same stimulus coding
    # is different which need to be unified.
    
    sub1 = sub1[np.triu_indices(sub1.shape[0])]
    sub2 = sub2[np.triu_indices(sub2.shape[0])]
    r, _ = stats.spearmanr(sub1, sub2)
    print(r)


def execute(subs, tasks, runs, distance):
    for sub in subs:
        for task in tasks:
            for run in runs:
                ROI_beta_weights = compute_ROI(sub, task, run)
                compute_RDM(sub, task, run, ROI_beta_weights, distance)
    
    # TODO: pairwise all combo RSA?
    # For testing, we just selet 2 subs and 1 run
    compute_RSA(subs, tasks, runs, distance)
    

if __name__ == '__main__':        
    root_path = '/home/ken/projects/brain_data'
    base_dir = 'glm_test'
    subs = ['02', '03']
    tasks = ['1']
    runs = ['1']
    distance = 'euclidean'
    execute(
        subs=subs, tasks=tasks, runs=runs, 
        distance=distanc
    )

