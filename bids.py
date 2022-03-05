import subprocess
import sys
import json


def BidsData(root, subs):

    # create required directories and files
    subprocess.run(['mkdir', f'{root}/code'])
    subprocess.run(['echo', f'{root}/README', '>>', 'not empty!'])

    for sub in subs:

        print(f'[Check] processing.. sub{sub}')
        
        # uncompress
        orig_sub_file = f'{root}/subject_{sub}.tar.gz'
        subprocess.run(['tar', 'xvzf', f'{orig_sub_file}', '--directory', f'{root}/'])

        # rename subject dir name
        orig_sub_path = f'{root}/scratch/01812/mm63378/evolverep/subject_{sub}'
        new_sub_path = f'{root}/sub-{sub}'
        subprocess.run(['mv', f'{orig_sub_path}', f'{new_sub_path}'])

        # remove the untar `scratch` directory because next sub once untar will
        # also produce a directory called `scratch`
        subprocess.run(['rm', '-r', f'{root}/scratch'])

        # rename functional to func, anatomy to anat, fieldmap to fmap
        subprocess.run(['mv', f'{root}/sub-{sub}/functional', f'{root}/sub-{sub}/func'])
        subprocess.run(['mv', f'{root}/sub-{sub}/anatomy', f'{root}/sub-{sub}/anat'])
        subprocess.run(['mv', f'{root}/sub-{sub}/fieldmap', f'{root}/sub-{sub}/fmap'])

        # rename a bunch of directories and files
        typesOfData = ['func', 'fmap', 'anat']
        for typeOfData in typesOfData:

            # move into each sub dir and make changes
            sub_dir = f'{root}/sub-{sub}/{typeOfData}'
            if typeOfData == 'func':
                run = 0 
                for unique_run in range(1, total_runs+1):

                    if unique_run <=4:
                        study = 1
                    elif unique_run > 4 and unique_run <= 8:
                        study = 2
                    else:
                        study = 3
                    
                    if run == 4:
                        run = 1
                    else:
                        run += 1

                    # rename sub-task-run level file
                    print('--------------------------------------------------------------')
                    print(f'[Check]')
                    print(f'{sub_dir}/functional_{unique_run}/bold.nii.gz')
                    print(f'{sub_dir}/sub-{sub}_task-{study}_run-0{run}_bold.nii.gz')
                    print('--------------------------------------------------------------')
                    subprocess.run(
                        ['mv', 
                        f'{sub_dir}/functional_{unique_run}/bold.nii.gz',
                        f'{sub_dir}/sub-{sub}_task-{study}_run-0{run}_bold.nii.gz'
                        ]
                    )

                    # remove the old functional_* (now empty) directories
                    subprocess.run(
                        ['rm', '-r', f'{sub_dir}/functional_{unique_run}']
                    )

                    # add minimum meta data
                    fpath = f'{sub_dir}/sub-{sub}_task-{study}_run-0{run}_bold'
                    data_dictionary = {}
                    data_dictionary['RepetitionTime'] = 2.
                    data_dictionary['TaskName'] = 'SHJ'

                    with open(f'{fpath}.json', 'w') as f:
                        json.dump(data_dictionary, f)
                        print(f'{fpath}.json dumped.')
        
            
            # Simply remove fmap data as not used in Mack et al.
            # Removing this will not cause error when BIDS
            elif typeOfData == 'fmap':
                subprocess.run(
                    ['rm', '-r', f'{sub_dir}']
                )

            elif typeOfData == 'anat':
                subprocess.run(
                        ['mv', 
                        f'{sub_dir}/highres.nii.gz',
                        f'{sub_dir}/sub-{sub}_T1w.nii.gz'
                        ]
                    )
                
                subprocess.run(['rm', f'{sub_dir}/coronal1.nii.gz'])
                subprocess.run(['rm', f'{sub_dir}/coronal2.nii.gz'])


if __name__ == '__main__':
    subs = []
    for i in range(2, 25):
        if len(f'{i}') == 1:
            subs.append(f'0{i}')
        else:
            subs.append(f'{i}')
            
    total_runs = 12    # each problem has 4 runs
    num_studies = 3    # problem_type 1, 2, 6
    runs_per_study = 12 / 3
    root = 'Mack-Data/dropbox'
    BidsData(root, subs)
