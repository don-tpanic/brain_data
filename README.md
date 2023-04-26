# A controller-peripheral architecture and costly energy principle for learning

### Description
This is a complementary repo to <link> which focuses on fMRI BOLD data processing and analysis in this [research paper](https://www.biorxiv.org/content/10.1101/2023.01.16.524194v1).

### Structure and main components
This repo can be roughly divided into two parts: *fMRI data processing* and *fMRI data analysis*.
<br /> For fMRI data processing, 
1. `bids.py` organises raw fMRI datasets into the standard Brain Imaging Data Structure (BIDS).
2. `glm.py` fits Generalised Linear Models (GLMs) on preprocessed fMRI BOLD data and extracts beta weights for actual analysis.

For fMRI data analysis,
1. `roi_rsa.py` runs a similar representational similarity analysis in [Mack et al., 2016](https://www.pnas.org/doi/10.1073/pnas.1614048113)
2. `pca.py` runs a similar analysis in [Mack et al., 2020](https://www.nature.com/articles/s41467-019-13930-8)
3. `pca_3runs.py` runs a similar analysis in [Ahlheim et al., 2018](https://www.sciencedirect.com/science/article/pii/S1053811918305226)
4. `decoding.py` runs a similar neural decoding analysis in [Braunlich & Love, 2019](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC6586152/)

### Environment setup
1. Create a docker file using neurodocker
```python
neurodocker generate docker \
--pkg-manager apt \
--base-image neurodebian:stretch-non-free \
--arg DEBIAN_FRONTEND=noninteractive \
--install convert3d fsl ants gcc g++ graphviz tree \
        git-annex-standalone vim emacs-nox nano less ncdu \
        tig git-annex-remote-rclone octave netbase \
--spm12 version=r7771 \
--miniconda \
version=latest \
conda_install="python=3.8 pytest jupyter jupyterlab jupyter_contrib_nbextensions
                traits pandas matplotlib scikit-learn scikit-image seaborn nbformat
                nb_conda" \
pip_install="https://github.com/nipy/nipype/tarball/master
                https://github.com/INCF/pybids/tarball/master
                nilearn nipy duecredit nbval" \
> nipype.Dockerfile
```
2. Build an image based on the dockerfile (make sure the directory where the Dockerfile is is empty). `.` means using the current directory.
```
docker build --tag nipype .
```
3. Start an iterative session inside the image, changes will be removed after exiting.
```
docker run -it --rm nipype
```
4. Mount a local directory inside the above container’s diretory. Changes made inside the container will change the local container.<br />
```
docker run -it --rm -v /home/ken/projects/brain_data/:/home/ken/projects/brain_data/ nipype
```

### Attribution
```
@article {Luo2023.01.16.524194,
    author = {Xiaoliang Luo and Robert M. Mok and Brett D. Roads and Bradley C. Love},
    title = {A controller-peripheral architecture and costly energy principle for learning},
    elocation-id = {2023.01.16.524194},
    year = {2023},
    doi = {10.1101/2023.01.16.524194},
    publisher = {Cold Spring Harbor Laboratory},
}
```
