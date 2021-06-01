# Deep Markov Factor Analysis (DMFA)

Codes and experiments for deep Markov factor analysis (DMFA)


### Dependencies: 
Numpy, Scipy, Pytorch, Nibabel, Tqdm, Matplotlib, Sklearn, Json, Pandas  

## Autism Dataset:

Run the following snippet to restore results from pre-trained checkpoints for Autism dataset in `./fMRI_results` folder. A few instances from each dataset are included to help the code run without errors. 
You may replace `{site}` with `Caltec`, `Leuven`, `MaxMun`, `NYU_00`, `SBL_00`, `Stanfo`, `Yale_0`, `USM_00`, `DSU_0`, `UM_1_0`, or set `-exp autism` for the full dataset. Here, checkpoint files for `Caltec`, `SBL_00`, `Stanfo` are only included due to storage limitations.

`python dmfa_fMRI.py -t 75 -exp autism_{site} -dir ./data_autism/ -smod ./ckpt_fMRI/ -dpath ./fMRI_results/ -restore`

or run the following snippet for training with batch size of 10 (full dataset needs to be downloaded and preprocessed/formatted beforehand):

`python dmfa_fMRI.py -t 75 -exp autism_{site} -dir ./data_autism/ -smod ./ckpt_fMRI/ -dpath ./fMRI_results/ -bs 10`

After downloading the full Autism dataset, run the following snippet to preprocess/format data:

`python generate_fMRI_patches.py -T 75 -dir ./path_to_data/ -ext /*.gz -spath ./data_autism/`

## Depression Dataset:

Run the following snippet to restore results from pre-trained checkpoints for Depression dataset in `./fMRI_results` folder. A few instances from the dataset are included to help the code run without errors.
You may replace `{ID}` with `1`, `2`, `3`, `4`. ID `4` corresponds to the first experiment on Depression dataset in the paper. IDs `2`, `3` correspond to the second experiment on Depression dataset in the paper.

`python dmfa_fMRI.py -exp depression_{ID} -dir ./data_depression/ -smod ./ckpt_fMRI/ -dpath ./fMRI_results/ -restore`

or run the following snippet for training with batch size of 10 (full dataset needs to be downloaded and preprocessed/formatted beforehand):

`python dmfa_fMRI.py -exp depression_{ID} -dir ./data_depression/ -smod ./ckpt_fMRI/ -dpath ./fMRI_results/ -bs 10`

After downloading the full Depression dataset, run the following snippet to preprocess/format data:

`python generate_fMRI_patches_depression.py -T 6 -dir ./path_to_data/ -spath ./data_depression/`


## Synthetic fMRI data:

Run the following snippet to restore results from the pre-trained checkpoint for the synthetic experiment in `./synthetic_results` folder (synthetic fMRI data is not included due to storage limitations).

`python dmfa_synthetic.py`
