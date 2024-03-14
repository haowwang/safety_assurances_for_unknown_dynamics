# Providing Safety Assurances for Systems with Unknown Dynamics
arxiv link: https://arxiv.org/abs/2403.05771

TurtleBot experiment supplemental video link: https://youtu.be/D6ZxZG0rQ_U

Important Note: Please remember to add *all* the files in this repository to your MATLAB path. 

## Inverted Pendulum Experiment
- The experiment directory contains 
    - Datasets for training ensemble dynamics models in ```dyn_datasets```
    - A trained ensemble dynamics model using 100 training samples.
    - Ensemble forward results for the ensemble model trained with 100 training samples. Located in directory ```ensemble_dyn_forward_results```
    - A grid for helperOC used in BRT computation ```inv_pend_pi_10_200_200_grid_data.mat```
    - BRT figures for the ground truth, 0std (Baseline 1 Mean Dynamics), 3std (our method) located in directory ```figs```
- To visualize BRT figures for this experiment (Fig. 1 in the paper)
    - Run ```brt_plotting.m``` in MATLAB
- To generate your datasets
    - Run ```python inv_pend_dyn_data_collection.py```
        - You can change the parameters of the pendulum ```l,m,b``` and the number of training samples within the script. The dataset will be saved to ```dyn_datasets```. 
- To traing your own ensemble dynamics model and generate the ensemble forward results
    - Run ```python inv_pend_dyn_training_driver.py```
        - You can change training arguments, such as number of ensemble models, number of hidden layers, and learning rate, in ```inv_pend_dyn_training_config.txt```. 
    - Run ```python inv_pend_ensemble_dyn_forward.py```
        - You need to make sure the training experiment name is properly spelled out within the script
- To compute the BRT (after generating the ensemble forward results)
    - Run ```inv_pend_brt_comp_script.m``` in MATLAB. Make sure the dataset name and forward result file name are correct within the script.
    - You can save the figure and use ```brt_plotting.m``` to compare BRTs of different dynamics models, model uncertainty level, etc. 


Last updated 3/14/24