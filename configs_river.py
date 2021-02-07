import os, sys, inspect
from utils_keras_models import *
from model_configs import *

### Get Data ###

base_path= 'C:/Users/doyle/Documents/Coding/HAL24K/'
sub_folder='darlaston'
data_folder= base_path + 'data/river_trent/' + sub_folder +'/'


### Define Model Inputs ###
input_size = 2 # no of features
# Define Prediction variable
pred_var = 'stage'
all_vars = ['stage', 'flow', 'rain']
stage = False
flow = False
rain = True
#specify which variables need to be scaled
scaler_vars = ['stage', 'flow', 'rain']

#add selected variables to dictionary
cond_vars_dict = dict(((k, eval(k)) for k in all_vars))

cond_vars_selected = {k: v for k, v in cond_vars_dict.items() if v is not False}
no_cond_vars=(sum(value == True for value in cond_vars_dict.values()))
assert no_cond_vars == input_size-1, "select the correct no of variables"

#set folder name for storing results

if input_size==1:
    input_names= '{}_unconditional'.format(pred_var)
else:
    input_names='__'.join(map(str, cond_vars_selected.keys()))
    input_names= '{}_{}'.format(pred_var, input_names)

results_folder= 'results_{}/'.format(model_type.lower())
if os.path.exists(base_path + results_folder)==False:
    os.mkdir(base_path + results_folder)

results_folder= results_folder + '{}/'.format(sub_folder)
if os.path.exists(base_path + results_folder)==False:
    os.mkdir(base_path + results_folder)

results_folder= results_folder + input_names +'/'
if os.path.exists(base_path + results_folder)==False:
    os.mkdir(base_path + results_folder)
    print('created results folder at', base_path + results_folder)
results_folder=base_path +results_folder
