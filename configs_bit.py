import os, sys, inspect
from utils_keras_models import *
from model_configs import *

### Get Data ###
sys_path = 'C:/Users/doyle/Documents/Coding/WaveNet_shannon/'
sys.path.insert(1, sys_path)
base_path= 'C:/Users/doyle/Documents/Coding/WaveNet_shannon/'
data_folder= 'data/'
sub_folder = 'binance_bitcoin_daily'  #specifies which data we are using to predict. Can also try the bitstamp data


### Define Model Inputs ###
input_size=4 # no of features
# Define Prediction variable
pred_var= 'high'
#Define conditioning variables
all_vars= ['low', 'year', 'month_sine', 'month_cosine', 'eth', 'gold'] #add weekday
low=True
year=False
eth=True
gold= True
#Encoding month with sine and cosine value instead of labels
month_sine=False
month_cosine=False
assert sum([month_sine, month_cosine])== 0 or 2,  "Month sine and cosine should be used together"

#add selected variables to dictionary
cond_vars_dict = dict(((k, eval(k)) for k in all_vars))

cond_vars_selected = {k: v for k, v in cond_vars_dict.items() if v is not False}
no_cond_vars=(sum(value == True for value in cond_vars_dict.values()))
assert no_cond_vars == input_size-1, "select the correct no of variables"

print(cond_vars_dict)

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

