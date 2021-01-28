##Maybe make this the main file

### General Model Parameters ###
output_size= 1 #refers to the number of features to output
lr_decay=0.0
dropout=0.0
load_model=False
save_model=False

#for dataloader
predict_size=1 #predicting 1 day ahead/ only optimising on the last x values

#Use a sliding window of a certain size or not (False or Number)
sliding_window=False
if sliding_window is not False:
    assert predict_size is 1, "prediction size should be 1 for sliding window approach "

### WaveNet Model Parameters ###
model_type = 'WaveNet'
epochs = 300   #300 default
loss_metric = 'RMSE'
lr=0.0000075  # learning rate should be 0.00075 or lower
batch_size=1
num_layers= 3 #num layers with dilation, num_layers 2-10 results in receptive field of 4,8, 16, 64, 128, 256, 512, 1024 with 1 stack
stacks=2
n_filters= 600

#num_layers3, stakcs 2 , n filter 200  with low and eth was 0.99 , better than 1 stack and num_layer 5 (receotive field is 16 so one more than 15)

#
# ### Model Parameters  LSTM ###
# model_type = 'LSTM'
# epochs = 2
# lr=0.0002
# batch_size=64
# hidden_layer_size = 120
#
