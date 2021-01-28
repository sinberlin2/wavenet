Predictions with Wavenet and LSTM

To use with your own dataset, specify:
- paths in configs file.
- possible conditioning variables in configs file
- specify filenames for data files
- adjust read_data function to read in data files in data_loader.py
- define which variables need scaling and initialise scaler

To run model:
-specify which and how many conditioning variables in configs.py
- specify model parameters in model_configs.py
- adjust axes for plotting etc in utils file

Recommendation:
Filter number should be increased with number of dilations or input features.

#dilation=[1,2,4...,512]. receptive field = 1+(1+2+4+...+512) * (2-1)=1+1023=1024
Developed with Keras, Tensorboard, Tensorflow (add versions)

 


