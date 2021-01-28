from __future__ import print_function, division
import numpy as np
from keras.callbacks import EarlyStopping
# fix random seed for reproducibility
np.random.seed(7)

from wavenet_model_keras import wavenet_model
from lstm_model_keras import lstm_model
from data_loader_bit import DataLoader
from configs_bit import *

def main():
    data_loader= DataLoader(pred_var, sliding_window, predict_size, input_size, base_path, data_folder, sub_folder, cond_vars_dict)
    dataset=data_loader.load_data()
    train_x, train_y, val_x, val_y, test_x, test_y = data_loader.split_scale_transform(dataset)

    def train_model(model_type, train_x, train_y, val_x, val_y, output_size, epochs, batch_size):

        # timeseries input is 1-D numpy array, orecast_size is the forecast horizon
        if model_type == 'WaveNet':
            model= wavenet_model(predict_size, learning_rate= lr, num_layers=num_layers, stacks= stacks, dropout=dropout, input_size= input_size, n_filters= n_filters)
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=50, min_delta=0.002)

        elif model_type == 'LSTM':
            model = lstm_model(hidden_layer_size, lr, dropout, input_size)
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=2, patience=10, min_delta=0.005)
        history = model.fit(train_x, train_y , validation_data=(val_x, val_y),
                        batch_size= batch_size ,
                        epochs=epochs, callbacks=[es])

        print('\n\nModel with input size {}, output size {}'.
                                    format(model.input_shape, model.output_shape))

        return model, history

    if load_model==False:
        print(epochs)
        model, history= train_model(model_type, train_x, train_y, val_x, val_y, output_size, epochs=epochs, batch_size=batch_size)
        print('Done training.')
        print(history)
        plot_train_val_loss(history, epochs, base_path, results_folder)


    # # save model in results folder (can load it from there more easily)
    if save_model == True:

        model_path =  "{}{}{}_tw_{}_nfil{}_b_{}_dil_{}.h5".format(base_path, results_folder , model_type , sliding_window, n_filters, batch_size, num_layers)
        model.save(model_path)
        print('Model saved in ' + model_path)


    #Generate Predictions, evaluate model
    yhat = model.predict(test_x)
    naive_preds = test_x[:, :, 0:1]

    if sliding_window is not False:
        print(yhat.shape)
        if yhat.ndim>2:
            yhat=yhat[:,-1,:]
        yhat = np.expand_dims(yhat, axis=2)
        test_y = np.expand_dims(test_y, axis=2)
        naive_preds = naive_preds[:, -1, :]
        naive_preds = np.expand_dims(naive_preds, axis=2)

    rmse = loss_function(test_y, yhat)
    print("test_loss_scaled: %f" % rmse)

    #Evaluate Test Loss on rescaled data, expects [batch_size, len batch sample, no_features]
    rmse = model.evaluate(test_x, test_y, batch_size=test_x.shape[0])  # batch size should be number of input windows, as we want to evaluate in one go
    print('Test RMSE by evaluation: %.3f' % rmse)

    #Evaluate Loss on rescaled data

    #Scale back data, inverse scaler expects array of shape  [samples, timesteps, features]
    test_preds = data_loader.scale_back(yhat)
    test_y = data_loader.scale_back(test_y)
    naive_preds= data_loader.scale_back(naive_preds)


    # calculate RMSE on inverted data, don't use model evaluate as it is not trained on rescaled data.
    rmse = loss_function(test_y, test_preds)
    print("test_loss: %f" % rmse)
    rmse_naive = loss_function(test_y, naive_preds)
    print("test_loss_naive: %f" % rmse_naive)


    perc_rmse_model_to_naive= 1- (rmse/ rmse_naive)
    print("perc_rmse_model_to_naive: %f" % perc_rmse_model_to_naive)

    u2_value= get_u2_value(test_preds,test_y, naive_preds)
    print("u2_value: %f" % u2_value)
    e_value = get_e_value(test_preds, test_y)
    print("e_value: %f" % e_value)

    # #Plot Predictions
    test_preds = np.expand_dims(test_preds, axis=1)# expects testpreds as [x , pred_size, 1] np array
    plot_predictions(model_type, pred_var, test_y, naive_preds, base_path, results_folder, test_preds, u2_value, perc_rmse_model_to_naive,  test_preds.shape[0],
                                 predict_size ,  len_preds='all', sliding_window=sliding_window)  #predict size for multi-step predictions

    # plot_predictions(model_type, pred_var, test_y, naive_preds, base_path, results_folder, test_preds, u2_value, perc_rmse_model_to_naive,  test_preds.shape[0],
    #                              predict_size,  len_preds=100 , sliding_window=sliding_window)
    #
    # plot_predictions(model_type, pred_var, test_y, naive_preds, base_path, results_folder, test_preds, u2_value, perc_rmse_model_to_naive,  test_preds.shape[0],
    #                              predict_size,  len_preds=250 , sliding_window=sliding_window)





if __name__ == '__main__':
    main()
