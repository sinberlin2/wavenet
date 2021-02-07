import matplotlib.pyplot as plt
import numpy as np
# import torch.nn as nn
# import torch
import pandas as pd

def loss_function(y_true,y_pred):
    loss= np.sqrt(np.mean(((y_pred-y_true)**2)))
    return loss


def get_u2_value(y_pred, y_true, y_prev):  #x is y_pred

    y_true=np.squeeze(y_true)
    y_pred=np.squeeze(y_pred)
    y_prev=np.squeeze(y_prev)

    nom1 =np.subtract(y_pred, y_true)
    nom1 =np.square(nom1)
    nom2= np.square(y_prev)

    nom = np.divide(nom1 ,nom2, out=np.zeros_like(nom1), where=nom2!=0)
    nom = nom.mean()
    nom = np.sqrt(nom)

    denom1= np.subtract(y_true, y_prev)
    denom1 = np.square(denom1)
    denom2= np.square(y_prev)
    denom = np.divide(denom1,denom2,out=np.zeros_like(denom1), where=denom2!=0)

    denom= denom.mean()
    denom = np.sqrt(denom)

    u2_value= nom/denom

    return u2_value


def get_e_value(y_pred, y_true):  #x is y_pred

    y_true=np.squeeze(y_true)
    y_pred=np.squeeze(y_pred)
    y_true_avg= np.mean(y_true)

    nom =np.subtract(y_true, y_pred)
    nom =np.square(nom)
    nom= np.sum(nom)

    denom = np.subtract(y_true, y_true_avg)
    denom = np.square(denom)
    denom = np.sum(denom)

    value = np.divide(nom,denom,out=np.zeros_like(nom), where=denom!=0)
    e_value= 1-value
    return e_value

def plot_train_val_loss(history, epochs, results_folder ):
    train_losses= history.history['loss']
    val_losses= history.history['val_loss']
    # plot history
    plt.plot(history.history['loss'], label='train')
    plt.plot(history.history['val_loss'], label='val')
    plt.legend()
    fig_path = results_folder  + 'train_val_loss'
    plt.savefig(fig_path)
   # plt.show()
    plt.close()

    total_train_loss = float((sum(train_losses))/epochs)
    print("total_training_loss: %f" % total_train_loss)
    total_val_loss = float((sum(val_losses))/epochs)
    print("total_val_loss: %f" % total_val_loss)

#whole sequence and sliding window
def plot_predictions(model_type, pred_var, test_data, naive_preds, results_folder, all_preds, u2_value, predict_size, prediction_day, len_preds='all', sliding_window=True):
    prediction_day= 1 # first predicted day (only relevant if we make multiple step ahead predictions)

    actual_predictions = all_preds[prediction_day - 1, :] #select the first prediction (in this case all)
    #actual_predictions = np.squeeze(actual_predictions)

    naive_preds= np.squeeze(naive_preds)
    if len_preds is 'all':
        len_preds= actual_predictions.shape[0]

    test_data = test_data[prediction_day - 1, :]
    # plot the last test  data
    # pd.DataFrame(actual_predictions).to_csv(results_folder + "preds.csv")
    # pd.DataFrame(test_data).to_csv( results_folder + 'test.csv')

    test_x_len = test_data.shape[0]
    test_x_start = test_x_len -len_preds
    plt.plot(test_data, label='test data')
    #plt.plot(test_data[test_x_start: ], label='test data') #


    # plot predictions
    pred_delay=prediction_day - 1 # in case we are not plotting the first prediction day (with multiple pred steps)
    x = np.arange( pred_delay ,  pred_delay +len_preds, 1)
    plt.plot(x, actual_predictions, label='model predictions') #[-len_preds:]

    if sliding_window is False:
        sliding_window= 'WS'
    else:
        sliding_window= 'SL'

    u2_value= str(round(u2_value, 4))
    plt.title('U2 value: ' + u2_value , fontsize=10)
    plt.ylabel(pred_var)
    plt.xlabel('Days')  #adjust
    plt.grid(False)
    plt.autoscale(axis='x', tight=True)
    plt.legend(loc='upper left', frameon=False)
    fig_path = results_folder + "{}_{}_Test_predictions_{}{}_pred_size_{}_pred_len_{}_u2_{}".format(model_type, sliding_window, pred_var, prediction_day, predict_size, len_preds, u2_value.split('.')[1])
    plt.savefig(fig_path)
    plt.show()
    plt.close('all')


