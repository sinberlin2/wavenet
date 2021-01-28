
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow import keras
from tensorflow.keras import layers
from keras import backend as K


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))



def lstm_model(hidden_layer_size, learning_rate, dropout, input_size):


    # design network
    model = keras.Sequential()
    model.add(keras.Input(shape=(None, input_size)))
    model.add(layers.LSTM(hidden_layer_size, return_sequences=True, activation="tanh"))  #try relu
    model.add(layers.LSTM(hidden_layer_size, activation="tanh"))  #input shape doesnt have to be specified but return sequences has to be true
    model.add(layers.Dropout(dropout))
    #model.add(layers.TimeDistributed(layers.Dense(1))) #for multiple outputs, return_sequences needs to be true then
    model.add(layers.Dense(1))
    model.compile(optimizer= 'adam' , loss=root_mean_squared_error) #'adam'  SGD(lr=0.0001, momentum=0.9)

    print(model.summary())

    return model
#
# def lstm_model_new(hidden_layer_size, learning_rate, dropout, input_size):
#     # design network
#     history_seq = Input(shape=(None, input_size))
#     x = Lambda(lambda x: x[:, :, 0:1])(history_seq)  # this selects only the first one
#
#     if input_size > 1:
#         c = Lambda(lambda x: x[:, :, 1:])(history_seq)
#
#     model.add(layers.LSTM(hidden_layer_size, return_sequences=True, activation="tanh"))
#     model.add(layers.LSTM(hidden_layer_size, activation="tanh"))  #input shape doesnt have to be specified but return sequences has to be true
#     #model.add(layers.Dropout(0.2))
#     model.add(layers.Dense(input_size))  #try to see if i can also do multiple outputs from here, do I need a timedistributed layer
#     model.compile(optimizer= 'adam' , loss=root_mean_squared_error) #'adam'  SGD(lr=0.0001, momentum=0.9)
#     #model.build(input_shape=(None, None, input_size))
#
#     return model
