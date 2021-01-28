from keras.models import Model
from keras.layers import Input, Conv1D, Dense, Activation, Dropout, Lambda, Multiply, Add, Concatenate, Reshape, TimeDistributed
from keras.optimizers import Adam
from keras import backend as K
from keras import regularizers


def root_mean_squared_error(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))



#import predict size
def wavenet_model(predict_size, learning_rate, num_layers, stacks, dropout, input_size, n_filters):

    # convolutional operation parameters
    filter_width = 2 # considers the relation between two data points
    n_filters  # capture X different properties of the data
    cond_in_channels=  n_filters *(input_size-1) #keep
    residual_channels= n_filters  *2 # has to be the same as gate channels ,for convolution at the beginning of residual block
    gate_channels= n_filters *2
    skip_channels= n_filters  #no of filters at the skip connection
    out_channels= n_filters  # no of filters a the penultimate convolution, last convolution is 1x1
    dilation_rates = [2 ** i for i in range(num_layers)]

    #input is a sort of input layer
    # define an input history series and pass it through a stack of dilated causal convolution blocks.
    initialiser= 'glorot_uniform'  #or he_uniform for kaimain intialisation


    history_seq = Input(shape=(None, input_size)) #


    x = Lambda(lambda x: x[:, :, 0:1] )(history_seq)  #this selects only the first feature

  ##  x = Conv1D(x_in_channels, 1, kernel_initializer=initialiser, padding='same', activation='relu')(x)  #not used

    if input_size>1:
        c=  Lambda(lambda x: x[:, :, 1:])(history_seq)
        c_conv = Conv1D(cond_in_channels, 1, kernel_initializer=initialiser, padding='same', activation='relu')(c)  # convolution on input conditional input, sort of pre-processing

    skips = []
    for dilation_rate in dilation_rates:

        # preprocessing - equivalent to time-distributed dense
        x = Conv1D(residual_channels, 1, kernel_initializer= initialiser,padding='same', activation='relu')(x)

        #no of filters here is g_channels
        x_dil_conv = Conv1D(filters=gate_channels,
                     kernel_size=filter_width,kernel_initializer= initialiser,
                    padding='causal',
                     dilation_rate=dilation_rate)(x)

        if  input_size >1 and dilation_rate==1:

            c_conv = Conv1D(gate_channels, filter_width, padding='causal')(c_conv)  #g -gates
            tanh_gate=   Add()([x_dil_conv, c_conv])
            sigm_gate = Add()([x_dil_conv, c_conv])

        else:
            tanh_gate = x_dil_conv
            sigm_gate = x_dil_conv


        #multiply filter and gating branches
        z = Multiply()([Activation('tanh')(tanh_gate),
                        Activation('sigmoid')(sigm_gate)])

        # postprocessing - equivalent to time-distributed dense
        s = Conv1D(skip_channels, 1, padding='same', kernel_initializer= initialiser,activation='relu')(z)
        z = Conv1D(residual_channels, 1, padding='same',kernel_initializer= initialiser, activation='relu')(z)

    # residual connection used as input in next dilation
        x = Add()([x_dil_conv, z])

        # collect skip connections for final output
        skips.append(s)

    # add all skip connection outputs
    out = Activation('relu')(Add()(skips))

    # final time-distributed dense layers
    out = Conv1D(out_channels, 3, kernel_initializer= initialiser,padding='causal')(out) #is kernel size 1 in traditional setup #was 3
    out = Activation('relu')(out)

    out = Activation('relu')(out)
    out = Dropout(dropout)(out)

    out = Conv1D(1, 1, kernel_initializer= initialiser, padding='same')(out)  #out shape [B, S, F]  same no of samples as training seq


    pred_seq_train = Lambda(lambda x: x[:, :, :])(out)
    receptive_field_size=  ((sum(dilation_rates) * stacks) * (filter_width - 1)) + 1
    print('Receptive fiels is : ', receptive_field_size)

    model = Model(history_seq, pred_seq_train)  #input, output
    model.compile(Adam(learning_rate=learning_rate, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0),
                  loss=root_mean_squared_error)

    return model

