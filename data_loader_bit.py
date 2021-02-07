
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import pytz


#Adjust this function to the relevant dataset
def read_bitcoin_data(base_path, data_folder, subfolder, file_name, var):
    dataset = pd.read_csv(base_path + data_folder + subfolder + "/" + file_name + '.csv', header=1, low_memory=False, usecols=['date', var])
    dataset.index=  pd.to_datetime(dataset.date, format= '%Y-%m-%d %H:%M:%S')
    dataset= dataset.drop(columns=['date'])
    print('No of NA values:', dataset.isna().sum())
    dataset = dataset.dropna(axis=0)
    dataset = dataset.sort_index()
    print(len(dataset))
    dataset= dataset[-1000:]

    return dataset

def read_gold_data(base_path, data_folder, subfolder, file_name, var):
    dataset = pd.read_csv(base_path + data_folder + subfolder + "/" + file_name + '.csv', header=0, low_memory=False, usecols=['Date', var])
    print(dataset)

    dataset.index=  pd.to_datetime(dataset.Date, format= '%Y-%m-%d')
    print(dataset)
    #dataset.index= pd.to_datetime(dataset.index.dt.strftime('%Y-%m-%d %H:%M:%S'))
    dataset= dataset.drop(columns=['Date'])
    print('No of NA values:', dataset.isna().sum())
    dataset = dataset.dropna(axis=0)
    dataset = dataset.sort_index()
    print(dataset)
    dataset= dataset[-1000:]

    return dataset


#define a conversion function for the native timestamps in the csv file
def dateparse (time_in_secs):
    return pytz.utc.localize(datetime.datetime.fromtimestamp(float(time_in_secs)))


def read_kaggle(base_path, data_folder, subfolder, file_name, var):
    data = pd.read_csv('../input/coinbaseUSD_1-min_data_2014-12-01_to_2018-11-11.csv', parse_dates=[0],
                       date_parser=dateparse)
    data['Timestamp'] = data['Timestamp'].dt.tz_localize(None)
    data = data.groupby([pd.Grouper(key='Timestamp', freq='H')]).first().reset_index()
    data = data.set_index('Timestamp')
    data = data[['Weighted_Price']]
    data['Weighted_Price'].fillna(method='ffill', inplace=True)
    #dataset= dataset[-2000:]

    return dataset


def combine_data(main_df, cond_df):
    cols = list(main_df) + list(cond_df)
    dataset = main_df.merge(cond_df,  left_index=True, right_index=True)
    dataset.columns = cols

    return dataset



class DataLoader(object):
    def __init__(self, pred_var, sliding_window, predict_size, input_size, base_path, data_folder, sub_folder, cond_vars, scaler_vars):
        """
        :param xs:
        :param ys:
        :param batch_size:
        """

        self.tw= sliding_window
        self.predict_size= predict_size
        self.input_size= input_size
        self.pred_var= pred_var
        self.sliding_window=sliding_window

        self.base_path = base_path
        self.data_folder = data_folder
        self.sub_folder = sub_folder
        if self.sub_folder == 'binance_bitcoin_minute':
            self.bitcoin_data_file = 'Binance_BTCUSDT_minute'
            self.eth_data_file =  'Binance_ETHUSDT_minute'
        if self.sub_folder == 'binance_bitcoin_daily':
            self.bitcoin_data_file = 'Binance_BTCUSDT_d'
            self.eth_data_file =  'Binance_ETHUSDT_d'
            self.gold_data_file ='LBMA-GOLD'

        self.cond_vars = {k: v for k, v in cond_vars.items() if v is not False}

        #Specify which variables should have a scaler (apart from prediction variable which has a scaler)
        self.scaler=MinMaxScaler(feature_range=(0, 1))
        scaler_vars=['low', 'year' , 'eth', 'gold']
        self.scaler_low = MinMaxScaler(feature_range=(0, 1))
        self.scaler_year = MinMaxScaler(feature_range=(0, 1))
        self.scaler_eth = MinMaxScaler(feature_range=(0, 1))
        self.scaler_gold = MinMaxScaler(feature_range=(0, 1))
        self.scalers = list()
        self.scaler_index =dict()
        self.scaler_index['self.scaler']=0
        self.scaler_name=['self.scaler']

        for key in self.cond_vars:
            if key in scaler_vars:
                self.scaler_name.append('self.scaler_'+ key)
                self.scaler_index['self.scaler_'+ key] = list(self.cond_vars).index(key) + 1

        super(DataLoader, self).__init__()  #is this necessary?

    def load_data(self):
        # load the dataset
        if self.sub_folder in ['binance_bitcoin_minute', 'binance_bitcoin_daily']:
            dataset = read_bitcoin_data(self.base_path, self.data_folder, self.sub_folder, self.bitcoin_data_file, self.pred_var)
            if 'low' in self.cond_vars.keys():
                print('low is true')
                add_data= read_bitcoin_data(self.base_path, self.data_folder, self.sub_folder, self.bitcoin_data_file, 'low')
                dataset=combine_data(dataset, add_data)
            if 'eth' in self.cond_vars.keys():
                add_data = read_bitcoin_data(self.base_path, self.data_folder, self.sub_folder, self.eth_data_file , 'high')
                dataset=combine_data(dataset, add_data)
            if 'gold' in self.cond_vars.keys():
                add_data = read_gold_data(self.base_path, self.data_folder, self.sub_folder, self.gold_data_file , 'USD (AM)')
                dataset=combine_data(dataset, add_data)
                print(dataset)

        else:
            print('Enter valid prediction time series')


        if 'month_sine' in self.cond_vars.keys() and 'month_cosine' in self.cond_vars.keys():
            #get values or one hot encoded months
            month_dummies= pd.DataFrame(dataset.index.month, index=dataset.index)
            month_dummies.columns=['month']
            dataset= dataset.join(month_dummies, how='left')
            #get sine and cosine of years
            dataset['sin_time'] = np.sin(2 * np.pi * dataset.month / 12)
            dataset['cos_time'] = np.cos(2 * np.pi * dataset.month  / 12)
            dataset=dataset.drop(columns=['month'])

        if 'year' in self.cond_vars.keys():
            #get year dummies or values
            year_dummies= pd.DataFrame(dataset.index.year)
            #get years since present
            end= dataset.index[-1]
            dates= dataset.index
            time_since =  end - dates  # calculate timedelta
            years_since = time_since.map(lambda x: round(x.days/365)) # get no of days
            dataset['years']= years_since.tolist()

        dataset = pd.DataFrame(dataset)
        dataset = dataset.astype('float64')
        print(len(dataset),'length of dataset')

        return dataset

    def split_data(self, dataset):
        # split data
        train_data_end = round(len(dataset) * 0.6)
        val_data_end = round(len(dataset) * 0.8)
        train_data = dataset[:train_data_end]
        val_data = dataset[train_data_end:val_data_end]
        test_data_end= len(dataset)
        test_data = dataset[val_data_end:test_data_end]

        return train_data, val_data, test_data

    def scale_data(self, data):
        print(data.shape, 'inout shape')  #expects B,S,F
        print(self.scaler_index)
        scaler_no=0
        for i in range(data.shape[1]):
            if i in self.scaler_index.values():
                data_scaled =eval(self.scaler_name[scaler_no]).transform(data[:, self.scaler_index[self.scaler_name[scaler_no]]].reshape(-1, 1))
                output =  data_scaled
                scaler_no+=1
            else:
                data_scaled=data[:,i].reshape(-1, 1)
                output = np.hstack((output, data_scaled))  # hstack means stack along second axis

        return output

    def create_inout_sequences(self, input_data, tw, predict_size):
        inout_seq = []
        L = len(input_data)
        x = []
        y = []

        for i in range(L - tw):
            train_seq = input_data[i:i + tw]
            train_label = input_data[
                          i + tw:i + tw + predict_size]
            # for conditional model only keep the target variable values
            train_label = train_label[:, 0]
            inout_seq.append((train_seq, train_label))  # tuple containing the 7 days values and the next value
            if len(train_label) == predict_size:
                x.append(train_seq)
                y.append(train_label)
        x = np.array(x)
        y = np.array(y)

        return x, y

    def create_wavenet_inout_sequences(self, input_data_normalised):
        x_len = round(input_data_normalised.shape[0]) - 1  # - self.predict_size
        data_x, data_y = input_data_normalised[:x_len], input_data_normalised[1:,0:1]  # keep only prediction variable for y
        data_x = np.expand_dims(data_x, axis=0)
        data_y = np.expand_dims(data_y, axis=0)
        print(data_x.shape)
        return data_x, data_y

    def split_scale_transform(self, dataset):

        train_data, val_data, test_data = self.split_data(dataset)
        train_data = train_data.to_numpy()
        val_data = val_data.to_numpy()
        test_data = test_data.to_numpy()

        #fit scalers
        scaler_no = 0
        for i in range(dataset.shape[-1]):
            if i in self.scaler_index.values():   #if 0 is 0
                (eval(self.scaler_name[scaler_no])).fit(train_data[:, self.scaler_index[self.scaler_name[scaler_no]]].reshape(-1, 1))

                scaler_no += 1
            else:
                continue

        #scale data with fitted scalers
        train_data_normalized = self.scale_data(train_data)  #shape [samples, timesteps , features]
        val_data_normalized = self.scale_data(val_data)
        test_data_normalized = self.scale_data(test_data)

        if self.sliding_window is not False: # reshape into X=t->t+tw and Y=t+tw+predict_size
            print('sliding window method used', self.tw)
            train_x, train_y = self.create_inout_sequences(train_data_normalized, self.tw, self.predict_size)
            val_x, val_y = self.create_inout_sequences(val_data_normalized, self.tw, self.predict_size)
            test_x, test_y = self.create_inout_sequences(test_data_normalized, self.tw, self.predict_size)

        else:
            print('whole sequence wavenet method used')
            train_x, train_y = self.create_wavenet_inout_sequences(train_data_normalized)
            val_x, val_y = self.create_wavenet_inout_sequences(val_data_normalized)
            test_x, test_y = self.create_wavenet_inout_sequences(test_data_normalized)

        return  train_x, train_y, val_x, val_y, test_x, test_y

    def scale_back(self, data):
        # this expects a 2 dim input
        scaler_no=0
        for i in range(data.shape[2]):
            if i == 0:
                data_scaled = ((eval(self.scaler_name[scaler_no])).inverse_transform(
                    data[:,:, self.scaler_index[self.scaler_name[scaler_no]]].reshape(-1, 1)))
                output = data_scaled
                scaler_no += 1
            elif i not in self.scaler_index.values():
                data_scaled = data[:, :,i].reshape(-1, 1)
                output = np.hstack((output, data_scaled))  # hstack means stack along second axis
            else:
                data_scaled = (eval(self.scaler_name[scaler_no]).inverse_transform(
                    data[:, :,self.scaler_index[self.scaler_name[scaler_no]]].reshape(-1, 1)))
                output = np.hstack((output, data_scaled))
                scaler_no += 1

        return output