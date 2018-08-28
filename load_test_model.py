# coding: utf-8

import pandas as pd


pd.set_option("display.max_colwidth", 1000)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

import numpy as np

np.set_printoptions(threshold=np.inf)

import seaborn as sns
import matplotlib
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend("agg")
import oandapy
import configparser
import datetime
import pytz
from datetime import datetime, timedelta

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout
from keras.models import model_from_json

from sklearn.preprocessing import MinMaxScaler

import json

from get_indicator import getBollingerWrapper
from mysql_connector import MysqlConnector

def build_model(inputs, output_size, neurons, activ_func="linear", dropout=0.25, loss="mae", optimizer="adam"):
    model = Sequential()

    model.add(LSTM(neurons, input_shape=(inputs.shape[1], inputs.shape[2])))
    model.add(Dropout(dropout))
    model.add(Dense(units=output_size))
    model.add(Activation(activ_func))
    model.compile(loss=loss, optimizer=optimizer)

    return model


def iso_jp(iso):
    date = None
    try:
        date = datetime.strptime(iso, '%Y-%m-%dT%H:%M:%S.%fZ')
        date = pytz.utc.localize(date).astimezone(pytz.timezone("Asia/Tokyo"))
    except ValueError:
        try:
            date = datetime.strptime(iso, '%Y-%m-%dT%H:%M:%S.%f%z')
            date = dt.astimezone(pytz.timezone("Asia/Tokyo"))
        except ValueError:
            pass
    return date
 
def date_string(date):
    if date is None:
        return ''
    return date.strftime('%Y/%m/%d %H:%M:%S')


# 必要なデータを取得してDataFrameに突っ込んで返す
def getDataSet(base_time, con, window_size, learning_span, output_train_index):
    ### get daily dataset

    length = learning_span + output_train_index

    instrument = "GBP_JPY"
    target_time = base_time - timedelta(days=1)
    sql = "select max_price, min_price, start_price, end_price, insert_time from %s_%s_TABLE where insert_time < \'%s\' order by insert_time desc limit %s" % (instrument, "day", target_time, length) 
    response = con.select_sql(sql)
    #print(sql)

    max_price_list = []
    min_price_list = []
    start_price_list = []
    end_price_list = []
    time_list = []
    for res in response:
        max_price_list.append(res[0])
        min_price_list.append(res[1])
        start_price_list.append(res[2])
        end_price_list.append(res[3])
        time_list.append(res[4])

    max_price_list.reverse()
    min_price_list.reverse()
    start_price_list.reverse()
    end_price_list.reverse()
    time_list.reverse()
    
    sma1d20_list = []
    i = 0
    while len(end_price_list) != len(sma1d20_list):
        tmp_time = target_time - timedelta(days=i)
        dataset = getBollingerWrapper(tmp_time, instrument,  table_type="day", window_size=20, connector=con, sigma_valiable=2, length=0)
        try:
            sma1d20_list.append(dataset["base_lines"][-1])
        except Exception as e:
            pass

        i = i + 1

        #print(len(end_price_list))
        #print(len(sma1d20_list))

    sma1d20_list.reverse()

    original_dataset = {"end": end_price_list,
               "start": start_price_list,
               "max": max_price_list,
               "min": min_price_list,
               "sma1d20": sma1d20_list,
               "time": time_list}

    value_dataset = {"end": end_price_list,
               "start": start_price_list,
               "max": max_price_list,
               "min": min_price_list,
               "sma1d20": sma1d20_list}

    return original_dataset, value_dataset

def change_to_normalization(dataset):
    tmp_df = pd.DataFrame(dataset)
    np_list = np.array(tmp_df)
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalization_list = scaler.fit_transform(np_list)

    return normalization_list

 # 引数で与えられたndarrayをwindow_sizeで分割して返す(ndarray)
def createDataset(dataset, original_dataset, window_size, learning_span, output_train_index, output_flag):
    input_train_data = []
    output_train_data = []
    time_list = []
    for i in range(0, learning_span-(window_size+1)):

        temp = []
        temp_index = 0
        for k in range(i, i+window_size):
            temp.append(dataset[k].copy())
            temp_index = k

        input_train_data.append(temp)
        try:
            if output_flag:
                output_train_data.append(dataset[temp_index+output_train_index][0].copy())
                time_list.append(original_dataset["time"][temp_index+output_train_index])
                  
            else:
                pass

        except Exception as e:
            pass

    input_train_data = np.array(input_train_data)
    output_train_data = np.array(output_train_data)
    time_list = np.array(time_list)

    return input_train_data, output_train_data, time_list


def change_to_ptime(base_time):
    base_time = datetime.strptime(base_time, "%Y-%m-%d %H:%M:%S")
    return base_time


def join_dataframe(sdf, ddf):
    index = 0
    for col in ddf:
        sdf = sdf.rename(columns={index: col})
        index = index + 1
    
    ddf = ddf.append(sdf, ignore_index=True)

    return ddf

connector = MysqlConnector()
original_dataset, value_dataset = getDataSet(train_base_time, connector, window_size=30, learning_span=300, output_train_index=1)

# 以降テスト

# testデータの正規化のために、最大値と最小値を取得しておく
max_list = []
min_list = []
max_price = max(original_dataset["end"])
min_price = min(original_dataset["end"])

for col in original_dataset:
    max_list.append(max(original_dataset[col]))
    min_list.append(min(original_dataset[col]))

max_list = pd.DataFrame(max_list)
min_list = pd.DataFrame(min_list)

# あとで行追加するので転置しておく
max_list = max_list.T
min_list = min_list.T

test_base_time = change_to_ptime(base_time="2018-07-31 00:00:00")
test_original_dataset, test_value_dataset = getDataSet(test_base_time, connector, window_size=30, learning_span=0, output_train_index=0)

# 訓練データの最大、最小値を追加して、正規化する
# 正規化後はdropする
tmp = test_value_dataset.copy()
tmp = pd.DataFrame(tmp)

tmp = join_dataframe(max_list, tmp)
tmp = join_dataframe(min_list, tmp)

test_value_dataset = change_to_normalization(tmp)
test_value_dataset = pd.DataFrame(test_value_dataset)
test_value_dataset = test_value_dataset.iloc[:-2]
test_value_dataset = test_value_dataset.values

# shape数を揃えるためにappendする
input_test_data = []
input_test_data.append(test_value_dataset)
input_test_data = np.array(input_test_data)

model_filename = "model.json"
weights_filename = "model_weights.hdf5"

model = model_from_json(model_filename)
model.load_weights(weights_filename)

predict = model.predict(input_test_data)

print((predict[0][0]*(max_price-min_price))+min_price)

#sql = "select end_price, insert_time from GBP_JPY_day_TABLE where insert_time < \'2018-08-01 00:00:00\' order by insert_time desc limit 2"

#response = connector.select_sql(sql)
#end_price_list = []
#end_time_list = []
#for res in response:
#    end_price_list.append(res[0])
#    end_time_list.append(res[1])
#
#end_price_list.reverse()
#end_time_list.reverse()


#train_predict = scaler.inverse_transform(train_predict)
#print(train_predict)

#file = open("result.txt", "w")
#file.write(numpy_list)
#file.write("\n==============================================\n")
#file.write(normalization_list)

#file.close()
#numpy_pd = pd.DataFrame(numpy_list)
#normalization_pd = pd.DataFrame(normalization_list)
#print(numpy_pd)
#print(normalization_pd)

#numpy_list = df.values
#print numpy_list
