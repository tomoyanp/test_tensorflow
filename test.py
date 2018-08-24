# coding: utf-8

import pandas as pd


pd.set_option("display.max_colwidth", 1000)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

import numpy as np

np.set_printoptions(threshold=np.inf)

import seaborn as sns
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import oandapy
import configparser
import datetime
import pytz
from datetime import datetime, timedelta

from keras.models import Sequential
from keras.layers import Activation, Dense
from keras.layers import LSTM
from keras.layers import Dropout

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


def getDataSet(base_time, con, window_size, learning_span, output_train_index):
    ### get daily dataset

    length = learning_span + output_train_index

    instrument = "GBP_JPY"
    target_time = base_time - timedelta(days=1)
    sql = "select max_price, min_price, start_price, end_price, insert_time from %s_%s_TABLE where insert_time < \'%s\' order by insert_time desc limit %s" % (instrument, "day", target_time, length) 
    response = con.select_sql(sql)
    print(sql)

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

        print(len(end_price_list))
        print(len(sma1d20_list))

    sma1d20_list.reverse()

    dataset = {"end": end_price_list,
               "start": start_price_list,
               "max": max_price_list,
               "min": min_price_list,
               "sma1d20": sma1d20_list,
               "time": time_list}


    df = pd.DataFrame([])
    df["end"] = end_price_list
    df["time"] = time_list
    df["max"] = max_price_list
    df["min"] = min_price_list
    df["start"] = start_price_list
    df["sma1d20"] = sma1d20_list

    return dataset, df
 

connector = MysqlConnector()
base_time = "2018-08-01 00:00:00"
base_time = datetime.strptime(base_time, "%Y-%m-%d %H:%M:%S")
window_size = 30
learning_span = 300

#window_size = 10
#learning_span = 10
numpy_list = []
normalization_list = []
right_data_list = []

min_list = []
max_list = []
original_list = []
time_list = []


output_train_index = 1
dataset, df = getDataSet(base_time, connector, window_size, learning_span, output_train_index)

del df["time"]
np_list = df.values

scaler = MinMaxScaler(feature_range=(0, 1))
normalization_list = scaler.fit_transform(np_list)
max_price = max(dataset["end"])
min_price = min(dataset["end"])

input_train_data = []
output_train_data = []
time_list = []

for i in range(window_size, learning_span+1):

    temp = []
    temp_index = 0
    for k in range((i-window_size), i):
        temp.append(normalization_list[k].copy())
        print(normalization_list[k])
        temp_index = k
        
        
    input_train_data.append(temp)
    try:
        output_train_data.append(normalization_list[temp_index+output_train_index][0].copy())
        time_list.append(dataset["time"][temp_index+output_train_index])
    except Exception as e:
        pass

#input_train_data = input_train_data[:-1]
input_train_data = np.array(input_train_data)
output_train_data = np.array(output_train_data)
time_list = np.array(time_list)

print(input_train_data)
print(output_train_data)
print(input_train_data.shape)
print(output_train_data.shape)
print(time_list.shape)

np.random.seed(202)
model = build_model(input_train_data, output_size=1, neurons=20)
history = model.fit(input_train_data, output_train_data, epochs=50, batch_size=1, verbose=2, shuffle=True)

train_predict = model.predict(input_train_data)

paint_predict = []
paint_right = []
for i in range(len(train_predict)):
    print(time_list[i])
    paint_predict.append((train_predict[i]*(max_price-min_price))+min_price)
    print((train_predict[i]*(max_price-min_price))+min_price)
    paint_right.append((output_train_data[i]*(max_price-min_price))+min_price)
    print((output_train_data[i]*(max_price-min_price))+min_price)

### paint predict train data
fig, ax1 = plt.subplots(1,1)
ax1.plot(time_list, paint_predict, label="Predict", color="blue")
ax1.plot(time_list, paint_right, label="Actual", color="red")

plt.savefig('figure.png')
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
