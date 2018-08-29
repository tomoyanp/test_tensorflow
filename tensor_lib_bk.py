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
def getDataset(base_time, con, window_size, learning_span, output_train_index, table_layout):
    ### get daily dataset

    length = learning_span + output_train_index

    instrument = "GBP_JPY"
    if table_layout == "day":
        target_time = base_time - timedelta(days=1)
    elif table_layout == "1h":
        target_time = base_time - timedelta(hours=1)
    elif table_layout == "5m":
        target_time = base_time - timedelta(minuites=5)
    elif table_layout == "1m":
        target_time = base_time - timedelta(minuites=1)

    sql = "select max_price, min_price, start_price, end_price, insert_time from %s_%s_TABLE where insert_time < \'%s\' order by insert_time desc limit %s" % (instrument, table_layout, target_time, length) 
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
        if table_layout == "day":
            tmp_time = target_time - timedelta(days=i)
        elif table_layout == "1h":
            tmp_time = target_time - timedelta(hours=i)
        elif table_layout == "5m":
            tmp_time = target_time - timedelta(minuites=(i*5))
        elif table_layout == "1m":
            tmp_time = target_time - timedelta(minuites=i)

        dataset = getBollingerWrapper(tmp_time, instrument,  table_type=table_layout, window_size=20, connector=con, sigma_valiable=2, length=0)
        try:
            sma1d20_list.append(dataset["base_lines"][-1])
        except Exception as e:
            pass

        i = i + 1


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
def createTrainDataset(dataset, original_dataset, window_size, learning_span, output_train_index):
    input_train_data = []
    output_train_data = []
    time_list = []
    for i in range(0, (learning_span-window_size)):
        temp = dataset[i:i+window_size].copy()
        input_train_data.append(temp)
        
    output_train_data = dataset[window_size+output_train_index:,0].copy()
    print(output_train_data)
    time_list = original_dataset["time"][window_size+output_train_index:].copy()
    print(output_train_data.shape)

    input_train_data = np.array(input_train_data)
    output_train_data = np.array(output_train_data)
    time_list = np.array(time_list)

    print(input_train_data.shape)

    return input_train_data, output_train_data, time_list

def createTestDataset(dataset, window_size, learning_span):
    input_test_data = []
    for i in range(0, (learning_span-window_size+1)):
        temp = dataset[i:i+window_size].copy()
        input_test_data.append(temp)
        
    input_test_data = np.array(input_test_data)

    return input_test_data


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

