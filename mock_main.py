# coding: utf-8

import pandas as pd


pd.set_option("display.max_colwidth", 100)
pd.set_option("display.max_columns", 20)
pd.set_option("display.max_rows", 1000)

import numpy as np
#import seaborn as sns
#import matplotlib.pyplot as plt
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


def getDailyIndicator(base_time, con, span):
    ### get daily dataset
    instrument = "GBP_JPY"
    target_time = base_time - timedelta(days=1)
    sql = "select max_price, min_price, start_price, end_price from %s_%s_TABLE where insert_time < \'%s\' order by insert_time desc limit %s" % (instrument, "day", target_time, span) 
    response = con.select_sql(sql)
    max_price_list = []
    min_price_list = []
    start_price_list = []
    end_price_list = []
    for res in response:
        max_price_list.append(res[0])
        min_price_list.append(res[1])
        start_price_list.append(res[2])
        end_price_list.append(res[3])

    max_price_list.reverse()
    min_price_list.reverse()
    start_price_list.reverse()
    end_price_list.reverse()
    
    sma1d20_list = []
    time_list = []

    for i in range(0, span):
        tmp_time = target_time - timedelta(days=i)
        dataset = getBollingerWrapper(tmp_time, instrument,  table_type="day", window_size=20, connector=con, sigma_valiable=2, length=0)
        sma1d20_list.append(dataset["base_lines"][-1])
        time_list.append(tmp_time)

    sma1d20_list.reverse()
    time_list.reverse()

    df = pd.DataFrame([])

    df["time"] = time_list
    df["max"] = max_price_list
    df["min"] = min_price_list
    df["start"] = start_price_list
    df["end"] = end_price_list
    df["sma1d20"] = sma1d20_list

    return df
    


connector = MysqlConnector()
base_time = "2018-08-01 00:00:00"
base_time = datetime.strptime(base_time, "%Y-%m-%d %H:%M:%S")
window_size = 30
learning_span = 10

numpy_list = []
normalization_list = []
scaler = MinMaxScaler(feature_range=(0, 1))
for i in range(0, learning_span):
    tmp_time = base_time - timedelta(days=i)
    df = getDailyIndicator(tmp_time, connector, window_size)
    normalization_tmp = df.copy()
    tmp = df.copy()
    del normalization_tmp["time"]
    #normalization_tmp = normalization_tmp * 10000
    #print(type(normalization_tmp))
    #print(type(tmp))

    tmp = tmp.values
    normalization_tmp = scaler.fit_transform(normalization_tmp)

    #print(tmp)
    numpy_list.append(tmp)
    normalization_list.append(normalization_tmp)

numpy_list = np.array(numpy_list)
print(numpy_list)
normalization_list = np.array(normalization_list)
print(normalization_list.shape)

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
