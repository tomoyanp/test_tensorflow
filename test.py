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
def getDataSet(base_time, con, window_size, learning_span, output_train_index, table_layout):
    ### get daily dataset

    length = (learning_span * window_size) + output_train_index

    instrument = "GBP_JPY"
    target_time = base_time - timedelta(days=1)
    sql = "select max_price, min_price, start_price, end_price, insert_time from %s_%s_TABLE where insert_time < \'%s\' order by insert_time desc limit %s" % (instrument, table_layout, target_time, length)
    response = con.select_sql(sql)

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
    
    original_dataset = {"end": end_price_list,
               "start": start_price_list,
               "max": max_price_list,
               "min": min_price_list,
               "time": time_list}

    original_dataset = pd.DataFrame(original_dataset)
    return original_dataset

def change_to_normalization(dataset):
    tmp_df = pd.DataFrame(dataset)
    np_list = np.array(tmp_df)
    scaler = MinMaxScaler(feature_range=(0, 1))
    normalization_list = scaler.fit_transform(np_list)

    return normalization_list

 # 引数で与えられたndarrayをwindow_sizeで分割して返す(ndarray)
def createInputDataset(value_dataset, window_size, learning_span):
    input_train_data = []
    for i in range(0, learning_span):
        temp = []
        for k in range(i, i+window_size):
            temp.append(value_dataset[k].copy())

        input_train_data.append(temp)

    input_train_data = np.array(input_train_data)

    return input_train_data


def createOutputDataset(value_dataset, original_dataset, window_size, learning_span, output_train_index):
    output_train_data = []
    time_list = []
    output_train_data = value_dataset[-(window_size+learning_span):][0].copy()
#    index = 1
#    for i in range(1, learning_span+output_train_index+1):
#        if i % ((window_size*index)+output_train_index) == 0:
#            print((window_size*index)+output_train_index)
#            print(i)
#            output_train_data.append(value_dataset[i][0])
#            time_list.append(original_dataset["time"][i])
#            index = index + 1

    output_train_data = np.array(output_train_data)
    time_list = np.array(time_list)

    return output_train_data, time_list


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

train_base_time = change_to_ptime(base_time="2018-07-01 00:00:00")
output_train_index = 1
window_size = 20
learning_span = 12

original_dataset, value_dataset = getDataSet(train_base_time, connector, window_size, learning_span, output_train_index)
df = pd.DataFrame(value_dataset.copy())

value_dataset = change_to_normalization(value_dataset)
input_train_data = createInputDataset(value_dataset, window_size, learning_span)
output_train_data, output_time_list = createOutputDataset(value_dataset, original_dataset, window_size, learning_span, output_train_index)

# 訓練データの学習
np.random.seed(202)
model = build_model(input_train_data, output_size=1, neurons=20)
history = model.fit(input_train_data, output_train_data, epochs=50, batch_size=1, verbose=2, shuffle=True)



# testデータの正規化のために、最大値と最小値を取得しておく
max_list = []
min_list = []
max_price = max(original_dataset["end"])
min_price = min(original_dataset["end"])

for col in df:
    max_list.append(max(df[col]))
    min_list.append(min(df[col]))

max_list = pd.DataFrame(max_list)
min_list = pd.DataFrame(min_list)

# あとで行追加するので転置しておく
max_list = max_list.T
min_list = min_list.T

# predict用
output_train_index = 0
window_size = 20
learning_span = 1

test_base_time = change_to_ptime(base_time="2018-07-31 00:00:00")
test_original_dataset, test_value_dataset = getDataSet(test_base_time, connector, window_size, learning_span, output_train_index)

#### ここまで

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

input_test_dataset = createInputDataest(test_value_dataset, window_size, learning_span)

# shape数を揃えるためにappendする
#input_test_data = []
#input_test_data.append(test_value_dataset)
#input_test_data = np.array(input_test_data)

sql = "select end_price, insert_time from GBP_JPY_day_TABLE where insert_time < \'2018-08-01 00:00:00\' order by insert_time desc limit 2"
response = connector.select_sql(sql)
end_price_list = []
end_time_list = []
for res in response:
    end_price_list.append(res[0])
    end_time_list.append(res[1])

end_price_list.reverse()
end_time_list.reverse()

test_predict = model.predict(input_test_dataset)
print(test_predict)


#print("at %s end_price=%s" % (end_time_list[0], end_price_list[0]))
#print("at %s end_price=%s" % (end_time_list[1], end_price_list[1]))
#print("predict price=%s" % ((train_predict[0][0]*(max_price-min_price))+min_price))

#paint_predict = []
#paint_right = []
##print(time_list)
#for i in range(len(train_predict)):
#    print(time_list[i])
#    paint_predict.append((train_predict[i]*(max_price-min_price))+min_price)
#    print((train_predict[i]*(max_price-min_price))+min_price)
#    paint_right.append((output_train_data[i]*(max_price-min_price))+min_price)
#    print((output_train_data[i]*(max_price-min_price))+min_price)
#
#
#### paint predict train data
#fig, ax1 = plt.subplots(1,1)
#ax1.plot(time_list, paint_predict, label="Predict", color="blue")
#ax1.plot(time_list, paint_right, label="Actual", color="red")
#
#plt.savefig('figure.png')



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
