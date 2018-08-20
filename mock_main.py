# coding: utf-8

import pandas as pd
import numpy as np
import seaborn as sns
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

import json

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


############################
#
# MOCK PROGRAM 
#
#ACCOUNT_ID = "2542764"
#ACCESS_TOKEN = "cb570464152b22d04da3f0f5cad2ddd4-0d543f436361df398e1b2ffa6daf227d"
#ENV = "practice"
#
#oanda = oandapy.API(environment=ENV, access_token=ACCESS_TOKEN)
#
#response = oanda.get_history(instrument="USD_JPY", granularity="D")

file = open("output.json", "r")
response = json.load(file)

#
#
#############################

#print(response)
#print("============================")

#print(response["candles"])
#print("============================")
res = pd.DataFrame(response["candles"])
#print(res.head())


res["time"] = res["time"].apply(lambda x: iso_jp(x))
res["time"] = res["time"].apply(lambda x: date_string(x))

#print(res.head())

#print(res.shape)
#print(res.columns)

#print(res['time'])
df = res[['time', 'openAsk', 'closeAsk', 'highAsk', 'lowAsk', 'volume']]
#print(df.columns)

df.columns = ['time', 'openAsk', 'closeAsk', 'highAsk', 'lowAsk', 'volume']
#print(df.columns)

train, test = df[:400], df[400:]

#train = train * 100
#test = test * 100

del train["time"]
del test["time"]

#print(train)
#print(test)

window_len = 21

train_lstm_in = []

#print(train.columns)
#print(test.columns)


for i in range(len(train) - window_len):
    temp = train[i:(i+window_len)].copy()
    #print(temp)
    #print("=======================")

    for col in train:
        #print(col)
        #print("=======================")
        #print(temp[col])
        #print("=======================")
        #print(temp[col].iloc[0])
        temp.loc[:, col] = temp[col]/temp[col].iloc[0] - 1
        #print(temp)
    train_lstm_in.append(temp)

train_lstm_out = (train["closeAsk"][window_len:].values/train["closeAsk"][:-window_len].values) - 1
#train_lstm_out = (train[:][window_len:].values/train[:][:-window_len].values) - 1


test_lstm_in = []
for i in range(len(test) - window_len):
    temp = test[i:(i+window_len)].copy()
    for col in test:
        temp.loc[:, col] = temp[col]/temp[col].iloc[0] - 1

    test_lstm_in.append(temp)

test_lstm_out = (test["closeAsk"][window_len:].values/test["closeAsk"][:-window_len].values) - 1

#print(train_lstm_in)

train_lstm_in = [np.array(train_lstm_input) for train_lstm_input in train_lstm_in]
train_lstm_in = np.array(train_lstm_in)

test_lstm_in = [np.array(test_lstm_input) for test_lstm_input in test_lstm_in]
test_lstm_in = np.array(test_lstm_in)

#print(train_lstm_in)

np.random.seed(202)
yen_model = build_model(train_lstm_in, output_size=1, neurons=20)

print(train_lstm_in.shape)
print(train_lstm_out.shape)

#print(train_lstm_out)
#print(train_lstm_in)

yen_history = yen_model.fit(train_lstm_in, train_lstm_out, epochs=50, batch_size=1, verbose=2, shuffle=True)

### paint epoch, loss rate
#
#fig, ax1 = plt.subplots(1,1)
#ax1.plot(yen_history.epoch, yen_history.history["loss"])
#ax1.set_title("TrainingError")
#
#if yen_model.loss == "mae":
#    ax1.set_ylabel("Mean Absolute Error(MAE)", fontsize=12)
#else:
#    ax1.set_ylabel("Model Loss", fontsize=12)
#ax1.set_xlabel("# Epochs", fontsize=12)
#plt.show()
#

#result = (yen_model.predict(train_lstm_in)+1) * train["closeAsk"].values[:-window_len]

#print(result[0])
#print(len(result[0]))

#pd_result = pd.DataFrame(result[0])
#print(pd_result)


### paint predict train data
fig, ax1 = plt.subplots(1,1)
ax1.plot(df["time"][window_len:].astype(datetime), train["closeAsk"][window_len:], label="Actual", color="blue")
ax1.plot(df["time"][window_len:].astype(datetime), ((np.transpose(yen_model.predict(train_lstm_in))+1) * train["closeAsk"].values[:-window_len])[0], label="Actual", color="blue")

