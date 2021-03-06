# coding: utf-8

import pandas as pd

pd.set_option("display.max_colwidth", 1000)
pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

import numpy as np
np.set_printoptions(threshold=np.inf)
import matplotlib.pyplot as plt
plt.switch_backend("agg")

from mysql_connector import MysqlConnector
from tensor_lib import change_to_ptime, getDataset, change_to_normalization, createTrainDataset, build_model

connector = MysqlConnector()
train_base_time = change_to_ptime(base_time="2018-07-01 00:00:00")

table_layout = "1h"
output_train_index = 8
learning_span = 24*120 # 半年分
window_size = 24
original_dataset, value_dataset = getDataset(train_base_time, connector, window_size, learning_span, output_train_index, table_layout)
df = pd.DataFrame(value_dataset.copy())

max_price = max(original_dataset["end"])
min_price = min(original_dataset["end"])

value_dataset = change_to_normalization(value_dataset)
input_train_data, output_train_data, time_list = createTrainDataset(value_dataset, original_dataset, window_size, learning_span, output_train_index)

np.random.seed(202)
model = build_model(input_train_data, output_size=1, neurons=200)
history = model.fit(input_train_data, output_train_data, epochs=500, batch_size=1, verbose=2, shuffle=True)

# モデルの保存
json_string = model.to_json()
open("model.json", "w").write(json_string)
model.save_weights("model_weights.hdf5")

## 訓練データで予測
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

