# coding: utf-8

import oandapy
import json

ACCOUNT_ID = "2542764"
ACCESS_TOKEN = "cb570464152b22d04da3f0f5cad2ddd4-0d543f436361df398e1b2ffa6daf227d"
ENV = "practice"

oanda = oandapy.API(environment=ENV, access_token=ACCESS_TOKEN)

response = oanda.get_history(instrument="USD_JPY", granularity="D")


print(type(response))

file = open("output.json", "w")
json.dump(response, file, ensure_ascii=False)
