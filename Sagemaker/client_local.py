import requests
import base64
import json

with open("cat.jpg", "rb") as f:
    raw_data = f.read()
    dataString = base64.encodestring(raw_data).decode('utf-8')
    payload = {"data": dataString}

for i in range(0, 10):

  response = requests.post(
    url="http://localhost:5000/predict",
    json=payload
  )

  print(response.text)
  print('code: ' + str(response.status_code) + ', latency: ' + str(response.elapsed.total_seconds()))
