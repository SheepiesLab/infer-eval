import requests
import base64
import json
import click
import csv
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import time
from os.path import abspath, dirname, join

import logging
logging.basicConfig(
    level   =   logging.INFO,
    format  =   '%(asctime)s %(levelname)s %(message)s',
)

from aws_requests_auth.boto_utils import BotoAWSRequestsAuth

# AWS lambda layers: arn:aws:lambda:us-east-1:347034527139:layer:tf_keras_pillow:5
# refer to: https://github.com/antonpaquin/Tensorflow-Lambda-Layer/blob/master/arn_tables/tensorflow_keras_pillow.md

urls = {
  'sm': 'https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/sm-keras-end/invocations',
  'la': 'https://6wcbl1yw9g.execute-api.us-east-1.amazonaws.com/dev/predict'
  # 'la': 'https://4sw934su03.execute-api.us-east-1.amazonaws.com/default/test'
}

def get_payload():
  with open("cat.jpg", "rb") as f:
    raw_data = f.read()
    dataString = base64.encodestring(raw_data).decode('utf-8')
    payload = {"data": dataString}
    return payload

def get_meta(env):
  meta = {
    'url': urls[env],
    'data': 'https://farm5.staticflickr.com/4275/34103081894_f7c9bfa86c_k_d.jpg'
  }

  if env == 'sm':
    host='runtime.sagemaker.us-east-1.amazonaws.com'

    auth = BotoAWSRequestsAuth(aws_host=host, aws_region='us-east-1', aws_service='sagemaker')
    meta['auth']=auth
  return meta

def sender(meta):
  response = requests.get(meta['url'])
  if response.status_code == 200:
    logging.info('succ. latency: {}; ete: {}'.format(response.json()['latency'], response.elapsed.total_seconds() * 1000))
  else:
    logging.info('fail. code: {}'.format(response.status_code))