import requests
import base64
import json
import click
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

import logging
logging.basicConfig(
    level   =   logging.INFO,
    format  =   '%(asctime)s %(levelname)s %(message)s',
)

from aws_requests_auth.boto_utils import BotoAWSRequestsAuth

with open("cat.jpg", "rb") as f:
  raw_data = f.read()
  dataString = base64.encodestring(raw_data).decode('utf-8')
  payload = {"data": dataString}
  # payload = {"image": raw_data}

urls = {
  'sm': 'https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/sm-keras-end/invocations',
  'la': 'https://797y0ky6nf.execute-api.us-east-1.amazonaws.com/default/eval-nasnet'
}

# AWS lambda layers: arn:aws:lambda:us-east-1:347034527139:layer:tf_keras_pillow:5
# refer to: https://github.com/antonpaquin/Tensorflow-Lambda-Layer/blob/master/arn_tables/tensorflow_keras_pillow.md

@click.command()
@click.argument('env', type=click.Choice([ 'sm', 'la' ]))
@click.option('--num', type=int, default=1000)
def eval(env, num):
  meta = {
    'url': urls[env],
    'json': payload
  }
  if env == 'sm':
    host='runtime.sagemaker.us-east-1.amazonaws.com'

    auth = BotoAWSRequestsAuth(aws_host=host,
                           aws_region='us-east-1',
                           aws_service='sagemaker')

    meta['auth']=auth

  def sender():
    response = requests.post(**meta)
    if response.status_code == 200:
      logging.info('succ. latency: {}; ete: {}'.format(response.json()['latency'], response.elapsed.total_seconds() * 1000))
    else:
      logging.info('fail. code: {}; ex: {}'.format(response.status_code, response.text))


  pool = ThreadPoolExecutor(max_workers=3000)

  logging.info('request num: {}'.format(num))
  lam = (60 * 1000.0) / num
  samples = np.random.poisson(lam, num)
  for s in samples:
    pool.submit(sender)
    time.sleep(s/1000.0)

if __name__ == '__main__':
  eval()