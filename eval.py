import requests
import base64
import json
import click
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time

from aws_requests_auth.boto_utils import BotoAWSRequestsAuth

with open("cat.jpg", "rb") as f:
  raw_data = f.read()
  dataString = base64.encodestring(raw_data).decode('utf-8')
  payload = {"data": dataString}
  # payload = {"image": raw_data}

urls = {
  'sm': 'https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/sm-keras-end/invocations',
  'la': 'https://a5jmq9mpwb.execute-api.us-east-1.amazonaws.com/default/test-upload'
}

@click.command()
@click.argument('env', type=click.Choice([ 'sm', 'la' ]))
def eval(env):
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

  def sender(warmup=False):
    response = requests.post(**meta)
    if not warmup:
      if response.status_code == 200:
        print('succ. latency: {}; ete: {}'.format(response.json()['latency'], response.elapsed.total_seconds() * 1000))
      else:
        print('fail. code: {}'.format(response.status_code))

  sender(warmup=True)

  pool = ThreadPoolExecutor(max_workers=3000)
  nums = [30, 60, 90, 120, 150, 180] # first min to warm up
  for i in range(len(nums)):
    num = nums[i]
    print('request num: {}'.format(num))
    lam = (60 * 1000.0) / num
    samples = np.random.poisson(lam, num)
    for s in samples:
        pool.submit(sender)
        time.sleep(s/1000.0)

if __name__ == '__main__':
  eval()