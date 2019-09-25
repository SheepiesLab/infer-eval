import requests
import base64
import json
import click
import csv
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
from os.path import abspath, dirname, join
import utils

import logging
logging.basicConfig(
    level   =   logging.INFO,
    format  =   '%(asctime)s %(levelname)s %(message)s',
)

folder = abspath(dirname(__file__))

@click.command()
@click.argument('env', type=click.Choice([ 'sm', 'la' ]))
def eval(env):
  meta = utils.get_meta(env)
  sender = utils.sender

  pool = ThreadPoolExecutor(max_workers=10000)

  # nums = np.arange(100, 1001, 100)
  # nums = np.append(nums, np.arange(1000, 99, -100))
  nums = []
  with open('{}/workload/test_2h.csv'.format(folder), 'r') as f:
    reader = csv.DictReader(f)
    nums = [ int(row['tweets']) for row in reader ]
    print(sum(nums))

  for i in range(len(nums)):
    num = nums[i]
    logging.info('request num: {}'.format(num))
    lam = (60 * 1000.0) / num
    samples = np.random.poisson(lam, num)
    for s in samples:
        pool.submit(sender, meta)
        time.sleep(s/1000.0)

if __name__ == '__main__':
  eval()