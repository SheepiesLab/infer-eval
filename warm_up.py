import requests
import base64
import json
import click
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import time
import utils

import logging
logging.basicConfig(
    level   =   logging.INFO,
    format  =   '%(asctime)s %(levelname)s %(message)s',
)

@click.command()
@click.argument('env', type=click.Choice([ 'sm', 'la' ]))
@click.option('--num', type=int, default=1000)
def eval(env, num):
  meta = utils.get_meta(env)
  sender = utils.sender

  pool = ThreadPoolExecutor(max_workers=1000)

  logging.info('request num: {}'.format(num))
  lam = (60 * 1000.0) / num
  samples = np.random.poisson(lam, num)
  for s in samples:
    pool.submit(sender, meta)
    time.sleep(s/1000.0)

if __name__ == '__main__':
  eval()