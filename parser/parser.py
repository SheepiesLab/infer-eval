import numpy as np
import cvxpy as cp
import re
import random
import glob
import matplotlib
matplotlib.use('TkAgg')
import matplotlib as mpl
import matplotlib.pyplot as plt
import statsmodels.api as sm
from matplotlib import rc
import click
from matplotlib.ticker import MaxNLocator
import os
import math
# mpl.rcParams['xtick.direction'] = 'in'
# mpl.rcParams['ytick.direction'] = 'in'        
# plt.rcParams.update({'font.size': 16})
# mpl.rcParams['figure.autolayout']= True

@click.command()
@click.argument('path', type=click.Path(exists=True, resolve_path=True))
def eval(path):
  pass


if __name__ == '__main__':
  parse()