import numpy as np
import pandas as pd
import re
import torch
import random
import pickle
import json
import torch.nn as nn # provides a way to define custom neural network modules by subclassing the nn.Module class.
import transformers
import matplotlib.pyplot as plt

from torch.optim import Adam
from tqdm import tqdm
from torchinfo import summary

import os

# Optimizer
from transformers import AdamW

# Class Weights
from sklearn.utils.class_weight import compute_class_weight

from torch.optim import lr_scheduler

from torch.optim.lr_scheduler import LambdaLR
from torch.optim.lr_scheduler import OneCycleLR

from transformers import get_linear_schedule_with_warmup

import matplotlib.pyplot as plt
#%matplotlib inline
import seaborn as sns

from transformers import DistilBertTokenizer, DistilBertModel 
#from transformers import AutoTokenizer, AutoModel

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

import time
import datetime
from datetime import date

import csv, sys, uuid, joblib

from collections import defaultdict

import onnx
import onnxruntime

import matplotlib.ticker as ticker