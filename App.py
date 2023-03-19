
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import os 
from glob import glob
from PIL import Image

project_directory= "C:/Users/pc/Nextcloud/Python/GITHUB/Computer_vision_CNN/"
data_directory=project_directory+"data/"

data=pd.read_csv(data_directory+"HAM10000_metadata.csv")
