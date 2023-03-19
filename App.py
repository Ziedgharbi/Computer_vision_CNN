
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

import os 
from glob import glob
from PIL import Image

project_directory= "C:/Users/pc/Nextcloud/Python/GITHUB/Computer_vision_CNN/"
data_directory=project_directory+"data/"
image_directory= data_directory +"images/"

df=pd.read_csv(data_directory+"HAM10000_metadata.csv")
df.columns

df['image_path']= [os.path.join(image_directory+str(name)+'.jpg') for name in df["image_id"]]


"""----- we limit data volume cause need so much ressources that i don't have that :) ---------"""

frac = 0.3  # take 30% of data only 

data=df.sample(frac=frac, random_state=1)

data["image_array"]= data['image_path'].map(lambda x: np.asarray(Image.open(x).resize((32,32))))


# plot random picture
n_sample=15
random=np.random.randint(0,data.shape[0],n_sample)

plt.subplots (3,5, figsize=(5,5))

for i in range(15):
    plt.subplot(3,5,i+1)
    plt.imshow(data.image_array.iloc[random[i]])
    plt.title(data.dx.iloc[i])
plt.show()


# some descreptive statistic of classes 
data.dx.value_counts().plot(kind='pie')


# some manuel encoding for taregt variable 
""" nv : Melanocytic nevi  / mel : Melanoma    /
  bkl : Benign keratosis-like lesions  / bcc : Basal cell carcinoma
  akiec : Actinic keratoses   / vas : Vascular lesions /   df :  Dermatofibroma     """
data.dx.unique()

classe=dict(zip(data.dx.unique(), range (1,len(data.dx.unique())+1)   ))

data["target"]=data.dx.map(lambda x : classe[x]  )


"""------- do somes transformation on image and for each we train a model and voting for rslt ---------"""
