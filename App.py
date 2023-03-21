import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import os 
from glob import glob
from PIL import Image
from skimage import color, exposure
from skimage.util import img_as_ubyte
from skimage.filters import rank
from skimage.morphology import disk

import h5py



project_directory= "C:/Users/pc/Nextcloud/Python/GITHUB/Computer_vision_CNN/"
data_directory=project_directory+"data/"
image_directory= data_directory +"images/"
data_transorfmed_directory=project_directory+'transformed_data/'

df=pd.read_csv(data_directory+"HAM10000_metadata.csv")
df.columns

df['image_path']= [os.path.join(image_directory+str(name)+'.jpg') for name in df["image_id"]]


"""----- if you don't have so much ressources you can limit data ---------"""

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
data.dx.value_counts().plot(kind='pie') # data are unbalanced, we should balance it


# some manuel encoding for taregt variable 
""" nv : Melanocytic nevi  / mel : Melanoma    /
  bkl : Benign keratosis-like lesions  / bcc : Basal cell carcinoma
  akiec : Actinic keratoses   / vas : Vascular lesions /   df :  Dermatofibroma     """
data.dx.unique()

classe=dict(zip(data.dx.unique(), range (1,len(data.dx.unique())+1)   ))

data["target"]=data.dx.map(lambda x : classe[x]  )


"""------- do somes transformation on image and for each we train a model and voting for rslt ---------"""

# first balance data

data.dx.value_counts()

n_sample=300 # you can chos other number


data_balanced=pd.DataFrame()
for i in data.target.unique() :
    d=data[data.target == i]
    temp= d.sample(n=n_sample, replace=True)
    data_balanced=pd.concat( [data_balanced, temp])
    


X=np.asarray(data_balanced.image_array) 
X=X/255
y=data_balanced.target


# transformation 

# ---- RGB / Histogram Equalization
X_rgb_h=[]
for img in X:
    hsv = color.rgb2hsv(img)
    hsv[:, :, 2] = exposure.equalize_hist(hsv[:, :, 2])
    img = color.hsv2rgb(hsv)
    X_rgb_h.append(img)   

plt.imshow(X_rgb_h[1])  

# ---- Grayscale
X_gray=[]
for img in X:
    img=color.rgb2gray(img)
    X_gray.append(img)
          
plt.imshow(X_gray[1]) 

# ---- Grayscale / Histogram Equalization
X_gray_HE=[]
for img in X:
    img=color.rgb2gray(img)
    img=exposure.equalize_hist(img)
    X_gray_HE.append(img)

plt.imshow(X_gray_HE[1])   
     
# ---- Grayscale / Local Histogram Equalization
X_gray_L_HE=[]
for img in X:
    img=color.rgb2gray(img)
    img = img_as_ubyte(img)
    img=rank.equalize(img, disk(10))/255.
    X_gray_L_HE.append(img)
        
plt.imshow(X_gray_L_HE[1])  

# ---- Grayscale / Contrast Limited Adaptive Histogram Equalization (CLAHE)
X_gray_L_CLAHE=[]

for img in X:
    img=color.rgb2gray(img)
    img=exposure.equalize_adapthist(img)
    X_gray_L_CLAHE.append(img)

plt.imshow(X_gray_L_CLAHE[1]) 


