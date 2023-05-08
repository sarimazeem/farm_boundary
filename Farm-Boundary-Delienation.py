#Importing libraries
from tensorflow.keras.utils import Sequence
import cv2
import tensorflow as tf 
import pickle
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sklearn
from sklearn.cluster import KMeans
from tensorflow.keras.layers import *
from tensorflow.keras import models
from tensorflow.keras.callbacks import * 
from tensorflow.keras.applications import ResNet50
import glob2
from sklearn.utils import shuffle
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.models import load_model
import streamlit as st
import pandas as pd
from PIL import Image
import base64
from skimage import data
from skimage import filters
from skimage import exposure
from numpy import asarray
import numpy as np
import os.path
import gdown
import rasterio
from rasterio.features import shapes
import geopandas as gpd



#Defining Functions for Mean IOU
class m_iou():
    def __init__(self, classes: int) -> None:
        self.classes = classes
    def mean_iou(self,y_true, y_pred):
        y_pred = np.argmax(y_pred, axis = 3)
        miou_keras = MeanIoU(num_classes= self.classes)
        miou_keras.update_state(y_true, y_pred)
        return miou_keras.result().numpy()
    def miou_class(self, y_true, y_pred):
        y_pred = np.argmax(y_pred, axis = 3)
        miou_keras = MeanIoU(num_classes= self.classes)
        miou_keras.update_state(y_true, y_pred)        
        values = np.array(miou_keras.get_weights()).reshape(self.classes, self.classes)
        for i in  range(self.classes):
            class_iou = values[i,i] / (sum(values[i,:]) + sum(values[:,i]) - values[i,i])
            print(f'IoU for class{str(i + 1)} is: {class_iou}')


#Predict Model Function
def predict(model, image_test, label, color_mode, size):
    image = Image.open(image_test)
    image= np.asarray(image)
    #image = cv2.imread(image_test)
    if color_mode == 'hsv':
        image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    elif color_mode == 'rgb':
        image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    elif color_mode == 'gray':
        image_cvt = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        image_cvt = tf.expand_dims(image_cvt, axis = 2)
    image_cvt = tf.image.resize(image_cvt, size, method= 'nearest')
    image_cvt = tf.cast(image_cvt, tf.float32) 
    image_norm = image_cvt / 255.
    image_norm = tf.expand_dims(image_norm, axis= 0)
    new_image = model(image_norm)
    image_argmax = np.argmax(tf.squeeze(new_image, axis = 0), axis = 2)
    image_decode = decode_label(image_argmax, label)
    predict_img = tf.cast(tf.image.resize(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), size, method = 'nearest'), tf.float32) * 0.7 + image_decode * 0.3
    return np.floor(predict_img).astype('int'), new_image


#Label Decode Function
def decode_label(predict, label):
    d = list(map( lambda x: label[int(x)], predict.reshape(-1,1)))
    img =  np.array(d).reshape(predict.shape[0], predict.shape[1], 3)
    return img

#Opening and loading files
#Model
path = 'model.h5'
check_file = os.path.isfile(path)

if check_file!= True: 
    url = 'https://drive.google.com/uc?id=1sdV4Ju_4sE10Ev27W3j9xxuC8PNos54k'
    output = 'model.h5'
    gdown.download(url, output, quiet=False)
    
m = m_iou(3)
model = load_model(r'model.h5', compile= False)

#custom_objects = {'mean_iou': m.mean_iou}
model.compile()

#Label
with open(r'label.pickle', 'rb') as handel:
    label = pickle.load(handel)

#K-Means 
with open(r'kmean.pickle', 'rb') as handel:
    kmean = pickle.load(handel)

icon=(r'Icon.jpg')
with st.sidebar: 
    st.title("Automatic Farm Boundary Delienation")
    st.image(icon)
    st.info("This project helps you delienate farm boundaries.")
    choice = st.radio("Navigation", ["Satellite Imagery Upload","Boundary Delienation", "Shapefile Download"])

st.subheader("Upload your Satellite Imagery")
file = st.file_uploader("")
if file:
    mask= predict(model, file, label, 'rgb', (224,224))
    
if choice=="Boundary Delienation":
    #From Array to Image conversion
    np, tf= mask
    tensor_arr = tf[0]
    mask=tensor_arr[:,:,0]
    z= mask.numpy()
    mask= Image.fromarray((z * 255).astype('uint8'), mode='L')
    #Otus Binary Threshold
    array=asarray(mask)
    camera = array
    val = filters.threshold_otsu(camera)
    new_array= camera<val
    new_array= new_array.astype(int)
    z= new_array
    mask= Image.fromarray((z * 255).astype('uint8'), mode='L')
    import numpy as np
    img = asarray(mask)
    kernel = np.ones((2, 2), np.uint8)
    img_erosion = cv2.erode(img, kernel, iterations=1)
    
    cv2.imwrite("mask.png", img_erosion)
    img_erosion=Image.fromarray(img_erosion)
    mask= img_erosion
    st.image(mask)

if choice== "Shapefile Download":
    mask = cv2.imread('mask.png')
    flip = cv2.flip(mask, 1)
    rotate = cv2.rotate(flip, cv2.ROTATE_90_CLOCKWISE)
    rotate = cv2.rotate(rotate, cv2.ROTATE_90_CLOCKWISE)
    cv2.imwrite("mask.png", rotate)
    file = r"mask.png"
    mask = None
    with rasterio.open(file) as src:
        image = src.read(1) # 
        results = (
    {'properties': {'raster_val': v}, 'geometry': s}
    for i, (s, v) 
    in enumerate(
        shapes(image, mask=mask, transform=src.transform)))
    geoms = list(results)
    df  = gpd.GeoDataFrame.from_features(geoms)
    df = df.set_crs(3006)
    df["raster_val"] = np.floor(df["raster_val"]).astype(int) 
    #Restore the original raster values
    df.to_file(r"Mask_Shapefile.shp")
    #Zipping Files
    file_paths= ['Mask_Shapefile.shp', 'Mask_Shapefile.dbf', 'Mask_Shapefile.shx']
    from zipfile import ZipFile
    with ZipFile('Shapefile.zip','w') as zip:
        for file in file_paths:
            zip.write(file)
    with open("Shapefile.zip", "rb") as fp:
        st.download_button(
        label="Download ZIP",
        data=fp,
        file_name="Shapefile.zip",
        mime="application/zip"
    )