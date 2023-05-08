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
from osgeo import gdal, ogr, osr


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

#Flip function
def flip(m, axis):
    if not hasattr(m, 'ndim'):
        m = asarray(m)
    indexer = [slice(None)] * m.ndim
    try:
        indexer[axis] = slice(None, None, -1)
    except IndexError:
        raise ValueError("axis=%i is invalid for the %i-dimensional input array"
                         % (axis, m.ndim))
    return m[tuple(indexer)]

#Opening and loading files
#Model
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
    in_path = (r'mask.png')
    out_path = (r'Mask_Shapefile.shp')
    #  get raster datasource
    src_ds = gdal.Open( in_path )
    srcband = src_ds.GetRasterBand(1)
    dst_layername = 'Extracted Field Boundaries'
    drv = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds = drv.CreateDataSource( out_path )
    sp_ref = osr.SpatialReference()
    sp_ref.SetFromUserInput('EPSG:4326')
    dst_layer = dst_ds.CreateLayer(dst_layername, srs = sp_ref )
    fld = ogr.FieldDefn("HA", ogr.OFTInteger)
    dst_layer.CreateField(fld)
    dst_field = dst_layer.GetLayerDefn().GetFieldIndex("HA")
    gdal.Polygonize( srcband, None, dst_layer, dst_field, [], callback=None )
    del src_ds
    del dst_ds
    #Zipping Files
    file_paths= ['Mask_Shapefile.shp', 'Mask_Shapefile.dbf', 'Mask_Shapefile.shx', 'Mask_Shapefile.prj']
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
