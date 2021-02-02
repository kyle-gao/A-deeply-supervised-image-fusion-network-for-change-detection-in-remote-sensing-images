import DSIFN
from loss import bce_dice_loss
import keras
import attention_module
from keras.models import load_model
from keras import backend as K
import os
import PIL
from keras.preprocessing import image 
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import tensorflow as tf
#need this or cudnn won't load properly
tf.keras.backend.clear_session()
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
from F1 import f1



cwd = "/home/keyang/Desktop/2021igrss_data/DSIFN/IFN/Keras version"
weights = "best_model_cncities.h5"
p1 = "/home/keyang/Desktop/2021igrss_data/DSIFN/TestSet_NAIP/1950_0_naip2013.jpeg"
p2 = "/home/keyang/Desktop/2021igrss_data/DSIFN/TestSet_NAIP/1950_0_naip2017.jpeg"

weights_path = os.path.join(cwd,weights)


model = load_model(weights_path, custom_objects={'bce_dice_loss': bce_dice_loss,'f1':f1})

model.compile(optimizer = 'Adam', loss = bce_dice_loss, metrics = ['accuracy'])



img1 = np.expand_dims(image.img_to_array(image.load_img(p1)),axis=0)/255
img2 = np.expand_dims(image.img_to_array(image.load_img(p2)),axis=0)/255

out = model.predict([img1,img2],batch_size = None)

out = out[-1][0,:,:,:]

pil_image = image.array_to_img(out)

savepath = os.path.join(cwd, "testresult.jpeg")
pil_image.save(savepath)
