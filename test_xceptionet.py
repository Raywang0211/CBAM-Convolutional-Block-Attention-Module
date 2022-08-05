
from tensorflow.keras.applications.xception import Xception 
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input,Dense,Flatten
from tensorflow.keras.models import Model
from attention_block import CBM_channel,CBM_spatial
from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.keras.models import load_model

xcepnet = Xception(include_top=False, weights="imagenet",input_tensor=None, input_shape=(512,736,3))
for layer in xcepnet.layers:
    layer.trainable = False

inputs = Input((512,736, 3))
X_net = xcepnet(inputs)
CBM_ch = CBM_channel(2048)(X_net)
CBM_SP = CBM_spatial()(CBM_ch)
flat = Flatten()(CBM_SP)
dens1 = Dense(10, activation="relu")(flat)
dens2 = Dense(4, activation="softmax")(dens1)

model = Model(inputs=inputs, outputs=dens2)
model.compile(optimizer=Adam(lr=1e-5),loss = "categorical_crossentropy", metrics=['accuracy'])

model.summary(line_length=200)

