import tensorflow_addons as tfa
import tensorflow as tf
import numpy as np
from tensorflow.keras.layers import GlobalAveragePooling2D,GlobalMaxPool2D,Reshape,Conv2D,Input,Flatten,Dense,Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer
from tensorflow.keras.utils import get_custom_objects


class CBM_channel(Layer):
    def __init__(self,in_channel,ratio=8,**kwargs):
        super(CBM_channel, self).__init__()
        self.in_channel = in_channel
        self.ratio = ratio
        self.avg_pool = GlobalAveragePooling2D(keepdims=True)
        self.max_pool = GlobalMaxPool2D(keepdims=True)
        self.dens1 = Conv2D(self.in_channel//self.ratio ,kernel_size=1,activation="relu")
        self.dens2 = Conv2D(self.in_channel,kernel_size=1,activation="relu")

    def call(self, inputs):
        avg_out = self.avg_pool(inputs)
        max_out = self.max_pool(inputs)
        avg_out = self.dens2(self.dens1(avg_out))
        max_out = self.dens2(self.dens1(max_out))
        out = tf.stack([avg_out, max_out], axis=1) 
        out = tf.reduce_sum(out, axis=1)  
        out = tf.nn.sigmoid(out)
        out = tf.multiply(inputs,out)

        return out
    def get_config(self):
        config = {
            'avg_pool': self.avg_pool,
            'max_pool': self.max_pool,
            'dens1': self.dens1,
            'dens2': self.dens2,
            "in_channel":self.in_channel,
            "ratio":self.ratio
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
get_custom_objects().update({'CBM_channel': CBM_channel})

class CBM_spatial(Layer):
    def __init__(self,**kwargs):
        super().__init__()
        self.conv = Conv2D(filters=1,kernel_size=(7,7),padding='same',activation='relu')

    def call(self, inputs):

        avg_out = tf.reduce_mean(inputs, axis=3)
        avg_out = tf.expand_dims(avg_out,-1)
        max_out = tf.reduce_max(inputs,axis=3)
        max_out = tf.expand_dims(max_out,-1)
        out = tf.concat([avg_out,max_out],axis=3)
        out = self.conv(out)
        out = tf.nn.sigmoid(out)
        out = tf.multiply(inputs,out)
        return out

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'conv': self.conv
        })
        return config
get_custom_objects().update({'CBM_spatial': CBM_spatial})
