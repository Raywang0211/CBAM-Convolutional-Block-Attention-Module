# CBAM-Convolutional-Block-Attention-Module
CBAM: Convolutional Block Attention Module
# 框架

tensorflow2.6.0

# 實作

在使用tensorflow 框架時有提供自定義層的功能，可以客製化任何操作並且使用於建構模型時，是個很方便的操作方式，今天就簡單的介紹如何使用tensorflow框架下的layer來建構客製化的網路層。

使用CBAM 的attention模快進行範例

建立自定義層的時候主要有3個區塊

1. init()：如果在自己定義的層中有使用到tf框架的其他方法，需要再這個地方先建立起來
2. call()：自定義層中的操作區塊操作，也就是說要對輸入的影像或是feature map做任何操作都是再這個區塊進行定義
3. get_config：因為最後模快還是要跟tf況架中的sequence或是function API進行連接，所以需要復寫layer中的get_config方法，才可以讓模型運行時抓到自定義的方法名稱進行運算，並且正確的顯示出來。

我們根據論文建構兩個模快一個是channel module 一個是spatial module

def init:

輸入自己需要的參數之後，最後面要加入**kwargs不然程式會報錯，因為有繼承到layer的init因此需要有這個輸入的參數，不然父類的init會有問題。

範例中可以發現我們利用到tf內建的Convene2D，參數可以自己寫死設定好，或是根據字定義層

數入決定，這邊需要注意如果有使用到****實例變數(instance variable)****，需要在後面get_config時進行定義讓父類知道。

def call

這邊完整的進行資料輸入之後的所有操作，最後return就是整個layer的output

def get_config

這邊需要複寫父類別的get_config

最後需要讓框架知道有一個custom_objects，叫什麼名字以及做了什麼操作

```python
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
```

### 實際使用

### 模型建立或是直接載入

這邊使用一個簡單的xception 的transfer learning搭配attention block進行模型架構練習

```python
from tensorflow.keras.applications.xception import Xception
from attention_block import CBM_channel,CBM_spatial

xcepnet = Xception(include_top=False, weights="imagenet",input_tensor=None, input_shape=(512,736,3))
for layer in xcepnet.layers:
    layer.trainable = False

inputs = Input((512,736, 3))
X_net = xcepnet(inputs)
CBM_ch = CBM_channel(2048)(X_net) #自定義模塊
CBM_SP = CBM_spatial()(CBM_ch) #自定義模塊
flat = Flatten()(CBM_SP)
dens1 = Dense(10, activation="relu")(flat)
dens2 = Dense(4, activation="softmax")(dens1)

model = Model(inputs=inputs, outputs=dens2)
model.compile(optimizer=Adam(lr=1e-5),loss = "categorical_crossentropy", metrics=['accuracy'])

model.summary(line_length=200)

new_model = load_model("./model.h5",compile=False)
```
