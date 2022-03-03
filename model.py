import tensorflow as tf
from tensorflow.keras.applications import mobilenet_v2
from tensorflow import keras
from tensorflow.keras.layers import Input
from tensorflow.keras import Model

def Backbone(num_classes=None):
    """
    This function creates the MobileNetV2 backbone.
    """
    def backbone(x):
        extractor = mobilenet_v2.MobileNetV2(input_shape=x.shape[1:], include_top=False, weights='imagenet', classes=num_classes)
        preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
        return Model(inputs=extractor.input, outputs=extractor.output, name='Backbone')(preprocess(x))
    return backbone

def MultiLabelClassifier(cfg, num_classes, training=False):
    """
    This function creates the MultiLabelClassifier.
    """
    input_size = int(cfg['train']['input_size']) if training else None
    x = inputs = Input([input_size, input_size, 3], name='input')
    x = Backbone(num_classes)(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    outputs = keras.layers.Dense(num_classes, activation='sigmoid')(x)
    return Model(inputs, outputs, name='MultiLabelClassifier')