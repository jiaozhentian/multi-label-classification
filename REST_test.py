import time
import json
import requests

import tensorflow as tf
image = tf.keras.utils.load_img(
    './data/test/test3.jpeg', target_size=(128, 128))
image_array = tf.keras.utils.img_to_array(image)
image_array = tf.expand_dims(image_array, 0)  # Create a batch
with open('test.json', 'w') as f:
    f.write(json.dumps({'inputs': image_array.numpy().tolist()}))
    f.close()

start = time.time()
r = requests.post("http://192.168.113.63:8507/v1/models/face_attribute:predict",
                  data=json.dumps({'inputs': image_array.numpy().tolist()}),
                  headers={'Content-Type': 'application/json'})
end = time.time()

print("r=", r.text)
print('time:', end - start)