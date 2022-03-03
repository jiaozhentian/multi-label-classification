import tensorflow as tf

model = tf.keras.models.load_model('./savedmodel')

image = tf.keras.utils.load_img('./data/test/test3.jpeg', target_size=(128, 128))
image_array = tf.keras.utils.img_to_array(image)
image_array = tf.expand_dims(image_array, 0) # Create a batch

predictions = model.predict(image_array)
print(predictions)