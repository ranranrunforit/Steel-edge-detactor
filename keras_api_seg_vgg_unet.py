from keras_segmentation.models.unet import vgg_unet
from keras.backend.tensorflow_backend import set_session
import tensorflow as tf
from keras.models import model_from_json

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras

#416，608
#input_h_w% 32 == 0
model = vgg_unet(n_classes=3 ,  input_height=416, input_width=608  )

model.train(
    train_images =  "dataset1/images_prepped_train/",
    train_annotations = "dataset1/annotations_prepped_train/",
    checkpoints_path = "tmp/checkpoint/vgg_unet_20" ,
    epochs=20
)


out = model.predict_segmentation(
    inp="dataset1/images_prepped_test/4003.png",
    out_fname="outputs/vgg_unet_20_out.png"
)

import matplotlib.pyplot as plt
plt.imshow(out)

# evaluating the model
print(model.evaluate_segmentation( inp_images_dir="dataset1/images_prepped_test/"  , annotations_dir="dataset1/annotations_prepped_test/" ) )

model.save("models/vgg_unet_20_model.h5")
print("Saved model to disk")

model_json = model.to_json()
with open("models/vgg_unet_20_model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("models/vgg_unet_20_model_weight.h5")
print("Saved model weights to disk")

"""

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# later...

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")

"""