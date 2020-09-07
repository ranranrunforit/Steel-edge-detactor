#from keras.models import load_model
#model = load_model("models/resnet50_pspnet_10_model.h5")
#model.summary()

from keras.models import model_from_json
# load json and create model
json_file = open('models/resnet50_unet_20_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("models/resnet50_unet_20_model.h5")
print("Loaded model from disk")
loaded_model.summary()


from keras_segmentation.predict import predict
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.85
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)
"""
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')

tf.config.experimental.set_virtual_device_configuration(gpus[0], [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024)])
"""

predict(
	checkpoints_path="tmp/checkpoint\\resnet50_unet_20",
	inp="dataset1/images_prepped_test/4003.png",
	out_fname="outputs/output.png"
)

'''
from keras_segmentation.predict import predict_multiple


predict_multiple(
	checkpoints_path="checkpoints/vgg_unet_1",
	inp_dir="dataset_path/images_prepped_test/",
	out_dir="outputs/"
)
'''