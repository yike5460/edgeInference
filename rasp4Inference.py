import dlr
from dlr import DLRModel
import numpy as np
import time

model_path = "/home/pi/inference/yolo3_mobilenet-rasp3b"
device = 'cpu'

model = dlr.DLRModel(model_path, device, 0)

file_name = "/home/pi/inference/test.jpg"

import PIL.Image

image = PIL.Image.open(file_name)

# try:
#     image = PIL.Image.open(file_name)
# except(OSError, NameError):
#     print('OSError, Path:', file_name)

image = np.asarray(image.resize((224, 224)))

# Normalize
mean_vec = np.array([0.485, 0.456, 0.406])
stddev_vec = np.array([0.229, 0.224, 0.225])
image = (image/255- mean_vec)/stddev_vec

# Transpose
if len(image.shape) == 2:  # for greyscale image
    image = np.expand_dims(image, axis=2)
    
image = np.rollaxis(image, axis=2, start=0)[np.newaxis, :]

print(image.shape)

#flatten within a input array
input_data = {'data': image}

# dry run
for _ in range(10):
    model.run(input_data)

print('Testing inference...')
start_time = time.time()
out = model.run(input_data) #need to be a list of input arrays matching input names
index = np.argmax(out[0][0,:])
prob = np.amax(out[0][0,:])
print('inference time is ' + str((time.time()-start_time)) + ' seconds')

# Load names for ImageNet classes
object_categories = {}
with open("imagenet1000_clsidx_to_labels.txt", "r") as f:
    for line in f:
        key, val = line.strip().split(':')
        object_categories[key] = val
print(index)
print("Result: label - " + object_categories[str(index)] + " probability - " + str(prob))
