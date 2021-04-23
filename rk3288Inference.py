import dlr
from dlr import DLRModel
import numpy as np
import time

model_path = "/home/teamhd/inference"
device = 'opencl'

model = dlr.DLRModel(model_path, device, 0)

import cv2
from PIL import Image
capture = cv2.VideoCapture(5)
while True:
    ret, frame = capture.read()
    cv2.imshow("OpenCVFrame", frame)
    # transform to PIL
    image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    image.show()

    image = np.asarray(image.resize((416, 416)))

    mean_vec = np.array([0.485, 0.456, 0.406])
    stddev_vec = np.array([0.229, 0.224, 0.225])
    image = (image/255 - mean_vec)/stddev_vec

    if len(image.shape) == 2:
        image = np.expand_dims(image, axis=2)

    image = np.rollaxis(image, axis=2, start=0)[np.newaxis, :]

    print(image.shape)

    input_data = {'data', image}
    for _ in range(10):
        model.run(input_data)

    print('start inference ...')
    start_time = time.time()
    out = model.run(input_data)
    index = np.argmax(out[0][0,:])
    prob = np.amax(out[0][0,:])
    print('inference time is '+ str((time.time()-start_time)) + ' seconds')

    # loop finished
    key = cv2.waitKey(50)
    if key  == ord('q'):
        break
cv2.destroyAllWindows()