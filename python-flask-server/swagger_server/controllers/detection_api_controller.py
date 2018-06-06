import connexion
import six

from swagger_server.models.api_response import ApiResponse  # noqa: E501
from swagger_server.models.detected_object import DetectedObject  # noqa: E501
from swagger_server import util

from keras.models import load_model
import os
from keras.applications import mobilenet

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
import keras.applications.mobilenet as mobilenet

import uuid
import numpy as np

global model
model = None

def get_model():
    global model
    if not model:
        model = load_model(os.environ['MODEL_FILE'], custom_objects = {
            'relu6': mobilenet.relu6,
            'DepthwiseConv2D': mobilenet.DepthwiseConv2D
        })
        return model
    else:
        return model

def upload_file(image, imageResolution, imageFormat):  # noqa: E501
    """Performs object detection on uploaded image

    Performs simple and low accuracy object detection to help filter-out irrelevant parts of the video stream # noqa: E501

    :param image: image to upload in order to perform object detection
    :type image: werkzeug.datastructures.FileStorage
    :param imageResolution: image resolution (Width x Height) in pixels
    :type imageResolution: str
    :param imageFormat: allowed image formats
    :type imageFormat: str

    :rtype: ApiResponse
    """
    model = get_model()
    path = '/tmp/{}'.format(uuid.uuid4().hex)
    image.save(path)
    original = load_img(path, target_size=(224, 224))
    os.remove(path)
    numpy_image = img_to_array(original)
    image_batch = np.expand_dims(numpy_image, axis=0)
    processed_image = mobilenet.preprocess_input(image_batch.copy())
    predictions = model.predict(processed_image)
    decoded = decode_predictions(predictions)
    return ApiResponse(detected_objects = [ DetectedObject(x, y) for (x, y) in decoded ])

def decode_predictions(predictions):
    results = []
    for prediction in predictions:
        results.append(('person', max(prediction[2], 1-prediction[1])))
    return results
