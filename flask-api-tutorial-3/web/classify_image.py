import tensorflow as tf
import tensorflow_hub as hub

import requests
from io import BytesIO
from PIL import Image
import numpy as np

original_image_cache = {}

def preprocess_image(image):
  image = np.array(image)
  print(image.shape)
  # reshape into shape [batch_size, height, width, num_channels]
  img_reshaped = tf.reshape(image, [1, image.shape[0], image.shape[1], image.shape[2]])
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  image = tf.image.convert_image_dtype(img_reshaped, tf.float32)
  return image

def load_image_from_url(url):
  """Returns an image with shape [1, height, width, num_channels]."""
  response = requests.get(url)
  image = Image.open(BytesIO(response.content))
  image = preprocess_image(image)
  return image

def load_image(image_url, image_size=256, dynamic_size=False, max_dynamic_size=512):
  """Loads and preprocesses images."""
  # Cache image file locally.
  
  if image_url in original_image_cache:
    img = original_image_cache[image_url]
  elif image_url.startswith('https://'):
    img = load_image_from_url(image_url)
    """
  else:
    fd = tf.io.gfile.GFile(image_url, 'rb')
    img = preprocess_image(Image.open(fd))
    """
  original_image_cache[image_url] = img
  
  # Load and convert to float32 numpy array, add batch dimension, and normalize to range [0, 1].
  img_raw = img
  if tf.reduce_max(img) > 1.0:
    img = img / 255.
  if len(img.shape) == 3:
    img = tf.stack([img, img, img], axis=-1)
  if not dynamic_size:
    img = tf.image.resize_with_pad(img, image_size, image_size)
  elif img.shape[1] > max_dynamic_size or img.shape[2] > max_dynamic_size:
    img = tf.image.resize_with_pad(img, max_dynamic_size, max_dynamic_size)
  return img, img_raw

def classify(image_url):
    image_size = 224
    dynamic_size = False

    model_name = "inception_v3" 
    

    model_image_size_map = {
    "efficientnet_b0": 224,
    "efficientnet_b1": 240,
    "efficientnet_b2": 260,
    "efficientnet_b3": 300,
    "efficientnet_b4": 380,
    "efficientnet_b5": 456,
    "efficientnet_b6": 528,
    "efficientnet_b7": 600,
    "inception_v3": 299,
    "inception_resnet_v2": 299,
    "mobilenet_v2_100_224": 224,
    "mobilenet_v2_130_224": 224,
    "mobilenet_v2_140_224": 224,
    "nasnet_large": 331,
    "nasnet_mobile": 224,
    "pnasnet_large": 331,
    "resnet_v1_50": 224,
    "resnet_v1_101": 224,
    "resnet_v1_152": 224,
    "resnet_v2_50": 224,
    "resnet_v2_101": 224,
    "resnet_v2_152": 224,
    "mobilenet_v3_small_100_224": 224,
    "mobilenet_v3_small_075_224": 224,
    "mobilenet_v3_large_100_224": 224,
    "mobilenet_v3_large_075_224": 224,
    }

    model_handle = "https://tfhub.dev/google/imagenet/inception_v3/classification/4"



    max_dynamic_size = 512
    if model_name in model_image_size_map:
        image_size = model_image_size_map[model_name]
        dynamic_size = False
        print(f"Images will be converted to {image_size}x{image_size}")
    else:
        dynamic_size = True
        print(f"Images will be capped to a max size of {max_dynamic_size}x{max_dynamic_size}")

    labels_file = "https://storage.googleapis.com/download.tensorflow.org/data/ImageNetLabels.txt"

    #download labels and creates a maps
    downloaded_file = tf.keras.utils.get_file("labels.txt", origin=labels_file)

    classes = []
    i = 0
    with open(downloaded_file) as f:
        labels = f.readlines()
        classes = [l.strip() for l in labels[1:]]
        i += 1

    image, original_image = load_image(image_url, image_size, dynamic_size, max_dynamic_size)

    classifier = hub.load(model_handle)

    input_shape = image.shape
    warmup_input = tf.random.uniform(input_shape, 0, 1.0)
    warmup_logits = classifier(warmup_input).numpy()

    probabilities = tf.nn.softmax(classifier(image)).numpy()

    top_5 = tf.argsort(probabilities, axis=-1, direction="DESCENDING")[0][:5].numpy()
    np_classes = np.array(classes)

    # Some models include an additional 'background' class in the predictions, so
    # we must account for this when reading the class labels.
    includes_background_class = probabilities.shape[1] == 1001
    jsontext = {}

    for i, item in enumerate(top_5):
        class_index = item if not includes_background_class else item - 1
        #jsontext[classes[class_index]] = str(probabilities[0][top_5][i])
        jsontext.update({classes[class_index]:str(probabilities[0][top_5][i])})

    return jsontext