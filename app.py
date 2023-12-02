import streamlit as st

from matplotlib import offsetbox
import numpy as np
from tqdm import tqdm


def Image_Match():
    import streamlit as st
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras.applications.vgg19 import preprocess_input
    from tensorflow.keras.models import Model

    import matplotlib.pyplot as plt
    plt.rcParams.update({'pdf.fonttype': 'truetype'})
    from matplotlib import offsetbox
    import numpy as np
    from tqdm import tqdm

    import glob
    import ntpath
    import cv2

    from sklearn.metrics.pairwise import cosine_similarity
    from sklearn import manifold
    import scipy as sc


    st.title("Explore Image Styles")



    import streamlit as st
    import cv2
    import ntpath
    import matplotlib.pyplot as plt
    import glob

    # Replace this with the full path to your directory
    directory_path = 'Data'

    # Use the directory_path in the glob function
    image_paths = glob.glob(f'{directory_path}/*.jpg')

    # Rest of the code remains the same
    #st.write(f'Found [{len(image_paths)}] images')

    images = {}
    for image_path in image_paths:
        image = cv2.imread(image_path, 3)
        b, g, r = cv2.split(image)           # get b, g, r
        image = cv2.merge([r, g, b])         # switch it to r, g, b
        image = cv2.resize(image, (200, 200))
        images[ntpath.basename(image_path)] = image


    import ssl

    ssl._create_default_https_context = ssl._create_unverified_context

    def load_image(image):
        image = plt.imread(image)
        img = tf.image.convert_image_dtype(image, tf.float32)
        img = tf.image.resize(img, [400, 400])
        img = img[tf.newaxis, :] # shape -> (batch_size, h, w, d)
        return img

    #
    # content layers describe the image subject
    #
    content_layers = ['block5_conv2'] 

    #
    # style layers describe the image style
    # we exclude the upper level layes to focus on small-size style details
    #
    style_layers = [ 
            'block1_conv1',
            'block2_conv1',
            'block3_conv1', 
            #'block4_conv1', 
            #'block5_conv1'
        ] 

    def selected_layers_model(layer_names, baseline_model):
        outputs = [baseline_model.get_layer(name).output for name in layer_names]
        model = Model([vgg.input], outputs)
        return model

    # style embedding is computed as concatenation of gram matrices of the style layers
    def gram_matrix(input_tensor):
        result = tf.linalg.einsum('bijc,bijd->bcd', input_tensor, input_tensor)
        input_shape = tf.shape(input_tensor)

        num_locations = tf.cast(input_shape[1]*input_shape[2], tf.float32)
        return result/(num_locations)

    class StyleModel(tf.keras.models.Model):
        def _init_(self, style_layers, content_layers):
            super(StyleModel, self)._init_()
            self.vgg =  selected_layers_model(style_layers + content_layers, vgg)
            self.style_layers = style_layers
            self.content_layers = content_layers
            self.num_style_layers = len(style_layers)
            self.vgg.trainable = False

        def call(self, inputs):
            # scale back the pixel values
            inputs = inputs*255.0
            # preprocess them with respect to VGG19 stats
            preprocessed_input = preprocess_input(inputs)
            # pass through the reduced network
            outputs = self.vgg(preprocessed_input)
            # segregate the style and content representations
            style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                            outputs[self.num_style_layers:])

            # calculate the gram matrix for each layer
            style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

            # assign the content representation and gram matrix in
            # a layer by layer fashion in dicts
            content_dict = {content_name:value
                        for content_name, value
                        in zip(self.content_layers, content_outputs)}

            style_dict = {style_name:value
                    for style_name, value
                    in zip(self.style_layers, style_outputs)}

            return {'content':content_dict, 'style':style_dict}


    def load_vgg19_model():
        vgg = tf.keras.applications.VGG19(include_top=False, weights=None)
        vgg.load_weights('vgg19_weights.h5')
        return vgg

    # Load the VGG19 model
    vgg = load_vgg19_model()


    def image_to_style(image_tensor):
        extractor = StyleModel(style_layers, content_layers)
        return extractor(image_tensor)['style']

    def style_to_vec(style):
        # concatenate gram matrics in a flat vector
        return np.hstack([np.ravel(s) for s in style.values()]) 

    #
    # Print shapes of the style layers and embeddings
    #
    image_tensor = load_image(image_paths[0])
    style_tensors = image_to_style(image_tensor)
    for k,v in style_tensors.items():
        print(f'Style tensor {k}: {v.shape}')
    style_embedding = style_to_vec( style_tensors )
    print(f'Style embedding: {style_embedding.shape}')

    image_style_embeddings = {}
    for image_path in tqdm(image_paths): 
        image_tensor = load_image(image_path)
        print(image_tensor)
        print(type(image_tensor))
        style = style_to_vec(image_to_style(image_tensor) )
        image_style_embeddings[ntpath.basename(image_path)] = style
