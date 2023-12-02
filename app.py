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