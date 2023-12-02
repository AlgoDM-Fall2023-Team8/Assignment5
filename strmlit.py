import streamlit as st

# Display the choice options to the user
option = st.selectbox("Choose the App", ("Image Match", "Image Search by Text"))

# Function for Image Match App
def Image_Match():
    import streamlit as st
    import pandas as pd
    import tensorflow as tf
    from tensorflow.keras.applications.vgg19 import preprocess_input
    from tensorflow.keras.models import Model
    print(tf.__version__)

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


    


    import streamlit as st
    import cv2
    import ntpath
    import matplotlib.pyplot as plt
    import glob

    # Replace this with the full path to your directory
    directory_path = 'Data'

    # Use the directory_path in the glob function
    image_paths = glob.glob(f'{directory_path}/*.jpg')
    st.title("Assignment 4 ")

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
        def __init__(self, style_layers, content_layers):
            super(StyleModel, self).__init__()
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
        vgg.load_weights('source/vgg19_weights.h5')
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

    #
    # compute styles
    #
    image_style_embeddings = {}
    for image_path in tqdm(image_paths): 
        image_tensor = load_image(image_path)
        print(image_tensor)
        print(type(image_tensor))
        style = style_to_vec(image_to_style(image_tensor) )
        image_style_embeddings[ntpath.basename(image_path)] = style

    import streamlit as st


    def search_similar_images(user_uploaded_image, image_style_embeddings, images, max_results=4):
        user_image_tensor = load_image(user_uploaded_image)
        user_style = style_to_vec(image_to_style(user_image_tensor))

        distances = {}
        for image_path, style_embedding in image_style_embeddings.items():
            d = sc.spatial.distance.cosine(user_style, style_embedding)
            distances[image_path] = d

        sorted_neighbors = sorted(distances.items(), key=lambda x: x[1])

        st.write("Most similar images:")
        for i, (image_path, distance) in enumerate(sorted_neighbors[:max_results]):
            st.image(images[image_path], caption=f"Distance: {distance}", use_column_width=True)

    # Streamlit UI

    st.write("Upload your image:")
    user_image = st.file_uploader("Your image", type=["jpg", "jpeg", "png"])
    if user_image:
        search_similar_images(user_image, image_style_embeddings, images)
    pass

# Function for Image Search by Text
# def Image_Search_by_Text():

#     import streamlit as st
#     import clip
#     import torch
#     import numpy as np
#     import pandas as pd
#     from PIL import Image
#     import glob
#     from pathlib import Path

#     # Load the CLIP model
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#     model, preprocess = clip.load("ViT-B/32", device=device)

#     # Load precomputed image features and image IDs
#     features_path = "features"
#     image_features = np.load(f"{features_path}/features.npy")
#     image_ids = pd.concat([pd.read_csv(file) for file in sorted(glob.glob(f"{features_path}/*.csv"))])['image_id'].tolist()

#     # Streamlit app
#     st.title("Image Search App")

#     # User input
#     search_query = st.text_input("Enter your search query:")

#     def encode_search_query(search_query):
#         with torch.no_grad():
#             text_encoded = model.encode_text(clip.tokenize(search_query).to(device))
#             text_encoded /= text_encoded.norm(dim=-1, keepdim=True)
#         return text_encoded

#     # def find_best_matches(text_features, image_features, image_ids, results_count=3):
#     #     similarities = (image_features @ text_features.T).squeeze(1)
#     #     best_image_idx = (-similarities).argsort()
#     #     return [image_ids[i] for i in best_image_idx[:results_count]]
#     import torch

#     def find_best_matches(text_features, image_features, image_ids, results_count=3):
#         text_features = text_features.to(device)  # Make sure text_features are on the same device
#         image_features_tensor = torch.tensor(image_features).to(device)  # Convert image_features to a PyTorch tensor
#         similarities = torch.mm(image_features_tensor, text_features.T).squeeze(1)
#         best_image_idx = (-similarities).argsort()
#         return [image_ids[i] for i in best_image_idx[:results_count]]


#     @st.cache_data()  # Use st.cache_data
#     def search(search_query, image_features, image_ids, results_count=3):
#         text_features = encode_search_query(search_query)
#         return find_best_matches(text_features, image_features, image_ids, results_count)

#     images_path = Path("images_with_product_ids")

#     if st.button("Search"):
#         if search_query:
#             result_image_ids = search(search_query, image_features, image_ids)

#             st.markdown(f"**Top {len(result_image_ids)} Results for '{search_query}':**")

#             for image_id in result_image_ids:
#                 image = Image.open(f'{images_path}/{image_id}.jpg')
#                 st.image(image, caption=image_id, use_column_width=True)
#                     # Display the images in one row with a specified width
#                 #st.image(image, width=150)


#     pass





def Image_Search_by_Text():
    import streamlit as st
    from serpapi import GoogleSearch
    import openai

  

    # Streamlit app
    st.title("Query Exploration App")

    # User input for query
    user_query = st.text_input("Enter your query:")

    # Function to generate explanation and answer using GPT-3
    # def generate_explanation_and_answer(query):
    #     prompt = f"Explain and answer the following question:\n{query}"
    #     response = openai.Completion.create(
    #         engine="text-davinci-002",
    #         prompt=prompt,
    #         max_tokens=150
    #     )
    #     explanation_and_answer = response['choices'][0]['text']
    #     return explanation_and_answer

    # Function to search for similar images using SerpApi
    def search_similar_images(query):
          # Replace with your actual SerpApi key
        params = {
            "q": query,
            "engine": "google_images",
            "api_key": '7437fb39559a83645ceaf9ff2fff9ef76152df6ee7cbe038f5fb9708fa43273b'
        }
        search = GoogleSearch(params)
        results = search.get_dict()
        images_results = results.get("images_results", [])
        return images_results
    # Main logic
    if user_query:
        # # Use GPT-3 to generate explanation and answer
        # explanation_and_answer = generate_explanation_and_answer(user_query)

        # st.subheader("Explanation and Answer:")
        # st.write(explanation_and_answer)

        # Image search using SerpApi
        similar_images = search_similar_images(user_query)
        print(similar_images)
        if similar_images:
            st.subheader("Similar Images:")
            for i, image_data in enumerate(similar_images[:5]):  # Limit to the first 5 images
                image_url = image_data["original"]
                st.image(image_url, caption=f"Image {i + 1}", use_column_width=True)
        else:
            st.warning("No similar images found.")
    pass





# Run the selected app based on the user's choice
if option == "Image Match":
    Image_Match()
elif option == "Image Search by Text":
    Image_Search_by_Text()
