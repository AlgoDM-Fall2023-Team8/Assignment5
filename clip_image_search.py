import clip
import torch
import math
import numpy as np
import pandas as pd

from PIL import Image
import shutil
import os

import matplotlib.pyplot as plt
plt.rcParams.update({'pdf.fonttype': 'truetype'})
from mpl_toolkits.axes_grid1 import ImageGrid

#
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

from pathlib import Path

# Set the path to the images
images_path = Path("Apparel\Apparel\Boys\Images\images_with_product_ids")


# List all JPGs in the folder
images_files = list(images_path.glob('*.jpg'))

# Print some statistics
print(f'Images found: {len(images_files)}')

# Display images
rows, columns = (4, 4)
fig = plt.figure(figsize=(10, 10))
grid = ImageGrid(fig, 111,
                 nrows_ncols=(rows, columns),
                 axes_pad=0.1,
                 )

for ax, im in zip(grid, images_files[:rows*columns]):
    ax.imshow(Image.open(im))
    ax.axis('off')
print(plt.show())


 #
# Path where the feature vectors will be stored
#
features_path = Path(".") / "features"
if os.path.exists(features_path):
    shutil.rmtree(features_path)
os.makedirs(features_path)

# Function that computes the feature vectors for a batch of images
def compute_clip_features(images_batch):
    # Load all the images from the files
    images = [Image.open(image_file) for image_file in images_batch]

    # Preprocess all images
    images_preprocessed = torch.stack([preprocess(image) for image in images]).to(device)

    with torch.no_grad():
        # Encode the images batch to compute the feature vectors and normalize them
        images_features = model.encode_image(images_preprocessed)
        images_features /= images_features.norm(dim=-1, keepdim=True)

    # Transfer the feature vectors back to the CPU and convert to numpy
    return images_features.cpu().numpy()

# Define the batch size
batch_size = 16


# Compute how many batches are needed
batches = math.ceil(len(images_files) / batch_size)

#
# Process each batch
#
for i in range(batches):
    print(f"Processing batch {i+1}/{batches}")

    batch_ids_path = features_path / f"{i:010d}.csv"
    batch_features_path = features_path / f"{i:010d}.npy"

    # Only do the processing if the batch wasn't processed yet
    if not batch_features_path.exists():
        try:
            # Select the images for the current batch
            batch_files = images_files[i*batch_size : (i+1)*batch_size]

            # Compute the features and save to a numpy file
            batch_features = compute_clip_features(batch_files)
            np.save(batch_features_path, batch_features)

            # Save the image IDs to a CSV file
            image_ids = [image_file.name.split(".")[0] for image_file in batch_files]
            image_ids_data = pd.DataFrame(image_ids, columns=['image_id'])
            image_ids_data.to_csv(batch_ids_path, index=False)
        except Exception as e:
            # Catch problems with the processing to make the process more robust
            print(f'Problem with batch {i} {e}')


            # Load all numpy files
features_list = [np.load(features_file) for features_file in sorted(features_path.glob("*.npy"))]

# Concatenate the features and store in a merged file
features = np.concatenate(features_list)
np.save(features_path / "features.npy", features)

# Load all the image IDs
image_ids = pd.concat([pd.read_csv(ids_file) for ids_file in sorted(features_path.glob("*.csv"))])
image_ids.to_csv(features_path / "image_ids.csv", index=False)



# Load the image IDs
image_ids = pd.read_csv(features_path / 'image_ids.csv')
image_ids = list(image_ids['image_id'])

# Load the features vectors
image_features = np.load(features_path / 'features.npy')

# Convert features to Tensors: Float32 on CPU and Float16 on GPU
if device == "cpu":
    image_features = torch.from_numpy(image_features).float().to(device)
else:
    image_features = torch.from_numpy(image_features).to(device)

# Print some statistics
print(f"Images loaded: {len(image_ids)}")


def encode_search_query(search_query):
    with torch.no_grad():
        # Encode and normalize the search query using CLIP
        text_encoded = model.encode_text(clip.tokenize(search_query).to(device))
        text_encoded /= text_encoded.norm(dim=-1, keepdim=True)

    # Retrieve the feature vector
    return text_encoded

def find_best_matches(text_features, image_features, image_ids, results_count=3):
  # Compute the similarity between the search query and each image using the Cosine similarity
  similarities = (image_features @ text_features.T).squeeze(1)

  # Sort the images by their similarity score
  best_image_idx = (-similarities).argsort()

  # Return the image IDs of the best matches
  return [image_ids[i] for i in best_image_idx[:results_count]]

def search(search_query, image_features, image_ids, results_count=3):
  # Encode the search query
  text_features = encode_search_query(search_query)

  # Find the best matches
  return find_best_matches(text_features, image_features, image_ids, results_count)

#
# Search for images and visualize the results
#
search_query = 'Red T-shirt'  # Replace with your desired search query
n_results_per_query = 3  # Number of results to display


result_image_ids = search(search_query, image_features, image_ids, n_results_per_query)

fig, ax = plt.subplots(1, n_results_per_query + 1, figsize=(15, 5))

# Display the search query
ax[0].text(0.0, 0.5, search_query)
ax[0].axis('off')

# Display the search results
for j, image_id in enumerate(result_image_ids):
    image = Image.open(f'{images_path}/{image_id}.jpg')
    ax[j + 1].imshow(image)
    ax[j + 1].axis('off')