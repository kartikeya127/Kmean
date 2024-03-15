import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Function for KMeans segmentation
def kmeans_segmentation(img_combine, n_clusters):
    img_shape = img_combine.shape
    img_2d = img_combine.reshape(img_shape[0] * img_shape[1], img_shape[2])

    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(img_2d)
    labels = kmeans.labels_

    labels_2d = np.reshape(labels, (img_shape[0], img_shape[1]))
    return labels_2d

# Function to display segmented image
def plot_kmeans_segmentation(labels_2d):
    fig, ax = plt.subplots()
    ax.imshow(labels_2d)
    ax.axis('off')  # Turn off axis labels
    st.pyplot(fig)

def main():
    st.title("KMeans Image Segmentation")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Read the image
        image = plt.imread(uploaded_file)

        st.image(image, caption='Uploaded Image', use_column_width=True)

        # Slider for selecting number of clusters
        num_clusters = st.slider("Select number of clusters", min_value=2, max_value=10, value=3)

        # Perform KMeans segmentation
        labels = kmeans_segmentation(image, num_clusters)

        st.subheader("Segmented Image")
        # Display segmented image
        plot_kmeans_segmentation(labels)

if __name__ == '__main__':
    main()
