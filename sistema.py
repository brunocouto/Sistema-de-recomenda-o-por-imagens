import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import os

# Load pre-trained VGG16 model
base_model = VGG16(weights="imagenet")
model = Model(inputs=base_model.input, outputs=base_model.get_layer("fc1").output)

def extract_features(image_path, model):
    """Extract features from an image using a pre-trained model."""
    image = load_img(image_path, target_size=(224, 224))
    image_array = img_to_array(image)
    image_array = np.expand_dims(image_array, axis=0)
    image_array = preprocess_input(image_array)

    features = model.predict(image_array)
    return features.flatten()

def build_feature_database(image_folder, model):
    """Build a database of image features for all images in a folder."""
    feature_database = {}
    for image_name in os.listdir(image_folder):
        image_path = os.path.join(image_folder, image_name)
        if image_name.lower().endswith(('png', 'jpg', 'jpeg')):
            features = extract_features(image_path, model)
            feature_database[image_name] = features
    return feature_database

def recommend_images(query_image_path, feature_database, model, top_n=5):
    """Recommend similar images based on a query image."""
    query_features = extract_features(query_image_path, model)

    similarities = {}
    for image_name, features in feature_database.items():
        similarity = cosine_similarity([query_features], [features])[0][0]
        similarities[image_name] = similarity

    # Sort by similarity score
    recommended_images = sorted(similarities.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return recommended_images

if __name__ == "__main__":
    # Define image folder path and query image
    image_folder = "dataset/images"
    query_image_path = "dataset/query.jpg"

    # Build feature database
    print("Building feature database...")
    feature_database = build_feature_database(image_folder, model)
    print("Feature database built successfully.")

    # Recommend images
    print("Recommending similar images...")
    recommendations = recommend_images(query_image_path, feature_database, model, top_n=5)

    print("Recommended Images:")
    for image_name, similarity in recommendations:
        print(f"{image_name}: Similarity Score = {similarity:.4f}")