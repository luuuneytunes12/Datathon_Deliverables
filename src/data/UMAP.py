import pandas as pd
import random
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import matplotlib.pyplot as plt
import networkx as nx
from bertopic import BERTopic
from umap import UMAP
import re
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances

# Ensure full reproducibility
random.seed(42)
numpy_seed = np.random.seed(42)

# Load the dataset
new_topics_file_path = 'C:/Users/Admin/topics_formatted.csv'
new_topics_df = pd.read_csv(new_topics_file_path)

# Ensure all values are strings and drop NaNs
if "0.0" in new_topics_df.columns:
    text_data = new_topics_df["0.0"].dropna().astype(str).tolist()
else:
    raise ValueError("Error: Expected column '0.0' not found in CSV file.")

# Debugging: Check for non-string values
print("Sample of text_data:", text_data[:5])
print(new_topics_df.dtypes)

# Remove empty or too short documents before running BERTopic
text_data = [doc for doc in text_data if len(doc.split()) > 3]  # Keep only documents with at least 3 words

# Ensure text_data is not empty before vectorization
if not text_data:
    raise ValueError("Error: After preprocessing, there are no valid documents left. Check your dataset!")

# 1️⃣ Use BERTopic with a modified vectorizer
vectorizer = CountVectorizer(stop_words='english', token_pattern=r'(?u)\b\w+\b')  # Allows words + numbers
bertopic_model = BERTopic(vectorizer_model=vectorizer, nr_topics=10, umap_model=UMAP(random_state=42))  # Set random_state in UMAP
bertopic_topics, probs = bertopic_model.fit_transform(text_data)

# Extract topic labels
topic_info = bertopic_model.get_topic_info()

# Function to clean topic names (remove numbers and underscores)
def clean_topic_names(topic_info):
    cleaned_names = []
    for name in topic_info["Name"]:
        cleaned_name = re.sub(r'[\d_]+', ' ', name).strip()
        cleaned_name = " ".join(cleaned_name.split())  # Ensure spacing
        if not cleaned_name:  # If name is empty, use a placeholder
            cleaned_name = "General"
        cleaned_names.append(cleaned_name)
    return cleaned_names

# Apply cleaned topic names
topic_info["Cleaned Name"] = clean_topic_names(topic_info)
print("Total Topics Extracted:", len(topic_info))
print("Topics in CSV:", topic_info[["Topic", "Cleaned Name"]])

# 2️⃣ Ensure Consistent Topic Labeling
custom_labels = {topic: name for topic, name in zip(topic_info["Topic"].tolist(), topic_info["Cleaned Name"].tolist())}
bertopic_model.set_topic_labels(custom_labels)

# 3️⃣ Encode topics using `all-mpnet-base-v2`
model = SentenceTransformer('all-mpnet-base-v2')
topic_embeddings = model.encode([custom_labels.get(topic, f"Topic {topic}") for topic in bertopic_topics])

# Debug: Check if embeddings are too similar
np.set_printoptions(precision=8, suppress=True)  # Fix floating-point precision inconsistencies
print("Sample similarity values before noise:", cosine_similarity(topic_embeddings))

# Add small noise to prevent over-similarity
np.random.seed(42)  # Ensure consistent noise
noise = np.random.normal(0, 0.01, topic_embeddings.shape)
topic_embeddings += noise

# Compute similarity matrix using pairwise distances
similarity_matrix = 1 - pairwise_distances(topic_embeddings, metric='cosine')
print("Sample similarity values after noise:", similarity_matrix[:5, :5])

# 4️⃣ Visualize topics using UMAP
umap_model = UMAP(n_components=2, n_neighbors=15, min_dist=0.1, metric='cosine', random_state=42)  # Set random_state in UMAP
reduced_embeddings = umap_model.fit_transform(topic_embeddings)

plt.figure(figsize=(12, 8))
plt.scatter(reduced_embeddings[:, 0], reduced_embeddings[:, 1], cmap='viridis', s=10)
plt.title("UMAP Visualization of Standardized Article Topics")
plt.colorbar()
plt.show()


