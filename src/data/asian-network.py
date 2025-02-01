import spacy
import pandas as pd
from spacy import displacy

# Load the spaCy model
nlp = spacy.load("en_core_web_lg")


# Load the data
df = pd.read_csv(
    "../../dataset/processed/combined_articles.csv", encoding="latin1", sep=","
)
df.columns = ["Text"]


# # ----------------------------------------------------------------------


topicdict = {}  # looking for topics sentence by sentence
for n in range(len(df["Text"])):
    doc = nlp(df["Text"][n])
    topiclist = []
    for sentence in doc.sents:
        topics = [
            token.text.lower()
            for token in sentence
            if token.dep_ in ["nsubj", "dobj", "pobj"]
        ]
        for t in topics:
            topiclist.append(t)
    topicdict[n] = topiclist


for key, value in topicdict.items():
    l = []
    uselesswords = [
        "he",
        "they",
        "all",
        "them",
        "this",
        "it",
        "she",
        "her",
        "him",
        "may",
        "that",
        "who",
        "you",
        "our",
        "no",
        "yes",
        "non",
        "which",
        "when",
        "am",
        "pm",
        "we",
        "monday",
        "wednesday",
        "tuesday",
        "thursday",
        "friday",
        "saturday",
        "sunday",
        "jan",
        "feb",
        "mar",
        "apr",
        "may",
        "jun",
        "jul",
        "aug",
        "sep",
        "oct",
        "nov",
        "dec",
        "january",
        "febuary",
        "march",
        "april",
        "may",
        "june",
        "july",
        "august",
        "september",
        "october",
        "november",
        "december",
        "reuters",
        "time",
        "date",
        "case",
        "people",
    ]
    for item in value:  # filtering for repeats and filler words
        if (
            item not in l
            and item not in uselesswords
            and item.isalpha()
            and len(item) > 1
        ):
            l.append(item)
    st = " ".join(l)
    topicdict[key] = [st]
topicdf = pd.DataFrame(topicdict)
topicdf.to_csv("../../dataset/processed/topics.csv")


topicscsv = pd.read_csv("../../dataset/processed/topics.csv", header=None)
dfv = topicscsv.T
dfv.to_csv("../../dataset/processed/topics_formatted.csv", index=False, header=False)
# -----------------------------------------------------------------
# Load the newly uploaded dataset to understand its structure
import pandas as pd

new_topics_file_path = "../../dataset/processed/topics_formatted.csv"
new_topics_df = pd.read_csv(new_topics_file_path)


from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

new_topics_df.columns = ['Dataset', 'Sentence']  
text_data = new_topics_df['Sentence'].dropna().tolist()

# Load and clean the dataset
text_data = new_topics_df["0.0"].dropna().tolist()  # Use the `0.0` column for LDA


# Preprocess the text (if needed)
# For example, lowercase and remove punctuation/numbers (optional)


# Perform LDA
def extract_topics(articles, num_topics=5, top_n_words=10):
    # Vectorise the articles
    vectorizer = CountVectorizer(stop_words="english")
    X = vectorizer.fit_transform(articles)

    # Fit LDA model
    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    lda.fit(X)

    # Extract top words for each topic
    topics = []
    for topic_idx, topic in enumerate(lda.components_):
        top_keywords = [
            vectorizer.get_feature_names_out()[i]
            for i in topic.argsort()[-top_n_words:]
        ]
        topics.append(f"Topic {topic_idx + 1}: {', '.join(top_keywords)}")
    return topics


# Extract topics from the dataset
num_topics = 25  # Adjust based on your needs
topics = extract_topics(text_data, num_topics=num_topics, top_n_words=20)


tl = []


# Display the topics
for idx, topic in enumerate(topics):
    print(f"Cluster {idx + 1} Topic: {topic}")
    tl.append(topic)


from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import numpy as np


# Combine topic keywords into a single string for each topic


# Load Sentence-BERT model
model = SentenceTransformer("all-MiniLM-L6-v2")


# Generate embeddings for each topic
topic_embeddings = model.encode(tl)


# Compute pairwise cosine similarity
similarity_matrix = cosine_similarity(topic_embeddings)


# Create a DataFrame for the similarity matrix
similarity_df = pd.DataFrame(
    similarity_matrix,
    index=[f"Topic {i+1}" for i in range(len(tl))],
    columns=[f"Topic {i+1}" for i in range(len(tl))],
)


# Display the similarity matrix
similarity_df = similarity_df.applymap(lambda x: x if x >= 0.7 and x < 1 else np.nan)


# Optionally save the matrix to a CSV file
# similarity_df.to_csv("C:/Users/Admin/25asian_SIMILARITY.csv", index=True)
# above is already done - exists as 25TOPICS_MATRIX.csv on desktop
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


# Define a threshold for similarity to create edges
threshold = 0.7


# Create a graph
G = nx.Graph()


# Add nodes (topics)
topics = [f"Topic {i+1}" for i in range(len(similarity_df))]
G.add_nodes_from(topics)


# Add edges based on similarity threshold
for i in range(len(similarity_df)):
    for j in range(i + 1, len(similarity_df)):  # Avoid duplicates and self-loops
        if similarity_df.iloc[i, j] >= threshold:
            G.add_edge(f"Topic {i+1}", f"Topic {j+1}", weight=similarity_df.iloc[i, j])


# Visualise the graph
plt.figure(figsize=(16, 16))
pos = nx.spring_layout(G, k=100, seed=42)  # Layout for the graph
nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=700,
    node_color="skyblue",
    font_size=10,
    edge_color="gray",
    alpha=0.8,
)


# Draw edge labels (weights)
edge_labels = nx.get_edge_attributes(G, "weight")
nx.draw_networkx_edge_labels(
    G, pos, edge_labels={k: f"{v:.2f}" for k, v in edge_labels.items()}, font_size=8
)


plt.title("Social Network Map of Topics")
plt.show()


node_degrees = dict(G.degree())  # Degree of each node
sorted_degrees = sorted(
    node_degrees.items(), key=lambda x: x[1], reverse=True
)  # Sort by degree in descending order


# Get the top nodes with the highest number of connections
top_nodes = sorted_degrees[:10]  # Adjust the number as needed for more results


# Display the top connected nodes
print("Top connected nodes (topics):")
for node, degree in top_nodes:
    print(f"{node}: {degree} connections")
