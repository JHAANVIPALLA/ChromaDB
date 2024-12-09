import csv
import chromadb
from chromadb.utils import embedding_functions


# Read items from CSV
with open('items.csv') as file:
    lines = csv.reader(file)

    documents = []
    metadata = []
    ids = []
    id = 1

    for i, line in enumerate(lines):
        if i == 0:
            continue

        documents.append(line[1])
        metadata.append({"item_id": line[0]})
        ids.append(str(id))
        id += 1

# Initialize Chroma client and create a collection
client = chromadb.Client()
sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="all-mpnet-base-v2")
collection = client.create_collection(name="my_collection", embedding_function=sentence_transformer_ef)

# Add data to collection
collection.add(
    ids=ids,
    documents=documents,
    metadatas=metadata
)

# Perform a query on the collection
results = collection.query(
    query_texts=["shrimp"],  # Correct argument for text-based queries
    n_results=5,
    include=["documents"]
)

# Print the results
print(results)
