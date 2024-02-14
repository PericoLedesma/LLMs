import chromadb


chroma_client= chromadb.Client()

collection = chroma_client.create_collection(name='my_collection')

collection.add(
    documents = ['My name is Pedro', 'My name is not Pedro'],
    metadatas = [{'source': "name is true"}, {'source': "name is false"}],
    ids = ['ide1', 'ide2']
)

results = collection.query(
    # Cosine similarity function
    query_texts = ['what is my name?'],
    n_results = 2
)

print(results)