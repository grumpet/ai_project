import wikipedia
import os
import chromadb
import requests
import json
from chromadb.utils import embedding_functions
import PyPDF2  # Add PyPDF2 for PDF processing



class OllamaEmbeddingFunction(embedding_functions.EmbeddingFunction):
    def __init__(self, model_name="nomic-embed-text"):
        self.model_name = model_name
        self.api_url = "http://localhost:11434/api/embeddings"
    
    def __call__(self, texts):
        embeddings = []
        for text in texts:
            response = requests.post(
                self.api_url,
                json={"model": self.model_name, "prompt": text}
            )
            if response.status_code == 200:
                result = response.json()
                embeddings.append(result["embedding"])
            else:
                raise Exception(f"Error from Ollama API: {response.text}")
        return embeddings

# Create the Ollama embedding function
ollama_ef = OllamaEmbeddingFunction(model_name="nomic-embed-text")

# Initialize the Chroma client with persistence
chroma_client = chromadb.PersistentClient(path="chroma_persistent_storage")


def get_collection_names():
    """
    Return a list of all collection names in the ChromaDB database.
    
    Returns:
        list: A list of strings containing all collection names
    """
    collections = chroma_client.list_collections()
    collection_names = [collection.name for collection in collections]
    
    return collection_names

# Function to split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap=20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks


# Function to generate embeddings using Ollama API
def get_ollama_embedding(text):
    response = requests.post(
        "http://localhost:11434/api/embeddings",
        json={"model": "nomic-embed-text", "prompt": text}
    )
    if response.status_code == 200:
        result = response.json()
        print("==== Generating embeddings... ====")
        return result["embedding"]
    else:
        raise Exception(f"Error from Ollama API: {response.text}")




print(get_collection_names())

def generate_documents(topic, collection_name=None):
    """
    Search Wikipedia for a topic, process articles into chunks,
    and store in ChromaDB collection.
    
    Args:
        topic: Topic to search on Wikipedia
        collection_name: Optional name for ChromaDB collection
    
    Returns:
        tuple: (collection object, number of documents added)
    """
    documents = []
    ids = []
    metadatas = []
    
    # Use topic name for collection if not specified
    if not collection_name:
        collection_name = f"wiki_{topic.replace(' ', '_').lower()}_collection"
    
    # Create or get collection
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=ollama_ef
    )
    
    # Search Wikipedia
    results = wikipedia.search(topic, results=5)
    print(f"Found {len(results)} articles for '{topic}'")
    
    for title in results:
        try:
            print(f"Processing: {title}")
            page = wikipedia.page(title)
            
            # Get article content and split into chunks
            chunks = split_text(page.content)
            print(f"  Split into {len(chunks)} chunks")
            
            # Process each chunk
            for i, chunk_text in enumerate(chunks):
                doc_id = f"{title.replace(' ', '_')}_{i}"
                
                # Store document text
                documents.append(chunk_text)
                
                # Store document ID
                ids.append(doc_id)
                
                # Store metadata
                metadata = {
                    "title": title,
                    "url": page.url,
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "source": "wikipedia",
                    "topic": topic
                }
                metadatas.append(metadata)
                
        except wikipedia.exceptions.DisambiguationError as e:
            print(f"  Disambiguation error for '{title}'. Options: {e.options[:3]}...")
        except wikipedia.exceptions.PageError:
            print(f"  Page '{title}' not found")
        except Exception as e:
            print(f"  Error processing '{title}': {str(e)}")
    
    # Add documents to collection if any were found
    if documents:
        collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"Added {len(documents)} document chunks to collection '{collection_name}'")
    else:
        print("No documents were added to the collection")
    
    return collection, len(documents)

def extract_topics_from_question(question, model="llama3"):
    """
    Use Ollama to extract relevant search topics from a user question
    
    Args:
        question: User's question
        model: Ollama model to use
    
    Returns:
        list: Relevant topics to search for on Wikipedia
    """
    prompt = (
        "Extract 1-3 most relevant Wikipedia topic search terms from this question. "
        "Return ONLY the topics as a comma-separated list without explanations or additional text. "
        "For example, if the question is 'How do quantum computers work?', "
        "you would return 'Quantum computing, Quantum mechanics, Quantum circuit'.\n\n"
        f"Question: {question}\n\n"
        "Topics:"
    )
    
    # Make request to Ollama's chat API
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": question}
            ],
            "stream": False
        }
    )
    
    if response.status_code == 200:
        result = response.json()
        topics_text = result["message"]["content"].strip()
        # Split the comma-separated topics and clean them up
        topics = [topic.strip() for topic in topics_text.split(",")]
        return topics
    else:
        print(f"Error extracting topics: {response.status_code}")
        return [question]  # Fall back to using the question as the topic



print(extract_topics_from_question("who is the king of england?"))
