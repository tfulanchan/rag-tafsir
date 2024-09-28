import argparse
import requests  # Import requests for making API calls
from langchain_chroma import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_community.llms.ollama import Ollama

from get_embedding_function import get_embedding_function

CHROMA_PATH = "chroma"

PROMPT_TEMPLATE = """
Answer the question based only on the following context:

{context}
---

Answer the question based on the above context: {question}
"""

API_URL = "http://localhost:3000/api/query"  # Define your API endpoint here

def main():
    # Create CLI.
    parser = argparse.ArgumentParser()
    parser.add_argument("query_text", type=str, help="The query text.")
    args = parser.parse_args()
    query_text = args.query_text
    query_rag(query_text)

def query_rag(query_text: str):
    # Prepare the DB.
    embedding_function = get_embedding_function()
    db = Chroma(persist_directory=CHROMA_PATH, embedding_function=embedding_function)

    # Search the DB.
    results = db.similarity_search_with_score(query_text, k=5)

    context_text = "\n\n---\n\n".join([doc.page_content for doc, _score in results])
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=query_text)

    model = Ollama(model="mistral")
    response_text = model.invoke(prompt)

    # Prepare data for POST request
    payload = {
        "query": query_text,
        "response": response_text,
        "context": context_text,
        "sources": [doc.metadata.get("id", None) for doc, _score in results]
    }

    # Make a POST request to the API
    try:
        api_response = requests.post(API_URL, json=payload)
        api_response.raise_for_status()  # Raise an error for bad responses
        print("API Response:", api_response.json())  # Print the response from the API
    except requests.exceptions.RequestException as e:
        print(f"Error during API call: {e}")

    formatted_response = f"Response: {response_text}\nSources: {payload['sources']}"
    print(formatted_response)
    
    return response_text

if __name__ == "__main__":
    main()