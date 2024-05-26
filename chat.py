import os
import pickle
import warnings
from flask import Blueprint, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from upload import get_db_connection

# Suppress specific warnings for now
warnings.filterwarnings("ignore", category=UserWarning, module="ebooklib")
warnings.filterwarnings("ignore", category=FutureWarning, module="ebooklib")

chat_bp = Blueprint('chat_bp', __name__)
local_llm = 'llama3'

VECTORSTORE_DIR = 'vectorstores'
if not os.path.exists(VECTORSTORE_DIR):
    os.makedirs(VECTORSTORE_DIR)

# Sanitize collection name
def sanitize_collection_name(name):
    sanitized_name = ''.join(c if c.isalnum() or c in ['_', '-'] else '_' for c in name)
    sanitized_name = sanitized_name.strip('_')
    return sanitized_name[:63]

# Load vectorstore
def load_vectorstore(book_key):
    vectorstore_path = os.path.join(VECTORSTORE_DIR, f'{book_key}.pkl')
    if os.path.exists(vectorstore_path):
        try:
            with open(vectorstore_path, 'rb') as file:
                print(f"Loading vectorstore from {vectorstore_path}")
                return pickle.load(file)
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Error loading vectorstore for book '{book_key}': {e}")
            os.remove(vectorstore_path)
    return None

# Route for handling chat requests
@chat_bp.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        question = data.get('question')
        book_id = data.get('book_id')
        print(f"Received question: {question} for book ID: {book_id}")

        # Fetch book title using book_id
        conn = get_db_connection()
        book = conn.execute('SELECT title FROM books WHERE id = ?', (book_id,)).fetchone()
        conn.close()

        if book is None:
            return jsonify({'error': 'Book not found'}), 404

        title = book['title']
        book_key = sanitize_collection_name(title.replace(" ", "_").lower())
        print(f"Book title: {title}, book key: {book_key}")

        # Load vectorstore
        vectorstore_data = load_vectorstore(book_key)
        if vectorstore_data is None:
            print(f"Vectorstore not found for book key: {book_key}")
            return jsonify({'error': 'Vectorstore not found'}), 404

        doc_splits = vectorstore_data.get('doc_splits', [])
        embeddings_data = vectorstore_data.get('embeddings', [])

        print(f"Document splits: {len(doc_splits)}, Embeddings data: {type(embeddings_data)}")

        # Initialize embeddings if not already an instance
        if not isinstance(embeddings_data, GPT4AllEmbeddings):
            embeddings = GPT4AllEmbeddings()
        else:
            embeddings = embeddings_data

        print(f"Loaded vectorstore for book key: {book_key}, doc_splits: {len(doc_splits)}, embeddings: {embeddings}")

        # Validate embeddings
        if not hasattr(embeddings, 'client'):
            print("Embeddings object does not have 'client' attribute")
            return jsonify({'error': 'Embeddings initialization error'}), 500

        # Create retriever and prompt template
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name=book_key,  # Use sanitized book key
            embedding=embeddings
        )
        retriever = vectorstore.as_retriever()

        prompt_template = PromptTemplate(
            template="Try to use only the provided context and make it clear when your answer is not coming from the book: {context}, answer the following question: {question}",
            input_variables=["context", "question"]
        )
        llm = ChatOllama(model=local_llm)
        answer_generator = prompt_template | llm | StrOutputParser()

        # Retrieve and generate answer
        print(f"Invoking retriever with question: {question}")
        retrieved_docs = retriever.retrieve(question, num_results=10)
        print(f"Retrieved {len(retrieved_docs)} documents for question: {question}")

        if retrieved_docs:
            combined_context = " ".join([doc.page_content for doc in retrieved_docs])
            print(f"Combined context for answer generation: {combined_context[:500]}...")  # Show a snippet for debugging
            try:
                answer = answer_generator.invoke({"context": combined_context, "question": question})
                print(f"Generated answer: {answer}")
                return jsonify({'answer': answer})
            except Exception as e:
                print(f"Error generating answer: {e}")
                return jsonify({'error': 'Error generating answer', 'details': str(e)}), 500
        else:
            print("No relevant documents found")
            return jsonify({'error': 'No relevant documents found'}), 404
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred', 'details': str(e)}), 500

# Route for testing the chat functionality directly
@chat_bp.route('/test_chat', methods=['GET'])
def test_chat():
    try:
        book_id = 1
        question = "what is this book about?"
        print(f"Testing question: {question} for book ID: {book_id}")

        # Fetch book title using book_id
        conn = get_db_connection()
        book = conn.execute('SELECT title FROM books WHERE id = ?', (book_id,)).fetchone()
        conn.close()

        if book is None:
            return jsonify({'error': 'Book not found'}), 404

        title = book['title']
        book_key = sanitize_collection_name(title.replace(" ", "_").lower())
        print(f"Book title: {title}, book key: {book_key}")

        # Load vectorstore
        vectorstore_data = load_vectorstore(book_key)
        if vectorstore_data is None:
            print(f"Vectorstore not found for book key: {book_key}")
            return jsonify({'error': 'Vectorstore not found'}), 404

        doc_splits = vectorstore_data.get('doc_splits', [])
        embeddings_data = vectorstore_data.get('embeddings', [])

        print(f"Document splits: {len(doc_splits)}, Embeddings data: {type(embeddings_data)}")

        # Initialize embeddings if not already an instance
        if not isinstance(embeddings_data, GPT4AllEmbeddings):
            embeddings = GPT4AllEmbeddings()
        else:
            embeddings = embeddings_data

        print(f"Loaded vectorstore for book key: {book_key}, doc_splits: {len(doc_splits)}, embeddings: {embeddings}")

        # Validate embeddings
        if not hasattr(embeddings, 'client'):
            print("Embeddings object does not have 'client' attribute")
            return jsonify({'error': 'Embeddings initialization error'}), 500

        # Create retriever and prompt template
        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name=book_key,  # Use sanitized book key
            embedding=embeddings
        )
        retriever = vectorstore.as_retriever()

        prompt_template = PromptTemplate(
            template="Try to use only the provided context and make it clear when your answer is not coming from the book: {context}, answer the following question: {question}",
            input_variables=["context", "question"]
        )
        llm = ChatOllama(model=local_llm)
        answer_generator = prompt_template | llm | StrOutputParser()

        # Retrieve and generate answer
        print(f"Invoking retriever with question: {question}")
        retrieved_docs = retriever.retrieve(question, num_results=10)
        print(f"Retrieved {len(retrieved_docs)} documents for question: {question}")

        if retrieved_docs:
            combined_context = " ".join([doc.page_content for doc in retrieved_docs])
            print(f"Combined context for answer generation: {combined_context[:500]}...")  # Show a snippet for debugging
            try:
                answer = answer_generator.invoke({"context": combined_context, "question": question})
                print(f"Generated answer: {answer}")
                return jsonify({'answer': answer})
            except Exception as e:
                print(f"Error generating answer: {e}")
                return jsonify({'error': 'Error generating answer', 'details': str(e)}), 500
        else:
            print("No relevant documents found")
            return jsonify({'error': 'No relevant documents found'}), 404
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred', 'details': str(e)}), 500
