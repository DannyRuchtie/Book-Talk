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
from upload import get_db_connection

# Suppress specific warnings
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
    print(f"Attempting to load vectorstore from {vectorstore_path}")
    if os.path.exists(vectorstore_path):
        try:
            with open(vectorstore_path, 'rb') as file:
                return pickle.load(file)
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Error loading vectorstore for book '{book_key}': {e}")
            os.remove(vectorstore_path)
    else:
        print(f"Vectorstore not found for book key: {book_key}")
    return None

@chat_bp.route('/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        if not data:
            print("No JSON payload received")
            return jsonify({'error': 'No JSON payload received'}), 400
        
        question = data.get('question')
        book_id = data.get('book_id')
        if not question or not book_id:
            print("Missing 'question' or 'book_id' in request")
            return jsonify({'error': "Missing 'question' or 'book_id' in request"}), 400
        
        print(f"Received question: {question} for book ID: {book_id}")

        # Fetch book title using book_id
        conn = get_db_connection()
        book = conn.execute('SELECT title FROM books WHERE id = ?', (book_id,)).fetchone()
        conn.close()

        if book is None:
            print("Book not found")
            return jsonify({'error': 'Book not found'}), 404

        title = book['title']
        book_key = sanitize_collection_name(title.replace(" ", "_").lower())
        print(f"Book title: {title}, book key: {book_key}")

        # Load vectorstore
        vectorstore_data = load_vectorstore(book_key)
        if vectorstore_data is None:
            print("Vectorstore not found")
            return jsonify({'error': 'Vectorstore not found'}), 404

        doc_splits = vectorstore_data.get('doc_splits', [])
        embeddings_data = vectorstore_data.get('embeddings', [])

        if not isinstance(embeddings_data, GPT4AllEmbeddings):
            embeddings = GPT4AllEmbeddings()
        else:
            embeddings = embeddings_data

        vectorstore = Chroma.from_documents(
            documents=doc_splits,
            collection_name=book_key,
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

# Optional: Route for testing the chat functionality directly
@chat_bp.route('/test_chat', methods=['GET'])
def test_chat():
    return jsonify({'message': 'Chat test endpoint is active'})

if __name__ == '__main__':
    from flask import Flask
    app = Flask(__name__)
    app.register_blueprint(chat_bp, url_prefix='/api')
    app.run(debug=True)
