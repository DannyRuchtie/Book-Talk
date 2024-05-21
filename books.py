import os
import base64
import pickle
import sqlite3
import warnings
from flask import Flask, render_template, request, jsonify
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document

# Suppress specific warnings for now
warnings.filterwarnings("ignore", category=UserWarning, module="ebooklib")
warnings.filterwarnings("ignore", category=FutureWarning, module="ebooklib")

app = Flask(__name__)
local_llm = 'llama3'
vectorstore_dir = 'vectorstores'

# Database connection
def get_db_connection():
    conn = sqlite3.connect('books.db')
    conn.row_factory = sqlite3.Row
    return conn

# Convert row to dictionary
def dict_from_row(row):
    return dict(row)

# Load vectorstore
def load_vectorstore(book_key):
    vectorstore_path = os.path.join(vectorstore_dir, f'{book_key}.pkl')
    if os.path.exists(vectorstore_path):
        try:
            with open(vectorstore_path, 'rb') as file:
                return pickle.load(file)
        except (EOFError, pickle.UnpicklingError) as e:
            print(f"Error loading vectorstore for book '{book_key}': {e}")
            os.remove(vectorstore_path)
    return None

# Save vectorstore
def save_vectorstore(doc_splits, embeddings, book_key):
    data = {
        'doc_splits': doc_splits,
        'embeddings': embeddings
    }
    with open(os.path.join(vectorstore_dir, f'{sanitize_collection_name(book_key)}.pkl'), 'wb') as file:
        pickle.dump(data, file)

# Sanitize collection name
def sanitize_collection_name(name):
    sanitized_name = ''.join(c if c.isalnum() or c in ['_', '-'] else '_' for c in name)
    sanitized_name = sanitized_name.strip('_')
    return sanitized_name[:63]  # Ensure it meets the length requirement

# Route for the home page
@app.route('/')
def index():
    conn = get_db_connection()
    books = conn.execute('SELECT * FROM books').fetchall()
    conn.close()

    # Convert rows to dictionaries and handle cover image
    books = [dict_from_row(book) for book in books]
    for book in books:
        if book['cover_image']:  # Check if cover_image is not None
            book['cover_image'] = base64.b64encode(book['cover_image']).decode('utf-8')
        else:
            book['cover_image'] = ''  # Handle case where cover_image is None

    return render_template('index.html', books=books)

# Route for the book detail page
@app.route('/book/<int:book_id>')
def book_detail(book_id):
    conn = get_db_connection()
    book = conn.execute('SELECT * FROM books WHERE id = ?', (book_id,)).fetchone()
    conn.close()

    if book is None:
        return "Book not found", 404

    book = dict_from_row(book)
    if book['cover_image']:
        book['cover_image'] = base64.b64encode(book['cover_image']).decode('utf-8')
    else:
        book['cover_image'] = ''

    print(f"Rendering detail page for book ID: {book_id} - Title: {book['title']}")  # Logging
    return render_template('detail.html', book=book)

# Route for handling chat requests
@app.route('/chat', methods=['POST'])
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
            answer = answer_generator.invoke({"context": combined_context, "question": question})
            print(f"Generated answer: {answer}")
            return jsonify({'answer': answer})
        else:
            return jsonify({'error': 'No relevant documents found'}), 404
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({'error': 'An error occurred', 'details': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)