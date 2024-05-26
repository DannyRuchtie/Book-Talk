# books.py

import os
import base64
import pickle
import sqlite3
import warnings
from flask import Flask, render_template, request, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import GPT4AllEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import Document
from upload import get_db_connection, allowed_file, process_epub
from chat import chat_bp

# Suppress specific warnings for now
warnings.filterwarnings("ignore", category=UserWarning, module="ebooklib")
warnings.filterwarnings("ignore", category=FutureWarning, module="ebooklib")

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['VECTORSTORE_DIR'] = 'vectorstores'
local_llm = 'llama3'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

if not os.path.exists(app.config['VECTORSTORE_DIR']):
    os.makedirs(app.config['VECTORSTORE_DIR'])

app.register_blueprint(chat_bp, url_prefix='/api')

# Sanitize collection name
def sanitize_collection_name(name):
    sanitized_name = ''.join(c if c.isalnum() or c in ['_', '-'] else '_' for c in name)
    sanitized_name = sanitized_name.strip('_')
    return sanitized_name[:63]

# Save vectorstore
def save_vectorstore(doc_splits, embeddings, book_key):
    data = {
        'doc_splits': doc_splits,
        'embeddings': embeddings
    }
    with open(os.path.join(app.config['VECTORSTORE_DIR'], f'{sanitize_collection_name(book_key)}.pkl'), 'wb') as file:
        pickle.dump(data, file)

# Convert row to dictionary
def dict_from_row(row):
    return dict(row)

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

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        print("No file part in the request.")
        return redirect(url_for('index'))  # Redirect to index if no file part
    file = request.files['file']
    print(f"Received file: {file.filename}")  # Debug statement
    if file.filename == '':
        print("No file selected for uploading.")
        return redirect(url_for('index'))  # Redirect to index if no file is selected
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        print(f"File {filename} saved to {file_path}")
        process_epub(file_path)
        return redirect(url_for('index'))
    print("Invalid file type.")
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
