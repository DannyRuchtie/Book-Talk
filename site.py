import base64
from flask import Flask, render_template, request
import sqlite3

app = Flask(__name__)

# Database connection
def get_db_connection():
    conn = sqlite3.connect('books.db')
    conn.row_factory = sqlite3.Row
    return conn

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

    return render_template('detail.html', book=book)

if __name__ == '__main__':
    app.run(debug=True)