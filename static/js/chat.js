// chat.js

document.getElementById('send-button').addEventListener('click', async () => {
    const userInput = document.getElementById('user-input').value;
    const bookId = document.getElementById('book-id').value;

    const response = await fetch('/api/chat', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ book_id: bookId, question: userInput }),
    });

    const data = await response.json();
    if (response.ok) {
        document.getElementById('chat-history').innerHTML += `<div class="user-message">${userInput}</div>`;
        document.getElementById('chat-history').innerHTML += `<div class="response-message">${data.answer}</div>`;
    } else {
        document.getElementById('chat-history').innerHTML += `<div class="error-message">${data.error}</div>`;
        if (data.details) {
            console.error('Error details:', data.details);
        }
    }

    document.getElementById('user-input').value = '';
});
