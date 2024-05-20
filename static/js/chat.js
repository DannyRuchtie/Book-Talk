// chat.js
document.addEventListener("DOMContentLoaded", function() {
  const sendButton = document.getElementById("send-button");
  const userInput = document.getElementById("user-input");
  const chatHistory = document.getElementById("chat-history");
  const bookId = document.getElementById("book-id").value;

  sendButton.addEventListener("click", function() {
    const message = userInput.value;
    if (message.trim() === "") return;

    const userMessageDiv = document.createElement("div");
    userMessageDiv.classList.add("user-message");
    userMessageDiv.textContent = message;
    chatHistory.appendChild(userMessageDiv);

    console.log(`Sending message: ${message}`);  // Logging
    fetch("/chat", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        book_id: bookId,
        question: message
      })
    })
    .then(response => response.json())
    .then(data => {
      if (data.error) {
        console.error(`Error: ${data.error}`);
        const botMessageDiv = document.createElement("div");
        botMessageDiv.classList.add("bot-message");
        botMessageDiv.textContent = `Error: ${data.error}`;
        chatHistory.appendChild(botMessageDiv);
      } else {
        console.log(`Received answer: ${data.answer}`);
        const botMessageDiv = document.createElement("div");
        botMessageDiv.classList.add("bot-message");
        botMessageDiv.textContent = data.answer;
        chatHistory.appendChild(botMessageDiv);
      }
    })
    .catch(error => {
      console.error("Error:", error);
    });

    userInput.value = "";
  });
});
