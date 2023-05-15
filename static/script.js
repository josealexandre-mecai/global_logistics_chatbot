function sendMessage(event) {
    // Prevents the page from reloading when submitting the form
    event.preventDefault();
  
    // Get the user message from the input field
    const userInput = document.getElementById("user_input").value;
    console.log("userInput = " + userInput)
  
    // Only send the message if the user has typed something
    if (userInput.trim() !== "") {
  
      // Create a new chat message with the user's input
      const userMessage = createMessage("user", userInput);
      appendMessage(userMessage);
      console.log("userMessage = " + userMessage)
  
      // Send the message to the server and wait for a response
      fetch("get_bot_response", {
          method: "POST",
          headers: {
              "Content-Type": "application/json"
          },
          body: JSON.stringify({ user_input: userInput })
      })
      .then(response => {
          if (!response.ok) {
              throw new Error("Network response was not ok");
          }
          const contentType = response.headers.get("content-type");
          if (contentType && contentType.includes("application/json")) {
              return response.json();
          } else {
              throw new TypeError("Response was not JSON");
          }
      })
      .then(data => {
          // Create a new chat message with the bot's response
          const botMessage = createMessage("bot", data.bot_response);
          console.log("botMessage = " + botMessage)
          appendMessage(botMessage);
      })
      .catch(error => console.error(error));
  
      // Clear the input field
      document.getElementById("user_input").value = "";
    }
  }
  
  
function createMessage(sender, text) {
    // Create a new chat message element
    const message = document.createElement("div");
    message.classList.add("chat-message");
    message.classList.add(`chat-${sender}-message`);
  
    // Add the message text to the element
    const messageContent = document.createElement("div");
    messageContent.classList.add("chat-message-content");
    messageContent.textContent = text;
    message.appendChild(messageContent);
  
    return message;
  }
  
function appendMessage(message) {
    // Append the message to the chat history
    const chatHistory = document.getElementById("chat-history");
    chatHistory.appendChild(message);
  
    // Scroll to the bottom of the chat history
    chatHistory.scrollTop = chatHistory.scrollHeight;
  }
  