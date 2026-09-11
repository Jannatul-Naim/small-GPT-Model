const form = document.getElementById("form");
const promptBox = document.getElementById("prompt");
const chat = document.getElementById("chat");

function addMessage(cls, text) {
  const e = document.createElement("div");
  e.className = "message " + cls;
  e.textContent = text;
  chat.appendChild(e);
  chat.scrollTop = chat.scrollHeight;
  return e;
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = promptBox.value.trim();
  if (!prompt) return;

  addMessage("user", "You: " + prompt);
  const output = addMessage("assistant", "GPT: ");
  promptBox.value = "";

  const response = await fetch("/stream", {
    method: "POST",
    headers: {"Content-Type": "application/json"},
    body: JSON.stringify({
      prompt,
      use_retrieval: document.getElementById("rag").checked,
      max_new_tokens: Number(document.getElementById("maxTokens").value),
      temperature: Number(document.getElementById("temperature").value),
      top_k: 40,
      top_p: 0.92
    })
  });

  if (!response.ok) {
    output.textContent = "GPT: server error";
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const {value, done} = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, {stream: true});
    const events = buffer.split("\n\n");
    buffer = events.pop();

    for (const event of events) {
      if (!event.startsWith("data: ")) continue;
      const payload = event.slice(6);
      if (payload === "[DONE]") continue;
      const data = JSON.parse(payload);
      output.textContent += data.token;
    }
    chat.scrollTop = chat.scrollHeight;
  }
});
