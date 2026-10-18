
const form = document.getElementById("form");
const promptBox = document.getElementById("prompt");
const chat = document.getElementById("chat");
const welcome = document.getElementById("welcome");

const askButton = document.getElementById("askButton");
const sendIcon = document.getElementById("sendIcon");
const clearButton = document.getElementById("clearChat");

const settingsToggle = document.getElementById("settingsToggle");
const advancedSettings = document.getElementById("advancedSettings");

let isGenerating = false;

function scrollToBottom() {
    chat.scrollTo({
        top: chat.scrollHeight,
        behavior: "smooth",
    });
}

function hideWelcome() {
    welcome.hidden = true;
}

function showWelcome() {
    welcome.hidden = false;
}

function getTime() {
    return new Intl.DateTimeFormat([], {
        hour: "2-digit",
        minute: "2-digit",
    }).format(new Date());
}

function addMessage(role, text = "") {
    const message = document.createElement("article");
    message.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "You" : "✳";
    avatar.setAttribute("aria-hidden", "true");

    const content = document.createElement("div");
    content.className = "message-content";

    const heading = document.createElement("div");
    heading.className = "message-heading";

    const name = document.createElement("span");
    name.textContent = role === "user" ? "You" : "Small-GPT";

    const time = document.createElement("time");
    time.className = "message-time";
    time.textContent = getTime();

    heading.append(name, time);

    const body = document.createElement("div");
    body.className = "message-body";
    body.textContent = text;

    content.append(heading, body);
    message.append(avatar, content);

    chat.appendChild(message);
    scrollToBottom();

    return {
        element: message,
        body,
        content,
    };
}

function setGenerating(value) {
    isGenerating = value;
    askButton.disabled = value;
    clearButton.disabled = value;
    promptBox.disabled = value;

    sendIcon.textContent = value ? "…" : "↑";
    askButton.setAttribute(
        "aria-label",
        value ? "Generating response" : "Send message"
    );

    if (value) {
        askButton.setAttribute("aria-busy", "true");
    } else {
        askButton.removeAttribute("aria-busy");
        promptBox.focus();
    }
}

function addCopyButton(message, body) {
    const actions = document.createElement("div");
    actions.className = "message-actions";

    const copyButton = document.createElement("button");
    copyButton.type = "button";
    copyButton.className = "message-action";
    copyButton.textContent = "Copy response";

    copyButton.addEventListener("click", async () => {
        try {
            await navigator.clipboard.writeText(body.textContent);
            copyButton.textContent = "Copied!";
        } catch {
            copyButton.textContent = "Copy unavailable";
        }

        setTimeout(() => {
            copyButton.textContent = "Copy response";
        }, 1500);
    });

    actions.appendChild(copyButton);
    message.content.appendChild(actions);
}

function addSources(message, sources) {
    if (!Array.isArray(sources) || sources.length === 0) {
        return;
    }

    const uniqueSources = [...new Set(sources)];

    const list = document.createElement("div");
    list.className = "source-list";

    const label = document.createElement("span");
    label.className = "source-chip";
    label.textContent = "Sources";

    list.appendChild(label);

    for (const source of uniqueSources) {
        const chip = document.createElement("span");
        chip.className = "source-chip";
        chip.textContent = source;
        list.appendChild(chip);
    }

    message.content.appendChild(list);
}

function processEvent(event, body, state) {
    if (!event.startsWith("data: ")) {
        return;
    }

    const payload = event.slice(6).trim();

    if (payload === "[DONE]") {
        state.done = true;
        return;
    }

    let data;

    try {
        data = JSON.parse(payload);
    } catch {
        return;
    }

    if (typeof data.token === "string") {
        body.textContent += data.token;
        scrollToBottom();
    }

    if (Array.isArray(data.sources)) {
        state.sources = data.sources;
    }
}

async function readStream(response, body) {
    if (!response.body) {
        throw new Error("Streaming is not supported by this response.");
    }

    const reader = response.body.getReader();
    const decoder = new TextDecoder();

    const state = {
        done: false,
        sources: [],
    };

    let buffer = "";

    try {
        while (true) {
            const { value, done } = await reader.read();

            if (done) {
                break;
            }

            buffer += decoder.decode(value, {
                stream: true,
            });

            const events = buffer.split("\n\n");
            buffer = events.pop() ?? "";

            for (const event of events) {
                processEvent(event, body, state);
            }

            if (state.done) {
                await reader.cancel();
                break;
            }
        }

        buffer += decoder.decode();

        if (buffer.trim()) {
            processEvent(buffer, body, state);
        }

        return state;
    } finally {
        reader.releaseLock();
    }
}

async function sendPrompt(rawPrompt) {
    const prompt = rawPrompt.trim();

    if (!prompt || isGenerating) {
        return;
    }

    hideWelcome();

    addMessage("user", prompt);

    const output = addMessage("assistant");
    output.element.classList.add("streaming");

    promptBox.value = "";
    promptBox.style.height = "auto";

    setGenerating(true);

    try {
        const temperature = Number(
            document.getElementById("temperature").value
        );

        const maxTokens = Number(
            document.getElementById("maxTokens").value
        );

        if (
            !Number.isFinite(temperature) ||
            temperature < 0.1 ||
            temperature > 2
        ) {
            throw new Error("Temperature must be between 0.1 and 2.");
        }

        if (
            !Number.isInteger(maxTokens) ||
            maxTokens < 1 ||
            maxTokens > 512
        ) {
            throw new Error("Max tokens must be between 1 and 512.");
        }

        const response = await fetch("/stream", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({
                prompt,
                use_retrieval: document.getElementById("rag").checked,
                max_new_tokens: maxTokens,
                temperature,
                top_k: 40,
                top_p: 0.92,
            }),
        });

        if (!response.ok) {
            const details = await response.text().catch(() => "");
            throw new Error(
                `Server error (${response.status})` +
                (details ? `: ${details.slice(0, 200)}` : "")
            );
        }

        const result = await readStream(response, output.body);

        output.element.classList.remove("streaming");

        if (!output.body.textContent.trim()) {
            output.body.textContent = result.done
                ? "The model returned an empty response."
                : "The connection ended before a response was received.";
        }

        addSources(output, result.sources);
        addCopyButton(output, output.body);
    } catch (error) {
        output.element.classList.remove("streaming");

        output.body.textContent =
            error instanceof Error
                ? `Unable to complete the request: ${error.message}`
                : "Unable to complete the request.";

        console.error("Small-GPT request failed:", error);
    } finally {
        setGenerating(false);
        scrollToBottom();
    }
}

form.addEventListener("submit", async (event) => {
    event.preventDefault();
    await sendPrompt(promptBox.value);
});

// Automatically resize the prompt field.
promptBox.addEventListener("input", () => {
    promptBox.style.height = "auto";
    promptBox.style.height = `${Math.min(promptBox.scrollHeight, 180)}px`;
});

// Ctrl+Enter or Cmd+Enter sends the prompt.
promptBox.addEventListener("keydown", (event) => {
    if (
        event.key === "Enter" &&
        (event.ctrlKey || event.metaKey)
    ) {
        event.preventDefault();
        form.requestSubmit();
    }
});

// Fill the composer when a suggested prompt is clicked.
document.querySelectorAll(".suggestion").forEach((button) => {
    button.addEventListener("click", () => {
        promptBox.value = button.dataset.prompt ?? "";
        promptBox.dispatchEvent(new Event("input"));
        promptBox.focus();
    });
});

// Clear the conversation displayed in the browser.
clearButton.addEventListener("click", () => {
    if (isGenerating) {
        return;
    }

    chat.replaceChildren(welcome);
    showWelcome();
    promptBox.focus();
});

// Show or hide the advanced settings panel.
settingsToggle.addEventListener("click", () => {
    const isExpanded =
        settingsToggle.getAttribute("aria-expanded") === "true";

    settingsToggle.setAttribute("aria-expanded", String(!isExpanded));
    advancedSettings.hidden = isExpanded;
});
