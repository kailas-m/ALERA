(function () {
	const STYLE_ID = "chatbot-widget-styles";
	const STORAGE_KEY = "chatbotWidgetMessages";

	function injectStyles() {
		if (document.getElementById(STYLE_ID)) return;
		const style = document.createElement("style");
		style.id = STYLE_ID;
		style.textContent = `
			#cb-launcher { position: fixed; right: 20px; bottom: 20px; z-index: 9999; }
			#cb-launcher button { background: #2563eb; color: #fff; border: none; border-radius: 999px; width: 56px; height: 56px; box-shadow: 0 8px 20px rgba(0,0,0,0.2); cursor: pointer; font-size: 22px; }
			#cb-launcher button:hover { background: #1d4ed8; }

			#cb-window { position: fixed; right: 20px; bottom: 90px; width: 320px; max-width: calc(100vw - 40px); height: 440px; max-height: calc(100vh - 120px); background: #ffffff; border: 1px solid #e5e7eb; border-radius: 12px; overflow: hidden; display: none; flex-direction: column; z-index: 9999; box-shadow: 0 12px 28px rgba(0,0,0,0.2); }
			#cb-header { background: #111827; color: #fff; padding: 12px 14px; display: flex; align-items: center; justify-content: space-between; }
			#cb-header .title { font-weight: 600; font-size: 14px; }
			#cb-header .actions button { background: transparent; color: #fff; border: none; cursor: pointer; font-size: 16px; margin-left: 8px; }

			#cb-messages { flex: 1; padding: 12px; overflow-y: auto; background: #f9fafb; }
			.cb-msg { max-width: 85%; margin: 8px 0; padding: 10px 12px; border-radius: 12px; line-height: 1.4; font-size: 13px; white-space: pre-wrap; word-wrap: break-word; }
			.cb-msg.user { background: #2563eb; color: #fff; margin-left: auto; border-bottom-right-radius: 4px; }
			.cb-msg.bot { background: #e5e7eb; color: #111827; margin-right: auto; border-bottom-left-radius: 4px; }
			.cb-hint { color: #6b7280; font-size: 12px; padding: 0 12px 8px; }

			#cb-input { border-top: 1px solid #e5e7eb; padding: 10px; background: #fff; display: flex; gap: 8px; }
			#cb-input textarea { flex: 1; resize: none; border: 1px solid #e5e7eb; border-radius: 8px; padding: 8px 10px; height: 40px; font-size: 13px; }
			#cb-input button { background: #2563eb; color: #fff; border: none; border-radius: 8px; padding: 0 14px; cursor: pointer; font-size: 13px; }
			#cb-input button:disabled { background: #93c5fd; cursor: not-allowed; }
		`;
		document.head.appendChild(style);
	}

	function createUI() {
		const launcher = document.createElement("div");
		launcher.id = "cb-launcher";
		launcher.innerHTML = `<button aria-label="Open chat" title="Chat">💬</button>`;

		const win = document.createElement("div");
		win.id = "cb-window";
		win.innerHTML = `
			<div id="cb-header">
				<div class="title">Assistant</div>
				<div class="actions">
					<button id="cb-minimize" title="Minimize">—</button>
					<button id="cb-close" title="Close">✕</button>
				</div>
			</div>
			<div class="cb-hint">Ask about this site, uploading DNA images, results, or contact info.</div>
			<div id="cb-messages" role="log" aria-live="polite"></div>
			<div id="cb-input">
				<textarea id="cb-text" placeholder="Type a message..."></textarea>
				<button id="cb-send">Send</button>
			</div>
		`;

		document.body.appendChild(launcher);
		document.body.appendChild(win);

		return { launcher, win };
	}

	function loadHistory() {
		try {
			const raw = localStorage.getItem(STORAGE_KEY);
			return raw ? JSON.parse(raw) : [];
		} catch (_) { return []; }
	}

	function saveHistory(messages) {
		try { localStorage.setItem(STORAGE_KEY, JSON.stringify(messages.slice(-100))); } catch (_) {}
	}

	function appendMessage(container, author, text) {
		const div = document.createElement("div");
		div.className = `cb-msg ${author}`;
		div.textContent = text;
		container.appendChild(div);
		container.scrollTop = container.scrollHeight;
	}

	async function getResponse(userText) {
		const t = (userText || "").toLowerCase().trim();
		if (!t) return "Please type a message.";

		// Quick intents
		if (/^(hi|hello|hey)\b/.test(t)) return "Hi! How can I help you today?";
		if (/help|support|assist/.test(t)) return "I can help with uploading DNA images, viewing results, and general site info.";
		if (/upload|image|photo/.test(t)) return "Use the Upload page to submit a clear DNA gel image (JPG/PNG).";
		if (/result|prediction|output/.test(t)) return "After upload, the results page will show your prediction and details.";
		if (/account|login|sign ?in/.test(t)) return "Use the Login page. New here? Try Sign Up to create an account.";
		if (/sign ?up|register|create account/.test(t)) return "Open the Sign Up page, fill the form, and submit to register.";
		if (/contact|email|reach/.test(t)) return "Check the Contact page for ways to reach us.";
		if (/about|what.*site|who are you/.test(t)) return "See the About page to learn more about this project.";
		if (/file|format|type/.test(t)) return "Supported image formats: JPG, JPEG, and PNG.";
		if (/privacy|data|secure|security/.test(t)) return "We only use your uploaded image to generate predictions; keep your files secure.";

		// Try backend chat if available
		try {
			const resp = await fetch('/chat', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ message: userText }) });
			if (resp.ok) {
				const data = await resp.json();
				if (data && data.reply) return data.reply;
			}
		} catch (_) {}

		// Fallback: short, friendly default
		return "Got it! I’m a simple on-page helper. Ask about uploads, results, or navigation.";
	}

	function init() {
		injectStyles();
		const { launcher, win } = createUI();
		const messagesEl = win.querySelector("#cb-messages");
		const inputEl = win.querySelector("#cb-text");
		const sendBtn = win.querySelector("#cb-send");

		let history = loadHistory();
		if (history.length === 0) {
			history = [{ author: "bot", text: "Welcome! Need help uploading a DNA image or finding a page?" }];
			saveHistory(history);
		}
		history.forEach(m => appendMessage(messagesEl, m.author, m.text));

		function openChat() { win.style.display = "flex"; inputEl.focus(); }
		function closeChat() { win.style.display = "none"; }

		launcher.addEventListener("click", openChat);
		win.querySelector("#cb-close").addEventListener("click", closeChat);
		win.querySelector("#cb-minimize").addEventListener("click", () => { win.style.display = "none"; });

		function send() {
			const text = (inputEl.value || "").trim();
			if (!text) return;
			appendMessage(messagesEl, "user", text);
			history.push({ author: "user", text });
			saveHistory(history);
			inputEl.value = "";
			sendBtn.disabled = true;
			(async () => {
				const reply = await getResponse(text);
				appendMessage(messagesEl, "bot", reply);
				history.push({ author: "bot", text: reply });
				saveHistory(history);
				sendBtn.disabled = false;
				inputEl.focus();
			})();
		}

		sendBtn.addEventListener("click", send);
		inputEl.addEventListener("keydown", (e) => {
			if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); send(); }
		});
	}

	if (document.readyState === "loading") {
		document.addEventListener("DOMContentLoaded", init);
	} else {
		init();
	}
})();


