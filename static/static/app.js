// Chat-only React App
const { useState, useEffect, useRef } = React;

function App() {
    const [chatOpen, setChatOpen] = useState(false);
    const [messages, setMessages] = useState([
        { type: 'bot', text: '⚠️ Disclaimer: I\'m an AI assistant, not a clinician. For emergencies, call emergency care. For diagnosis or treatment, see a healthcare professional.\n\n🩺 Welcome. Ask about allergy (hives, sneezing) or skin (dandruff, eczema, acne, moles).\n\nTip: Include symptom + duration + exposure (e.g., new product/food/sting/sun).' }
    ]);
    const [isLoading, setIsLoading] = useState(false);

    const sendMessage = async (text) => {
        const newMessage = { type: 'user', text };
        setMessages(prev => [...prev, newMessage]);
        setIsLoading(true);

        try {
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text, session_id: 'react_session', scope: 'both' })
            });
            const data = await response.json();
            setMessages(prev => [...prev, { type: 'bot', text: data.response }]);
        } catch (error) {
            setMessages(prev => [...prev, { type: 'bot', text: 'Please try again.' }]);
        }
        setIsLoading(false);
    };

    return (
        <div className="app-container">
            <ChatInterface 
                isOpen={chatOpen}
                onToggle={() => setChatOpen(!chatOpen)}
                messages={messages}
                onSendMessage={sendMessage}
                isLoading={isLoading}
            />
        </div>
    );
}

// Chat Interface Component
function ChatInterface({ isOpen, onToggle, messages, onSendMessage, isLoading }) {
    const [inputValue, setInputValue] = useState('');
    const messagesEndRef = useRef(null);

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
    };

    useEffect(scrollToBottom, [messages]);

    const handleSubmit = (e) => {
        e.preventDefault();
        if (inputValue.trim()) {
            onSendMessage(inputValue);
            setInputValue('');
        }
    };

    return (
        <>
            <button className="chat-toggle" onClick={onToggle}>
                <i className="fas fa-comments"></i>
            </button>
            {isOpen && (
            <div className={`chat-interface open`}>
                <div className="chat-header">
                    <div className="chat-title">
                        <i className="fas fa-robot"></i>
                        Allergy/Derm Assistant
                    </div>
                    <button className="close-chat" onClick={onToggle}>
                        <i className="fas fa-times"></i>
                    </button>
                </div>
                <div className="chat-messages">
                    {messages.map((message, index) => (
                        <div key={index} className={`message ${message.type}`}>
                            {message.text}
                        </div>
                    ))}
                    {isLoading && (
                        <div className="message bot">
                            <div className="loading">
                                <div className="spinner"></div>
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </div>
                <form className="chat-input" onSubmit={handleSubmit}>
                    <input
                        type="text"
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        placeholder="Ask about allergy or skin topics..."
                    />
                    <button type="submit">
                        <i className="fas fa-paper-plane"></i>
                    </button>
                </form>
            </div>
            )}
        </>
    );
}

// Render the app (React 18 compatible)
(function(){
    const container = document.getElementById('root');
    if (window.ReactDOM && ReactDOM.createRoot) {
        const root = ReactDOM.createRoot(container);
        root.render(React.createElement(App));
    } else if (window.ReactDOM) {
        ReactDOM.render(React.createElement(App), container);
    }
})();
