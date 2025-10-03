"""
Flask application for E-commerce AI Chatbot
Integrates Groq LLM for shopping assistance and product recommendations
"""

from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import logging
import sys
from werkzeug.utils import secure_filename
from groq import Groq
import httpx

# ---------------- CONFIG ----------------
app = Flask(__name__)
# Enable CORS for API routes to allow Flutter/web integrations
CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB
app.config['SECRET_KEY'] = 'your-secret-key-here'  # For session management

# Create upload folder
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Groq client
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

# Telegram config (optional)
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_WEBHOOK_SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "telegram-secret")
# Scope: 'allergy' | 'dermatology' | 'both' (default)
BOT_SCOPE = os.getenv("BOT_SCOPE", "both").lower()

# Global disclaimers
HEALTH_DISCLAIMER = (
    "⚠️ Disclaimer: I'm an AI assistant, not a clinician. For emergencies, call emergency care. For diagnosis or treatment, see a healthcare professional."
)
ALLERGY_DISCLAIMER = HEALTH_DISCLAIMER

# Shopping features removed; chatbot-only backend

# ---------------- HELPERS ----------------
def get_session_id():
    """Generate a simple session ID for demo purposes"""
    import uuid
    return str(uuid.uuid4())[:8]

def search_products(*args, **kwargs):
    return []

def get_cart_total(*args, **kwargs):
    return 0


def query_groq(prompt, model="llama-3.1-8b-instant"):
    """Send query to Groq LLM (concise health assistant output)."""
    try:
        completion = groq_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": (
                    "You are a consumer-facing assistant for allergies and dermatology (skin). "
                    "Avoid diagnosis; give succinct self-care and red flags. "
                    "Do NOT include any disclaimers (they are added by the backend). "
                    "If outside allergies/dermatology, politely refuse. "
                    "Output must be concise and formatted exactly as: \n"
                    "Symptoms: <1 short line>.\n"
                    "Causes: <1 short line>.\n"
                    "What to do now: <up to 4 short bullets>.\n"
                    "When to get help: <one short line>."
                )},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=500,
            top_p=0.9
        )
        return completion.choices[0].message.content
    except Exception as e:
        return f"❌ AI Assistant error: {e}"

def is_allergy_related(message: str):
    """Check if message is allergy-related"""
    keywords = [
        "allergy", "allergies", "allergic", "anaphylaxis", "hives", "urticaria", "angioedema",
        "hay fever", "rhinitis", "asthma", "wheezing", "eczema", "contact dermatitis", "rash",
        "itch", "sneeze", "runny nose", "itchy eyes", "pollen", "dust", "mites", "mold", "pet dander",
        "peanut", "tree nut", "milk", "egg", "wheat", "soy", "fish", "shellfish", "latex", "penicillin",
        "antihistamine", "cetirizine", "loratadine", "fexofenadine", "diphenhydramine", "steroid", "nasal spray",
        "eye drops", "epinephrine", "epi-pen", "epipen", "skin prick", "ige", "patch test", "avoidance",
        "trigger", "immunotherapy", "allergy shots"
    ]
    text = message.lower()
    return any(k in text for k in keywords)

def is_dermatology_related(message: str):
    """Check if message is dermatology-related (non-allergy skin topics)"""
    derm_keywords = [
        "skin", "acne", "eczema", "psoriasis", "dermatitis", "rash", "pimple", "mole", "spot",
        "itch", "itchy skin", "scalp", "dandruff", "fungal", "ringworm", "athlete's foot",
        "rosacea", "hsv", "herpes", "warts", "verruca", "boil", "abscess", "ulcer", "blister",
        "sunburn", "hyperpigmentation", "melasma", "scar",
        # Skin cancer terms
        "melanoma", "skin cancer", "basal cell", "squamous cell", "bcc", "scc", "changing mole", "abcde"
    ]
    text = message.lower()
    return any(k in text for k in derm_keywords)


async def telegram_send_message(chat_id: int, text: str):
    """Send message via Telegram Bot API"""
    if not TELEGRAM_BOT_TOKEN:
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    async with httpx.AsyncClient(timeout=10) as client:
        await client.post(url, json={"chat_id": chat_id, "text": text})


# ---------------- RULES & TEMPLATES (Allergy Assistant) ----------------
SYMPTOM_KEYWORDS = {
    "sneezing": ["sneeze", "sneezing"],
    "itchy_eyes": ["itchy eyes", "itching eyes", "eye itch"],
    "runny_nose": ["runny nose", "congestion", "stuffy"],
    "rash": ["rash", "red patches", "skin rash"],
    "hives": ["hives", "urticaria", "welts"],
    "wheeze": ["wheeze", "wheezing"],
    "throat_tightness": ["throat tight", "tight throat", "throat closing"],
    "swelling": ["swelling", "swollen", "face swelling", "lip swelling", "tongue swelling"],
    "nausea": ["nausea", "vomit", "stomach", "cramp"],
}

EXPOSURE_KEYWORDS = {
    "food": ["ate", "eating", "food", "meal", "shrimp", "shellfish", "peanut", "milk", "egg", "strawberry"],
    "medication": ["medicine", "medication", "penicillin", "antibiotic", "ibuprofen", "aspirin"],
    "bee_sting": ["bee", "wasp", "sting"],
    "pollen": ["pollen", "seasonal", "spring", "grass", "trees"],
    "pet": ["cat", "dog", "pet dander"],
    "dust": ["dust", "mites"],
    "mold": ["mold"],
}

EMERGENCY_TRIGGERS = ["can't breathe", "cannot breathe", "difficulty breathing", "trouble breathing", "throat tight", "throat closing", "wheeze", "faint", "dizziness", "face swelling", "lip swelling", "tongue swelling", "swollen face", "anaphylaxis"]

NON_DROWSY_ANTIHIST = "cetirizine or loratadine (if suitable for you)"

def extract_entities(user_text: str):
    text = user_text.lower()
    import re
    # duration detection (simple)
    duration_match = re.search(r"(\d+\s*(minutes?|hours?|days?))", text)
    duration = duration_match.group(1) if duration_match else "unspecified duration"

    symptoms = []
    for name, kws in SYMPTOM_KEYWORDS.items():
        if any(k in text for k in kws):
            symptoms.append(name)

    exposure = None
    for ex, kws in EXPOSURE_KEYWORDS.items():
        if any(k in text for k in kws):
            exposure = ex
            break

    severe = any(t in text for t in EMERGENCY_TRIGGERS)
    severity = "severe" if severe else ("moderate" if any(k in text for k in ["hives", "rash", "vomit", "stomach"]) else "mild")

    return {
        "duration": duration,
        "symptoms": symptoms,
        "exposure": exposure,
        "severity": severity,
        "severe": severe,
    }

def format_sections(symptoms_desc: str, causes: list, actions: list, escalation: str):
    lines = []
    # Symptoms
    lines.append(f"Symptoms: {symptoms_desc}.")
    # Blank line for readability
    if causes:
        lines.append("")
        lines.append("Causes: " + ", ".join(causes) + ".")
    # Blank line before actions
    if actions:
        lines.append("")
        lines.append("What to do now:")
        for a in actions[:4]:
            lines.append(f"- {a}")
    # Blank line before escalation
    lines.append("")
    lines.append(f"When to get help: {escalation}")
    return "\n".join(lines)

def format_symptoms_causes_cure(symptoms_desc: str, causes: list, actions: list):
    """Compact formatting with only Symptoms, Causes, Cure (actions)."""
    lines = []
    if symptoms_desc:
        lines.append(f"Symptoms: {symptoms_desc}.")
    if causes:
        lines.append("Causes: " + ", ".join(causes) + ".")
    if actions:
        lines.append("Cure: " + "; ".join(actions[:4]) + ".")
    return "\n".join(lines)


# ---------------- ROUTES ----------------
@app.route('/')
def index():
    return render_template("index.html")
@app.route('/api/session', methods=['POST'])
def create_session():
    """Issue a new ephemeral session id for clients (e.g., Flutter)."""
    return jsonify({"session_id": get_session_id()})



@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        message = data.get('message', '').strip()
        session_id = data.get('session_id', get_session_id())
        # Optional: client-requested scope override per request
        requested_scope = (data.get('scope') or request.args.get('scope') or '').lower()
        effective_scope = requested_scope if requested_scope in {"allergy", "dermatology", "both"} else BOT_SCOPE

        # Allow greetings with onboarding
        greetings = {"hi", "hello", "hey", "start", "help"}
        if message.lower() in greetings:
            onboarding = (
                f"{ALLERGY_DISCLAIMER}\n\n"
                "Tell me your main allergy concern. Examples:\n"
                "- Sneezing and itchy eyes for 2 days\n"
                "- Hives after eating shrimp 30 minutes ago\n"
                "- Throat feels tight after a bee sting\n"
                "- Can I take cetirizine?"
            )
            return jsonify({"response": onboarding, "type": "onboarding"})

        # Enforce scope (allergy-only by default)
        if effective_scope == "allergy":
            allowed = is_allergy_related(message)
        elif effective_scope == "dermatology":
            allowed = is_dermatology_related(message) or is_allergy_related(message)
        else:  # both
            allowed = is_dermatology_related(message) or is_allergy_related(message)

        if not allowed:
            return jsonify({
                "response": (
                    "❌ I can answer only "
                    + ("allergy-related " if effective_scope == "allergy" else "dermatology/allergy ")
                    + "questions."
                ),
                "type": "restricted",
                "scope": effective_scope
            })

        entities = extract_entities(message)

        # Decision rules
        if entities["severe"]:
            actions = [
                "Use your epinephrine auto-injector immediately if prescribed",
                "Call emergency services now",
                "Lie down with legs elevated if dizzy; avoid food/drink",
            ]
            causes = ["Possible severe allergic reaction (anaphylaxis)"]
            body = format_sections(
                symptoms_desc=", ".join(entities["symptoms"]) or "Severe symptoms reported",
                causes=causes,
                actions=actions,
                escalation="Immediate emergency — go to ER"
            )
        else:
            # Use Groq to generate structured response for allergy/dermatology
            scope_text = "allergy and dermatology" if effective_scope == "both" else effective_scope
            llm_prompt = (
                "Purpose: Help users with " + scope_text + ".\n"
                "Always include a disclaimer.\n"
                "Format strictly as:\n"
                "Symptoms: <1–3 short lines>.\n"
                "Causes: <1–2 concise bullets or a single short line>.\n"
                "What to do now: <up to 4 short bullets>.\n"
                "When to get help: <one line emergency/doctor guidance>.\n\n"
                f"User message: {message}\n"
                "Only answer if within allergy or dermatology; otherwise refuse."
            )
            body = query_groq(llm_prompt)

        # Support compact output if user requests only symptoms/causes/cure
        compact = ("max 4 bullets" in message.lower()) or ("symptoms / causes / what to do now only" in message.lower()) or ("symptoms causes cure" in message.lower())
        if compact:
            # Map our actions to "cure" phrasing
            # Extract causes from body for compact output by reusing the same data from branches above.
            # Since we don't retain structured data past here, keep body as-is but prefer a simple rebuild when possible.
            compact_text = format_symptoms_causes_cure(
                symptoms_desc=", ".join(entities["symptoms"]) or "skin/allergy concern",
                causes=["see above"] if "Causes:" in body else [],
                actions=[a.replace("Use ", "Use ").replace("Take ", "Take ") for a in []]
            )
            response = f"{HEALTH_DISCLAIMER}\n\n{compact_text}".strip()
        else:
            response = f"{ALLERGY_DISCLAIMER}\n\n{body}".strip()
        response = "\n".join([line.strip() for line in response.splitlines() if line.strip()])

        return jsonify({
            "response": response,
            "type": "allergy_answer",
            "scope": effective_scope
        })

    except Exception as e:
        logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            "response": "Sorry, I encountered an error. Please try again.",
            "type": "error"
        }), 500



@app.route('/api/add-to-cart', methods=['POST'])
def add_to_cart():
    """Add product to shopping cart"""
    try:
        data = request.get_json()
        product_id = data.get('product_id')
        quantity = data.get('quantity', 1)
        session_id = data.get('session_id', get_session_id())

        # Find the product
        product = None
        for category_products in PRODUCTS_DB.values():
            for p in category_products:
                if p['id'] == product_id:
                    product = p
                    break

        if not product:
            return jsonify({"error": "Product not found"}), 404

        # Add to cart
        if session_id not in app.shopping_cart:
            app.shopping_cart[session_id] = {}
        
        if str(product_id) in app.shopping_cart[session_id]:
            app.shopping_cart[session_id][str(product_id)] += quantity
        else:
            app.shopping_cart[session_id][str(product_id)] = quantity

            return jsonify({
            "message": f"Added {product['name']} to cart",
            "cart_count": len(app.shopping_cart[session_id])
        })

    except Exception as e:
        logger.error(f"Error adding to cart: {e}")
        return jsonify({"error": "Failed to add to cart"}), 500

@app.route('/api/products', methods=['GET'])
def get_products():
    """Get all products or filter by category"""
    try:
        category = request.args.get('category')
        
        if category and category in PRODUCTS_DB:
            products = PRODUCTS_DB[category]
        else:
            # Return all products
            products = []
            for cat_products in PRODUCTS_DB.values():
                products.extend(cat_products)
        
        return jsonify({"products": products})

    except Exception as e:
        logger.error(f"Error getting products: {e}")
        return jsonify({"error": "Failed to get products"}), 500

@app.route('/api/categories', methods=['GET'])
def get_categories():
    """Get all product categories"""
    try:
        categories = list(PRODUCTS_DB.keys())
        return jsonify({"categories": categories})
    except Exception as e:
        logger.error(f"Error getting categories: {e}")
        return jsonify({"error": "Failed to get categories"}), 500

@app.route('/static/<path:filename>')
def static_files(filename):
    return send_from_directory('static', filename)

@app.route('/api/search', methods=['GET'])
def search_products_api():
    """API endpoint for product search"""
    try:
        query = request.args.get('q', '')
        category = request.args.get('category', '')
        max_price = request.args.get('max_price')
        
        if max_price:
            try:
                max_price = float(max_price)
            except ValueError:
                max_price = None
        
        results = search_products(query, category, max_price)
        return jsonify({"products": results})

    except Exception as e:
        logger.error(f"Search error: {e}")
        return jsonify({"error": "Search failed"}), 500


@app.route(f"/telegram/webhook/<secret>", methods=["POST"])
def telegram_webhook(secret):
    """Telegram webhook endpoint: processes allergy-only queries."""
    try:
        if secret != TELEGRAM_WEBHOOK_SECRET:
            return "Forbidden", 403

        data = request.get_json(force=True, silent=True) or {}
        message = data.get("message") or data.get("edited_message")
        if not message:
            return jsonify({"ok": True})

        chat_id = message.get("chat", {}).get("id")
        text = (message.get("text") or "").strip()

        if not text:
            return jsonify({"ok": True})

        if not is_allergy_related(text):
            reply = (
                "❌ I can only answer allergy-related questions. "
                "Ask about symptoms, triggers, testing, prevention, or treatments."
            )
        else:
            allergy_context = (
                f"User question: {text}. Answer ONLY if related to allergies. "
                f"Provide practical, concise guidance and safety notes; avoid diagnosis."
            )
            reply = query_groq(allergy_context)
            reply = f"{ALLERGY_DISCLAIMER}\n\n{reply}".strip()
            reply = "\n".join([line.strip() for line in reply.splitlines() if line.strip()])

        # Fire-and-forget send
        try:
            import asyncio
            asyncio.get_event_loop().create_task(telegram_send_message(chat_id, reply))
        except RuntimeError:
            # If no running loop, run a new one briefly
            import asyncio
            asyncio.run(telegram_send_message(chat_id, reply))

        return jsonify({"ok": True})
    except Exception as e:
        logger.error(f"Telegram webhook error: {e}")
        return jsonify({"ok": False}), 200


# ---------------- MAIN ----------------
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
