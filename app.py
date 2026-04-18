import os
import json
import datetime
import re
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit, join_room, leave_room
import google.generativeai as genai

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'kidschat-secret-2024')
CORS(app, origins="*")
socketio = SocketIO(app, cors_allowed_origins="*")

GOOGLE_API_KEY   = os.environ.get("GOOGLE_API_KEY")
PARENT_PASSWORD  = os.environ.get("PARENT_PASSWORD", "parent123")

if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
    ai_model = genai.GenerativeModel('gemini-2.5-flash-lite')
else:
    ai_model = None

# ── Files ──
MESSAGES_FILE = "messages.json"
USERS_FILE    = "users.json"

def load_json(path, default):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return default

def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# ── Bad words filter ──
BAD_WORDS = [
    "fuck", "shit", "bitch", "ass", "bastard", "dick", "cock",
    "pussy", "cunt", "whore", "slut", "nigga", "faggot", "hell", "damn",
    "හුත්ත", "පුකේ", "මෝඩ", "හරක", "hutta", "puke", "moda", "harak",
    "stupid", "idiot", "dumb", "hate you", "kill", "die"
]

def filter_message(text):
    text_lower = text.lower()
    for word in BAD_WORDS:
        if word in text_lower:
            # replace with stars
            pattern = re.compile(re.escape(word), re.IGNORECASE)
            text = pattern.sub('⭐' * len(word), text)
    return text

def has_bad_words(text):
    text_lower = text.lower()
    return any(w in text_lower for w in BAD_WORDS)

# Online users tracking
online_users = {}  # sid -> {name, avatar, room, joined}

# ── HTTP Routes ──
@app.route('/health')
def health():
    return jsonify({"status": "ok"})

@app.route('/parent/messages', methods=['GET'])
def parent_messages():
    pwd = request.headers.get('X-Parent-Password', '')
    if pwd != PARENT_PASSWORD:
        return jsonify({"error": "Unauthorized"}), 401
    messages = load_json(MESSAGES_FILE, [])
    users    = load_json(USERS_FILE, {})
    return jsonify({
        "messages": messages[-200:],
        "total": len(messages),
        "online_users": list(online_users.values()),
        "registered_users": users
    })

@app.route('/parent/delete_message', methods=['POST'])
def delete_message():
    pwd = request.headers.get('X-Parent-Password', '')
    if pwd != PARENT_PASSWORD:
        return jsonify({"error": "Unauthorized"}), 401
    data = request.json
    msg_id = data.get('id')
    messages = load_json(MESSAGES_FILE, [])
    messages = [m for m in messages if m.get('id') != msg_id]
    save_json(MESSAGES_FILE, messages)
    return jsonify({"status": "ok"})

# ── Socket Events ──
@socketio.on('connect')
def on_connect():
    print(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def on_disconnect():
    if request.sid in online_users:
        user = online_users.pop(request.sid)
        emit('user_left', {'name': user['name'], 'online_count': len(online_users)}, broadcast=True)

@socketio.on('join')
def on_join(data):
    name   = str(data.get('name', 'Guest'))[:20]
    avatar = data.get('avatar', '🐱')
    room   = data.get('room', 'general')

    online_users[request.sid] = {
        'name': name,
        'avatar': avatar,
        'room': room,
        'joined': datetime.datetime.now().isoformat()
    }

    join_room(room)

    # Send last 50 messages to new user
    messages = load_json(MESSAGES_FILE, [])
    room_msgs = [m for m in messages if m.get('room') == room][-50:]
    emit('history', {'messages': room_msgs})

    # Notify others
    emit('user_joined', {
        'name': name,
        'avatar': avatar,
        'online_count': len(online_users),
        'online_users': list(online_users.values())
    }, broadcast=True)

@socketio.on('send_message')
def on_message(data):
    if request.sid not in online_users:
        return

    user    = online_users[request.sid]
    content = str(data.get('content', '')).strip()[:500]
    room    = user.get('room', 'general')

    if not content:
        return

    flagged = has_bad_words(content)
    clean   = filter_message(content)

    msg = {
        'id': f"{request.sid}-{datetime.datetime.now().timestamp()}",
        'name': user['name'],
        'avatar': user['avatar'],
        'content': clean,
        'original': content,
        'flagged': flagged,
        'room': room,
        'time': datetime.datetime.now().isoformat(),
        'type': 'user'
    }

    # Save
    messages = load_json(MESSAGES_FILE, [])
    messages.append(msg)
    messages = messages[-1000:]
    save_json(MESSAGES_FILE, messages)

    # Broadcast
    emit('new_message', msg, room=room)

    # Warning to sender if flagged
    if flagged:
        emit('warning', {'message': '⚠️ Please use kind words! Your message was filtered.'})

@socketio.on('ai_message')
def on_ai_message(data):
    if request.sid not in online_users:
        return

    user    = online_users[request.sid]
    content = str(data.get('content', '')).strip()[:300]

    if not content or not ai_model:
        emit('ai_reply', {'content': '🤖 AI is not available right now!'})
        return

    try:
        prompt = f"""You are a friendly, fun AI assistant for children aged 6-14.
Always be positive, encouraging, and age-appropriate.
Never discuss violence, adult content, or anything inappropriate for children.
Keep replies short (2-4 sentences), fun, and use emojis!
Child's name: {user['name']}
Child asks: {content}"""

        response = ai_model.generate_content(prompt)
        reply    = response.text[:400]

        ai_msg = {
            'id': f"ai-{datetime.datetime.now().timestamp()}",
            'name': '🤖 Buddy AI',
            'avatar': '🤖',
            'content': reply,
            'time': datetime.datetime.now().isoformat(),
            'type': 'ai'
        }

        # Save AI messages too
        messages = load_json(MESSAGES_FILE, [])
        messages.append({**ai_msg, 'room': user.get('room', 'general'), 'flagged': False})
        save_json(MESSAGES_FILE, messages)

        emit('ai_reply', ai_msg)

    except Exception as e:
        emit('ai_reply', {'content': f'🤖 Oops! I had a hiccup. Try again! 😅'})

@socketio.on('typing')
def on_typing(data):
    if request.sid in online_users:
        user = online_users[request.sid]
        emit('user_typing', {'name': user['name']}, broadcast=True, include_self=False)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
