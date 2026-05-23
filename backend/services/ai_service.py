import os
import re
import datetime
import google.generativeai as genai
from backend import config
from backend.database import load_json, save_json

# ── Voice & PDF libraries fallbacks (checked here or dynamically) ──
try:
    import fitz
    PDF_SUPPORT = True
except ImportError:
    PDF_SUPPORT = False

try:
    from pinecone import Pinecone
    PINECONE_SUPPORT = True
except ImportError:
    PINECONE_SUPPORT = False

# ── Gemini Configuration ──
if config.GOOGLE_API_KEY:
    genai.configure(api_key=config.GOOGLE_API_KEY)
gemini_flash = genai.GenerativeModel('gemini-2.5-flash-lite')
gemini_check = genai.GenerativeModel('gemini-2.5-flash-lite')

# ── Pinecone Setup ──
pinecone_index = None
if PINECONE_SUPPORT and config.PINECONE_API_KEY:
    try:
        pc = Pinecone(api_key=config.PINECONE_API_KEY)
        pinecone_index = pc.Index("sanz-tutor")
    except Exception as e:
        print(f"Pinecone setup error: {e}")

# ── Semantic Cache Helpers ──

def simple_vectorize(text: str) -> dict:
    words = re.findall(r"\w+", text.lower())
    vec   = {}
    for w in words:
        vec[w] = vec.get(w, 0) + 1
    total = sum(vec.values()) or 1
    return {k: v / total for k, v in vec.items()}

def cosine_similarity(vec1: dict, vec2: dict) -> float:
    common = set(vec1.keys()) & set(vec2.keys())
    if not common: return 0.0
    dot   = sum(vec1[w] * vec2[w] for w in common)
    norm1 = sum(v ** 2 for v in vec1.values()) ** 0.5
    norm2 = sum(v ** 2 for v in vec2.values()) ** 0.5
    if norm1 == 0 or norm2 == 0: return 0.0
    return dot / (norm1 * norm2)

def cache_lookup(question: str, subject: str, language: str):
    cache = load_json(config.CACHE_FILE, [])
    now   = datetime.datetime.now()
    q_vec = simple_vectorize(question)
    valid_cache = []
    for entry in cache:
        try:
            if (now - datetime.datetime.fromisoformat(entry["time"])).total_seconds() < config.CACHE_TTL_HOURS * 3600:
                valid_cache.append(entry)
        except:
            pass
    if len(valid_cache) != len(cache):
        save_json(config.CACHE_FILE, valid_cache)
    best_score, best_entry = 0.0, None
    for entry in valid_cache:
        if entry.get("subject") != subject or entry.get("language") != language:
            continue
        score = cosine_similarity(q_vec, simple_vectorize(entry["question"]))
        if score > best_score:
            best_score, best_entry = score, entry
    if best_score >= config.CACHE_SIMILARITY_THRESHOLD and best_entry:
        return best_entry, best_score
    return None, 0.0

def cache_store(question: str, subject: str, language: str, answer: str, graph_url, verified: bool):
    cache = load_json(config.CACHE_FILE, [])
    q_vec = simple_vectorize(question)
    for entry in cache:
        if entry.get("subject") == subject and entry.get("language") == language:
            if cosine_similarity(q_vec, simple_vectorize(entry["question"])) >= config.CACHE_SIMILARITY_THRESHOLD:
                return
    cache.append({
        "question": question, "subject": subject, "language": language,
        "answer": answer, "graph_url": graph_url, "verified": verified,
        "time": datetime.datetime.now().isoformat()
    })
    if len(cache) > config.CACHE_MAX_SIZE:
        cache = cache[-config.CACHE_MAX_SIZE:]
    save_json(config.CACHE_FILE, cache)

# ── RAG helpers (Pinecone & Local fallback) ──

def pinecone_embed(text: str) -> list:
    try:
        pc     = Pinecone(api_key=config.PINECONE_API_KEY)
        result = pc.inference.embed(model="multilingual-e5-large", inputs=[text[:500]],
                                    parameters={"input_type": "passage"})
        return result[0].values
    except Exception as e:
        print(f"Embed error: {e}")
        return [0.0] * 1024

def pinecone_search(question: str, grade: str = "", top_k: int = 5) -> list:
    if not PINECONE_SUPPORT or not pinecone_index or not config.PINECONE_API_KEY:
        return []
    try:
        pc       = Pinecone(api_key=config.PINECONE_API_KEY)
        q_vector = pc.inference.embed(model="multilingual-e5-large", inputs=[question[:500]],
                                      parameters={"input_type": "query"})
        grade_num   = re.search(r'\d+', str(grade))
        filter_dict = {"grade": {"$eq": grade_num.group()}} if grade_num else None
        results     = pinecone_index.query(vector=q_vector[0].values, top_k=top_k,
                                           include_metadata=True, filter=filter_dict)
        return [m["metadata"] for m in results["matches"] if m["score"] > 0.5]
    except Exception as e:
        print(f"Pinecone search error: {e}")
        return []

def get_all_pdf_chunks(question: str, grade: str = "") -> list:
    if PINECONE_SUPPORT and pinecone_index and config.PINECONE_API_KEY:
        results = pinecone_search(question, grade)
        if results:
            return [{"filename": r["filename"], "page": r["page"],
                     "text": r["text"], "grade": r.get("grade", ""), "score": 1.0} for r in results]
    if not PDF_SUPPORT:
        return []
    chunks   = []
    keywords = [kw for kw in question.lower().split() if len(kw) > 3]
    if not os.path.exists(config.PDF_FOLDER):
        return []
    for filename in os.listdir(config.PDF_FOLDER):
        if not filename.endswith(".pdf"): continue
        try:
            doc = fitz.open(os.path.join(config.PDF_FOLDER, filename))
            for page_num, page in enumerate(doc):
                text  = page.get_text()
                score = sum(1 for kw in keywords if kw in text.lower())
                if score > 0:
                    chunks.append({"filename": filename, "page": page_num+1,
                                   "text": text[:1500], "score": score})
        except:
            pass
    chunks.sort(key=lambda x: x["score"], reverse=True)
    return chunks[:5]

def agent_decide_rag(question: str, subject: str, chunks: list) -> dict:
    if not chunks:
        return {"use_rag": False, "reason": "No chunks", "selected_chunks": []}
    chunks_summary = "".join([f"\nChunk {i+1} [{c['filename']} - Page {c['page']}]:\n{c['text'][:300]}...\n"
                               for i, c in enumerate(chunks)])
    agent_prompt = f"""You are a RAG decision agent.
Question: "{question}"
Subject: {subject}
Available PDF chunks: {chunks_summary}
Decide:
1. Are these chunks RELEVANT? (yes/no)
2. Which chunk numbers? (e.g. "1,3" or "none")
Reply format:
USE_RAG: yes/no
CHUNKS: 1,2 or none
REASON: one short sentence"""
    try:
        from backend.services.tracker import track_api_call
        response = gemini_check.generate_content(agent_prompt)
        track_api_call("gemini-2.5-flash-lite", "rag_agent")
        text    = response.text.strip()
        use_rag = "USE_RAG: yes" in text
        selected = []
        chunks_line = re.search(r"CHUNKS:\s*(.+)", text)
        if chunks_line and use_rag:
            raw = chunks_line.group(1).strip()
            if raw.lower() != "none":
                for num in raw.split(","):
                    try:
                        idx = int(num.strip()) - 1
                        if 0 <= idx < len(chunks):
                            selected.append(chunks[idx])
                    except:
                        pass
        reason_line = re.search(r"REASON:\s*(.+)", text)
        reason = reason_line.group(1).strip() if reason_line else ""
        return {"use_rag": use_rag, "reason": reason, "selected_chunks": selected}
    except Exception as e:
        print(f"Decide RAG error: {e}")
        return {"use_rag": False, "reason": "Agent error", "selected_chunks": []}

def get_rag_context(question: str, grade: str) -> tuple:
    chunks   = get_all_pdf_chunks(question, grade)
    decision = agent_decide_rag(question, grade, chunks)
    if not decision["use_rag"] or not decision["selected_chunks"]:
        return "", False
    context_parts = [f"[{c['filename']} - Page {c['page']}]\n{c['text']}"
                     for c in decision["selected_chunks"]]
    return "\n\n".join(context_parts), True
