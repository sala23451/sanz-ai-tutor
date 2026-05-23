import os
import io
import re
from backend import config
from backend.services.tracker import detect_language
from backend.tutor_prompts import LANG_INSTRUCTIONS

try:
    import speech_recognition as sr
    STT_SUPPORT = True
except ImportError:
    STT_SUPPORT = False

try:
    from gtts import gTTS
    TTS_SUPPORT = True
except ImportError:
    TTS_SUPPORT = False

def process_voice_input(audio_contents: bytes) -> dict:
    if not STT_SUPPORT:
        return {"status": "error", "message": "STT not installed on this server. Install SpeechRecognition."}
    
    os.makedirs("temp_audio", exist_ok=True)
    temp_path = os.path.join("temp_audio", "input.wav")
    
    with open(temp_path, "wb") as f:
        f.write(audio_contents)
        
    try:
        recognizer = sr.Recognizer()
        with sr.AudioFile(temp_path) as source:
            audio_data = recognizer.record(source)
            
        detected_text, detected_lang = None, "en"
        for lang_code, lang_key in [("si-LK", "si"), ("ta-LK", "ta"), ("en-US", "en")]:
            try:
                text = recognizer.recognize_google(audio_data, language=lang_code)
                if text:
                    detected_text = text
                    detected_lang = lang_key
                    break
            except:
                pass
                
        try:
            os.remove(temp_path)
        except:
            pass
            
        if not detected_text:
            return {"status": "error", "message": "Could not understand audio. Try speaking clearly or use a different language."}
            
        text_lang = detect_language(detected_text)
        if text_lang != "en":
            detected_lang = text_lang
            
        return {"status": "success", "text": detected_text, "detected_language": detected_lang}
    except Exception as e:
        try:
            os.remove(temp_path)
        except:
            pass
        return {"status": "error", "message": str(e)}

def convert_text_to_speech(text: str, language: str = None) -> io.BytesIO:
    if not TTS_SUPPORT:
        raise RuntimeError("TTS not installed on this server. Install gTTS.")
        
    if not language or language not in LANG_INSTRUCTIONS:
        language = detect_language(text)
        
    gtts_lang  = LANG_INSTRUCTIONS[language]['gtts_lang']
    clean_text = re.sub(r'[*_#`\[\]()~>]', '', text)
    clean_text = re.sub(r'\s+', ' ', clean_text).strip()[:1000]
    
    tts = gTTS(text=clean_text, lang=gtts_lang, slow=False)
    audio_buffer = io.BytesIO()
    tts.write_to_fp(audio_buffer)
    audio_buffer.seek(0)
    return audio_buffer
