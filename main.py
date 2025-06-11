from flask import Flask, render_template, request, jsonify
from transformers import pipeline
from keybert import KeyBERT
from moviepy.editor import VideoFileClip
import speech_recognition as sr
import os, nltk

nltk.download('punkt')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

sentiment_analyzer = pipeline('sentiment-analysis')
kw_model = KeyBERT()

def transcribe_audio(audio_path):
    r = sr.Recognizer()
    with sr.AudioFile(audio_path) as src:
        audio = r.record(src)
    try:
        return r.recognize_google(audio)
    except:
        return ""

def extract_audio(video_path):
    audio_path = video_path.replace(".mp4", ".wav")
    clip = VideoFileClip(video_path)
    clip.audio.write_audiofile(audio_path, logger=None)
    return audio_path

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze_text', methods=['POST'])
def analyze_text():
    pitch = request.json.get('pitch', '')
    if not pitch:
        return jsonify({"error": "No pitch provided"}), 400
    sent = sentiment_analyzer(pitch)[0]
    kws = kw_model.extract_keywords(pitch, top_n=5)
    return jsonify({
        "sentiment": sent['label'],
        "confidence": round(sent['score'], 2),
        "keywords": [k[0] for k in kws],
        "feedback": "Great pitch!" if sent['label'] == "POSITIVE" else "Try more upbeat tone."
    })

@app.route('/analyze_media', methods=['POST'])
def analyze_media():
    f = request.files.get('file')
    if not f:
        return jsonify({"error": "No file uploaded"}), 400
    path = f"uploads/{f.filename}"
    os.makedirs('uploads', exist_ok=True)
    f.save(path)
    audio = extract_audio(path) if f.filename.lower().endswith('.mp4') else path
    text = transcribe_audio(audio)
    if not text:
        return jsonify({"error": "Transcription failed"}), 400
    sent = sentiment_analyzer(text)[0]
    kws = kw_model.extract_keywords(text, top_n=5)
    os.remove(path)
    if audio != path:
        os.remove(audio)
    return jsonify({
        "transcript": text,
        "sentiment": sent['label'],
        "confidence": round(sent['score'], 2),
        "keywords": [k[0] for k in kws],
        "feedback": "Nice delivery!" if sent['label'] == "POSITIVE" else "Consider more energy."
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
