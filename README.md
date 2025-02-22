# Healthcare Translation App

This application provides real-time transcription, translation, and text-to-speech synthesis for healthcare communication.

## Features
- **Transcription:** Converts spoken words into text using Whisper AI.
- **Translation:** Translates transcribed text into multiple languages using M2M100.
- **Text-to-Speech (TTS):** Converts translated text into speech using Coqui TTS.

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/SudaisAlam/HealthCare-Translation-Web_App.git
cd HealthCare-Translation-Web_App
```

### 2. Create a Virtual Environment (Recommended)
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the Backend Server
```bash
uvicorn main:app --host localhost --port 8000 --reload
```


## API Endpoints

| Endpoint       | Method | Description |
|---------------|--------|-------------|
| `/translate`  | POST   | Translate text from one language to another |
| `/transcribe` | POST   | Transcribe an audio file to text |
| `/synthesize` | POST   | Convert text to speech |

## Dependencies
- FastAPI
- Whisper
- Transformers (M2M100)
- Coqui TTS
- Torch
- Soundfile

Enjoy using the Healthcare Translation App! 🚀
