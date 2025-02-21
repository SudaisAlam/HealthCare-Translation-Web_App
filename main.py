from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
import torch
import tempfile
import whisper
import io
import soundfile as sf
from pydantic import BaseModel
from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer
from TTS.api import TTS
import os

app = FastAPI()

# Enable CORS for prototype testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load translation model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_name = "facebook/m2m100_418M"
tokenizer = M2M100Tokenizer.from_pretrained(model_name)
translation_model = M2M100ForConditionalGeneration.from_pretrained(model_name).to(device)

# Load Whisper for transcription
whisper_model = whisper.load_model("base")

# Initialize Coqui TTS for synthesis
tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2", gpu=torch.cuda.is_available())

# Pydantic models
class TranslationRequest(BaseModel):
    text: str
    src_lang: str
    tgt_lang: str

class SynthesisRequest(BaseModel):
    text: str
    language: str  # target language code (e.g., "en", "es", etc.)
    speaker: str = None  # optional speaker name

@app.post("/translate")
async def translate_text(req: TranslationRequest):
    try:
        tokenizer.src_lang = req.src_lang
        encoded = tokenizer(req.text, return_tensors="pt").to(device)
        generated_tokens = translation_model.generate(
            **encoded,
            forced_bos_token_id=tokenizer.get_lang_id(req.tgt_lang)
        )
        translated_text = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        return {"translated_text": translated_text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/transcribe")
async def transcribe_audio(file: UploadFile = File(...), lang: str = "en"):
    tmp_path = None
    try:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            tmp.write(contents)
            tmp_path = tmp.name
        result = whisper_model.transcribe(tmp_path, language=lang)
        return {"transcription": result.get("text", "")}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)

@app.post("/synthesize")
async def synthesize_speech(req: SynthesisRequest):
    try:
        # Use provided speaker or default to the first available
        speaker = req.speaker or (tts.speakers[0] if hasattr(tts, "speakers") and tts.speakers else None)
        if not speaker:
            raise HTTPException(status_code=400, detail="No speaker provided and no default speaker available.")
        
        # Generate speech audio as a NumPy array
        audio_array = tts.tts(req.text, language=req.language, speaker=speaker)
        if audio_array is None:
            raise HTTPException(status_code=500, detail="TTS returned no audio data.")

        # Create an in-memory WAV file
        buffer = io.BytesIO()
        sample_rate = getattr(getattr(tts, "synthesizer", None), "output_sample_rate", 24000)
        sf.write(buffer, audio_array, sample_rate, format="WAV")
        buffer.seek(0)
        return StreamingResponse(buffer, media_type="audio/wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Mount static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def get_index():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
