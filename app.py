import io
import spaces
import torch
import librosa
import requests
import tempfile
import numpy as np
import gradio as gr
import soundfile as sf
from transformers import AutoModel

# ------------------------------------------------------------
#  1️⃣  Flash‑Attention / SDPA & TF32 settings (run once)
# ------------------------------------------------------------
if torch.cuda.is_available():
    # Flash‑Attention / SDPA
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        print("✅ Flash Attention / SDPA enabled")
    except Exception as e:
        print(f"⚠️ Flash Attention not available: {e}")

    # TF32 for faster matrix ops on Ampere+ GPUs
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    print("✅ CUDA optimizations enabled (TF32, cuDNN benchmark)")

# ------------------------------------------------------------
#  2️⃣  Helper: download reference audio
# ------------------------------------------------------------
def load_audio_from_url(url: str):
    resp = requests.get(url)
    if resp.status_code == 200:
        audio_data, sr = sf.read(io.BytesIO(resp.content))
        return sr, audio_data
    return None, None

# ------------------------------------------------------------
#  3️⃣  Model loading – FP16 + compile
# ------------------------------------------------------------
repo_id = "ai4bharat/IndicF5"
print("Loading model:", repo_id)

# Load directly in half‑precision (FP16)
model = AutoModel.from_pretrained(
    repo_id,
    trust_remote_code=True,
    low_cpu_mem_usage=False,          # must stay False – prevents meta tensors
    torch_dtype=torch.float16,         # FP16 for speed
    token=os.getenv("HF_TOKEN") if os.getenv("HF_TOKEN") else None,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)
model = model.to(device)

# Optional torch.compile – huge inference speed boost
if hasattr(torch, "compile"):
    try:
        model = torch.compile(model, mode="reduce-overhead")
        print("✅ torch.compile enabled (reduce-overhead)")
    except Exception as e:
        print(f"⚠️ torch.compile failed: {e}")

# ------------------------------------------------------------
#  4️⃣  Inference wrapper – inference mode + autocast
# ------------------------------------------------------------
@gpu_decorator
def synthesize_speech(text, ref_audio, ref_text):
    # Basic validation
    if ref_audio is None or not ref_text.strip():
        return "Error: Please provide a reference audio and its corresponding text."

    # Unpack reference audio (Gradio gives (sr, np.ndarray))
    if isinstance(ref_audio, tuple) and len(ref_audio) == 2:
        sample_rate, audio_data = ref_audio
    else:
        return "Error: Invalid reference audio input."

    # Write temporary wav (no resampling needed)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
        sf.write(tmp_wav.name, audio_data, samplerate=sample_rate, format="WAV")
        tmp_wav.flush()

    # Fast inference
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=True):
        # The IndicF5 model implements __call__ as the inference entry point
        audio = model(text, ref_audio_path=tmp_wav.name, ref_text=ref_text)

    # Normalise int16 → float32 if the vocoder returned PCM16
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    return 24000, audio

# ------------------------------------------------------------
#  5️⃣  Example data (unchanged, just kept for UI)
# ------------------------------------------------------------
EXAMPLES = [
    {
        "audio_name": "PAN_F (Happy)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/PAN_F_HAPPY_00002.wav",
        "ref_text": "ਇੱਕ ਗ੍ਰਾਹਕ ਨੇ ਸਾਡੀ ਬੇਮਿਸਾਲ ਸੇਵਾ ਬਾਰੇ ਦਿਲੋਂਗਵਾਹੀ ਦਿੱਤੀ ਜਿਸ ਨਾਲ ਸਾਨੂੰ ਅਨੰਦ ਮਹਿਸੂਸ ਹੋਇਆ।",
        "synth_text": "मैं बिना किसी चिंता के अपने दोस्तों को अपने ऑटोमोबाइल एक्सपर्ट के पास भेज देता हूँ क्योंकि मैं जानता हूँ कि वह निश्चित रूप से उनकी सभी जरूरतों पर खरा उतरेगा।"
    },
    {
        "audio_name": "TAM_F (Happy)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/TAM_F_HAPPY_00001.wav",
        "ref_text": "நான் நெனச்ச மாதிரியே அமேசான்ல பெரிய தள்ளுபடி வந்திருக்கு. கம்மி காசுக்கே அந்தப் புது சேம்சங் மாடல வாங்கிடலாம்.",
        "synth_text": "ഭക്ഷണത്തിന് ശേഷം തൈര് സാദം കഴിച്ചാൽ ഒരു ഉഷാറാണ്!"
    },
    {
        "audio_name": "MAR_F (WIKI)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/MAR_F_WIKI_00001.wav",
        "ref_text": "दिगंटराव्दारे अंतराळ कक्षेतला कचरा चिन्हित करण्यासाठी प्रयत्न केले जात आहे.",
        "synth_text": "प्रारंभिक अंकुर छेदक. मी सोलापूर जिल्ह्यातील माळशिरस तालुक्यातील शेतकरी गणपत पाटील बोलतोय. माझ्या ऊस पिकावर प्रारंभिक अंकुर छेदक कीड आढळत आहे. क्लोरँट्रानिलीप्रोल (कोराजेन) वापरणे योग्य आहे का? त्याचे प्रमाण किती असावे?"
    },
    {
        "audio_name": "MAR_M (WIKI)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/MAR_M_WIKI_00001.wav",
        "ref_text": "या प्रथाला एकोणीसशे पंचातर ईसवी पासून भारतीय दंड संहिताची धारा चारशे अठ्ठावीस आणि चारशे एकोणतीसच्या अन्तर्गत निषेध केला.",
        "synth_text": "जीवाणू करपा. मी अहमदनगर जिल्ह्यातील राहुरी गावातून बाळासाहेब जाधव बोलतोय. माझ्या डाळिंब बागेत जीवाणू करपा मोठ्या प्रमाणात दिसतोय. स्ट्रेप्टोसायक्लिन आणि कॉपर ऑक्सिड्लोराईड फवारणीसाठी योग्य प्रमाण काय असावे?"
    },
    {
        "audio_name": "KAN_F (Happy)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/KAN_F_HAPPY_00001.wav",
        "ref_text": "ನಮ್‌ ಫ್ರಿಜ್ಜಲ್ಲಿ  ಕೂಲಿಂಗ್‌ ಸಮಸ್ಯೆ ಆಗಿ ನಾನ್‌ ಭಾಳ ದಿನದಿಂದ ಒದ್ದಾಡ್ತಿದ್ದೆ, ಆದ್ರೆ ಅದ್ನೀಗ ಮೆಕಾನಿಕ್ ಆಗಿರೋ ನಿಮ್‌ ಸಹಾಯ್ದಿಂದ ಬಗೆಹರಿಸ್ಕೋಬೋದು ಅಂತಾಗಿ ನಿರಾಳ ಆಯ್ತು ನಂಗೆ.",
        "synth_text": "চেন্নাইয়ের শেয়ারের অটোর যাত্রীদের মধ্যে খাবার ভাগ করে খাওয়াটা আমার কাছে মন খুব ভালো করে দেওয়া একটা বিষয়।"
    },
]

# ------------------------------------------------------------
#  6️⃣  Pre‑load example audio files
# ------------------------------------------------------------
for ex in EXAMPLES:
    sr, data = load_audio_from_url(ex["audio_url"])
    ex["sample_rate"] = sr
    ex["audio_data"] = data

# ------------------------------------------------------------
#  7️⃣  Gradio UI
# ------------------------------------------------------------
with gr.Blocks() as iface:
    gr.Markdown(
        """
        # **IndicF5: High‑Quality Text‑to‑Speech for Indian Languages**

        [![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/ai4bharat/IndicF5)

        Generate speech using a reference prompt audio and its transcript.
        """
    )
    with gr.Row():
        with gr.Column():
            txt = gr.Textbox(label="Text to Synthesize", placeholder="Enter text...", lines=3)
            ref_audio = gr.Audio(label="Reference Prompt Audio", type="numpy")
            ref_txt = gr.Textbox(label="Reference Text", placeholder="Enter transcript...", lines=2)
            btn = gr.Button("🎤 Generate Speech", variant="primary")
        with gr.Column():
            out = gr.Audio(label="Generated Speech", type="numpy")
    # Examples grid
    examples = [
        [ex["synth_text"], (ex["sample_rate"], ex["audio_data"]), ex["ref_text"]]
        for ex in EXAMPLES
    ]
    gr.Examples(
        examples=examples,
        inputs=[txt, ref_audio, ref_txt],
        label="Choose an example:",
    )
    btn.click(synthesize_speech, inputs=[txt, ref_audio, ref_txt], outputs=[out])

iface.launch()
