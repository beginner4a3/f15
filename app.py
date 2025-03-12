import io
import librosa
import requests
import tempfile
import numpy as np
import gradio as gr
import soundfile as sf
from transformers import AutoModel

# Function to load reference audio from URL
def load_audio_from_url(url):
    response = requests.get(url)
    if response.status_code == 200:
        audio_data, sample_rate = sf.read(io.BytesIO(response.content))
        return sample_rate, audio_data
    return None, None

def synthesize_speech(text, ref_audio, ref_text):
    if ref_audio is None or ref_text.strip() == "":
        return "Error: Please provide a reference audio and its corresponding text."
    
    # Ensure valid reference audio input
    if isinstance(ref_audio, tuple) and len(ref_audio) == 2:
        sample_rate, audio_data = ref_audio
    else:
        return "Error: Invalid reference audio input."
    
    # Save reference audio directly without resampling
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
        sf.write(temp_audio.name, audio_data, samplerate=sample_rate, format='WAV')
        temp_audio.flush()
    
    audio = model(text, ref_audio_path=temp_audio.name, ref_text=ref_text)
             
    # Normalize output and save
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0

    return 24000, audio


# Load TTS model
repo_id = "ai4bharat/IndicF5"
model = AutoModel.from_pretrained(repo_id, trust_remote_code=True)

# Example Data (Multiple Examples)
EXAMPLES = [
    {
        "audio_name": "PAN_F (Happy)",
        "audio_url": "https://github.com/AI4Bharat/IndicF5/raw/refs/heads/main/prompts/PAN_F_HAPPY_00001.wav",
        "ref_text": "ਭਹੰਪੀ ਵਿੱਚ ਸਮਾਰਕਾਂ ਦੇ ਭਵਨ ਨਿਰਮਾਣ ਕਲਾ ਦੇ ਵੇਰਵੇ ਗੁੰਝਲਦਾਰ ਅਤੇ ਹੈਰਾਨ ਕਰਨ ਵਾਲੇ ਹਨ, ਜੋ ਮੈਨੂੰ ਖੁਸ਼ ਕਰਦੇ  ਹਨ।",
        "synth_text": "मैं बिना किसी चिंता के अपने दोस्तों को अपने ऑटोमोबाइल एक्सपर्ट के पास भेज देता हूँ क्योंकि मैं जानता हूँ कि वह निश्चित रूप से उनकी सभी जरूरतों पर खरा उतरेगा।"
    },
]


# Preload all example audios
for example in EXAMPLES:
    sample_rate, audio_data = load_audio_from_url(example["audio_url"])
    example["sample_rate"] = sample_rate
    example["audio_data"] = audio_data


# Define Gradio interface with layout adjustments
with gr.Blocks(css="body { font-family: Arial, sans-serif; }") as iface:
    gr.Markdown(
        """
        # **IndicF5: High-Quality Text-to-Speech for Indian Languages**

        [![Hugging Face](https://img.shields.io/badge/HuggingFace-Model-orange)](https://huggingface.co/ai4bharat/IndicF5)

        We release **IndicF5**, a **near-human polyglot** **Text-to-Speech (TTS)** model trained on **1417 hours** of high-quality speech from **[Rasa](https://huggingface.co/datasets/ai4bharat/Rasa), [IndicTTS](https://www.iitm.ac.in/donlab/indictts/database), [LIMMITS](https://sites.google.com/view/limmits24/), and [IndicVoices-R](https://huggingface.co/datasets/ai4bharat/indicvoices_r)**.  

        IndicF5 supports **11 Indian languages**:  
        **Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu.**  
        
        Generate speech using a reference prompt audio and its corresponding text.
        """
    )
    
    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(label="Text to Synthesize", placeholder="Enter the text to convert to speech...", lines=3)
            ref_audio_input = gr.Audio(type="numpy", label="Reference Prompt Audio")
            ref_text_input = gr.Textbox(label="Text in Reference Prompt Audio", placeholder="Enter the transcript of the reference audio...", lines=2)
            submit_btn = gr.Button("🎤 Generate Speech", variant="primary")
        
        with gr.Column():
            output_audio = gr.Audio(label="Generated Speech", type="numpy")
    
    # Add multiple examples
    examples = [
        [ex["synth_text"], (ex["sample_rate"], ex["audio_data"]), ex["ref_text"]] for ex in EXAMPLES
    ]
    
    gr.Examples(
        examples=examples,
        inputs=[text_input, ref_audio_input, ref_text_input],
        label="Choose an example:"
    )

    submit_btn.click(synthesize_speech, inputs=[text_input, ref_audio_input, ref_text_input], outputs=[output_audio])

# Launch the app
if __name__ == "__main__":
    iface.launch(share=True)