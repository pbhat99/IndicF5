import gradio as gr
import numpy as np
import os
import torch
import csv
import time
# from huggingface_hub import hf_hub_download # Removed for independence
from f5_tts.api import F5TTS
from f5_tts.infer.utils_infer import transcribe

# Define the model repository ID
REPO_ID = "ai4bharat/IndicF5"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)

print(f"Initializing IndicF5...")

# 1. Check for local model and vocab
ckpt_file = os.path.join(MODEL_DIR, "model.safetensors")
vocab_file = os.path.join(MODEL_DIR, "vocab.txt") # Expecting it directly in model dir or checkpoints/vocab.txt
if not os.path.exists(vocab_file):
     vocab_file = os.path.join(MODEL_DIR, "checkpoints", "vocab.txt")

if not os.path.exists(ckpt_file) or not os.path.exists(vocab_file):
    print(f"Model files not found in {MODEL_DIR}. Attempting download...")
    try:
        from huggingface_hub import hf_hub_download
        ckpt_file = hf_hub_download(repo_id=REPO_ID, filename="model.safetensors", local_dir=MODEL_DIR)
        vocab_file = hf_hub_download(repo_id=REPO_ID, filename="checkpoints/vocab.txt", local_dir=MODEL_DIR)
        print(f"Model downloaded to: {ckpt_file}")
        print(f"Vocab downloaded to: {vocab_file}")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please place 'model.safetensors' and 'vocab.txt' in the 'model' folder manually.")
        ckpt_file = None
        vocab_file = None
else:
    print(f"Found local model at: {ckpt_file}")
    print(f"Found local vocab at: {vocab_file}")

# 2. Initialize F5TTS
f5tts = None
if ckpt_file and vocab_file:
    try:
        # Check for local vocoder
        vocoder_path = os.path.join(MODEL_DIR, "vocos")
        is_local_vocoder = False
        if os.path.exists(os.path.join(vocoder_path, "config.yaml")) and os.path.exists(os.path.join(vocoder_path, "pytorch_model.bin")):
             is_local_vocoder = True
             print(f"Found local vocoder at {vocoder_path}")
        
        # Check for local ASR model (Whisper)
        asr_model_path = None
        # Common local paths for whisper
        possible_asr_paths = [os.path.join(MODEL_DIR, "whisper-large-v3-turbo"), os.path.join(MODEL_DIR, "asr")]
        for p in possible_asr_paths:
            if os.path.exists(p):
                asr_model_path = p
                print(f"Found local ASR model at {p}")
                break

        # We use the default F5-TTS Base config as assumed for IndicF5
        f5tts = F5TTS(
            model_type="F5-TTS",
            ckpt_file=ckpt_file,
            vocab_file=vocab_file,
            device="cuda" if torch.cuda.is_available() else "cpu",
            vocoder_name="vocos",
            local_path=vocoder_path if is_local_vocoder else None,
            hf_cache_dir=os.path.join(MODEL_DIR, "vocos") if not is_local_vocoder else None,
            asr_model_path=asr_model_path
        )
        print("F5TTS Model loaded successfully.")
    except Exception as e:
        print(f"Error initializing F5TTS: {e}")
        import traceback
        traceback.print_exc()

LANGUAGE_CODES = {
    "Auto": None,
    "Assamese": "as",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Hindi": "hi",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Odia": "or",
    "Punjabi": "pa",
    "Tamil": "ta",
    "Telugu": "te"
}

def generate_speech(text, ref_audio, ref_text, language):
    if f5tts is None:
        return None, "Model failed to load. Please check console logs."
    
    if not text:
        return None, "Please enter the text to synthesize."
    if not ref_audio:
        return None, "Please upload a reference audio."
    
    # Auto-transcribe if ref_text is missing
    if not ref_text:
        print(f"No reference text provided. Transcribing (Language: {language})...")
        try:
            lang_code = LANGUAGE_CODES.get(language)
            # We use the helper from f5_tts.infer.utils_infer (or f5tts.transcribe if available)
            ref_text = f5tts.transcribe(ref_audio, language=lang_code)
            print(f"Transcribed Reference Text: {ref_text}")
        except Exception as e:
             return None, f"Error during transcription: {str(e)}"

    print(f"Synthesizing: {text}")
    print(f"Reference Audio: {ref_audio}")
    print(f"Reference Text: {ref_text}")

    try:
        # Generate audio using F5TTS API
        wav, sr, spect = f5tts.infer(
            ref_file=ref_audio,
            ref_text=ref_text,
            gen_text=text,
            remove_silence=False # Optional
        )

        return (sr, wav), f"Synthesis Successful. Ref Text used: {ref_text}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error during synthesis: {str(e)}"

def batch_generate_speech(txt_file, ref_audio, ref_text, language, output_dir="batch_output", progress=gr.Progress()):
    if f5tts is None:
        return "Model failed to load."
    
    if not txt_file:
        return "Please upload a TXT file."
    
    if not ref_audio:
        return "Please provide a global reference audio."

    os.makedirs(output_dir, exist_ok=True)
    results = []
    
    try:
        # Read all lines from the text file
        with open(txt_file.name, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f if line.strip()]
            
        total = len(lines)
        if total == 0:
            return "TXT file is empty."

        results.append(f"Starting batch processing for {total} lines...")
        
        # Auto-transcribe global reference text if needed
        curr_ref_text = ref_text
        if not curr_ref_text:
            print(f"No global reference text provided. Transcribing (Language: {language})...")
            try:
                lang_code = LANGUAGE_CODES.get(language)
                curr_ref_text = f5tts.transcribe(ref_audio, language=lang_code)
                print(f"Transcribed Global Reference Text: {curr_ref_text}")
                results.append(f"Global Ref Text Transcribed: {curr_ref_text}")
            except Exception as e:
                return f"Global transcription failed: {e}"

        for i, text in enumerate(lines):
            fname = f"output_{i+1}_{int(time.time())}.wav"
            out_path = os.path.join(output_dir, fname)
            
            progress((i)/total, desc=f"Processing line {i+1}/{total}")
            
            try:
                wav, sr, spect = f5tts.infer(
                    ref_file=ref_audio,
                    ref_text=curr_ref_text,
                    gen_text=text,
                    remove_silence=False
                )
                
                import soundfile as sf
                sf.write(out_path, wav, sr)
                results.append(f"Line {i+1}: Saved to {out_path}")
                print(f"Line {i+1}: Success")
                
            except Exception as e:
                msg = f"Line {i+1}: Generation failed: {e}"
                print(msg)
                results.append(msg)
                traceback.print_exc()

        progress(1.0, desc="Completed")
            
    except Exception as e:
        traceback.print_exc()
        return f"Error reading TXT: {e}"
        
    return "\n".join(results)

def load_examples():
    examples = []
    example_dir = "examples"
    if not os.path.exists(example_dir):
        return []
    
    # List all wav files
    wav_files = [f for f in os.listdir(example_dir) if f.endswith('.wav')]
    
    for wav_file in wav_files:
        wav_path = os.path.join(example_dir, wav_file)
        txt_file = os.path.splitext(wav_file)[0] + ".txt"
        txt_path = os.path.join(example_dir, txt_file)
        
        ref_text = ""
        if os.path.exists(txt_path):
            try:
                with open(txt_path, "r", encoding="utf-8") as f:
                    ref_text = f.read().strip()
            except Exception as e:
                print(f"Error reading example text {txt_path}: {e}")
        
        # Label can be the filename without extension
        label = os.path.splitext(wav_file)[0]
        
        # Example format: [label, text_to_synthesize_placeholder, ref_audio_path, ref_text, language]
        examples.append([label, "", wav_path, ref_text, "Auto"])
        
    return examples

# Create the Gradio Interface
with gr.Blocks(title="IndicF5 TTS") as app:
    gr.Markdown(
        """
        # IndicF5 Text-to-Speech
        
        **IndicF5** is a near-human polyglot TTS model supporting 11 Indian languages:
        Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu.
        """
    )
    
    with gr.Tabs():
        with gr.Tab("Single Inference"):
            with gr.Row():
                with gr.Column():
                    text_input = gr.Textbox(
                        label="Text to Synthesize", 
                        placeholder="Enter text in an Indic language (e.g., नमस्ते! संगीत की तरह जीवन भी खूबसूरत होता है...)",
                        lines=5
                    )
                    
                    ref_audio_input = gr.Audio(
                        label="Reference Audio Prompt", 
                        type="filepath",
                        interactive=True
                    )
                    
                    ref_text_input = gr.Textbox(
                        label="Reference Audio Text", 
                        placeholder="Enter the exact text spoken in the reference audio. Leave blank to auto-transcribe (requires download).",
                        lines=3
                    )
                    
                    language_input = gr.Dropdown(
                        label="Reference Audio Language (for Auto-Transcribe)",
                        choices=list(LANGUAGE_CODES.keys()),
                        value="Auto",
                        interactive=True
                    )
                    
                    generate_btn = gr.Button("Generate Speech", variant="primary")

                    # Dummy component to hold the example title/label
                    example_label = gr.Textbox(visible=False, label="Example Label")

                    gr.Examples(
                        examples=load_examples(),
                        inputs=[example_label, text_input, ref_audio_input, ref_text_input, language_input],
                        label="Examples (Click to load)"
                    )
                    
                with gr.Column():
                    audio_output = gr.Audio(label="Generated Speech")
                    status_output = gr.Textbox(label="Status", interactive=False)
                    
                    gr.Markdown(
                        """
                        ### Instructions:
                        1. **Text to Synthesize**: Enter the text you want the model to speak.
                        2. **Reference Audio**: Upload a short audio clip (voice prompt) of the speaker you want to clone or use as a style reference.
                        3. **Reference Text**: Transcribe exactly what is spoken in the reference audio. If left blank, the system will attempt to auto-transcribe.
                        4. **Language**: Select the language of the reference audio to improve auto-transcription accuracy.
                        """
                    )

            generate_btn.click(
                fn=generate_speech,
                inputs=[text_input, ref_audio_input, ref_text_input, language_input],
                outputs=[audio_output, status_output]
            )

        with gr.Tab("Batch Processing"):
            gr.Markdown("### Batch Processing from TXT")
            gr.Markdown("Upload a TXT file containing lines of text to synthesize. Global reference audio/text will be used for all lines.")
            
            with gr.Row():
                with gr.Column():
                    txt_input = gr.File(label="Upload TXT File", file_types=[".txt"])
                    
                    batch_ref_audio = gr.Audio(
                        label="Global Reference Audio", 
                        type="filepath", 
                        interactive=True
                    )
                    
                    batch_ref_text = gr.Textbox(
                        label="Global Reference Text", 
                        placeholder="Leave blank for auto-transcribe",
                        lines=2
                    )
                    
                    batch_language = gr.Dropdown(
                        label="Language (for Auto-Transcribe)",
                        choices=list(LANGUAGE_CODES.keys()),
                        value="Auto"
                    )
                    
                    batch_output_dir = gr.Textbox(
                        label="Output Directory", 
                        value="batch_outputs",
                        placeholder="Directory to save generated files"
                    )
                    
                    batch_btn = gr.Button("Start Batch Processing", variant="primary")
                    
                    # Dummy component to hold the example title/label
                    batch_example_label = gr.Textbox(visible=False, label="Example Label")

                    gr.Examples(
                        examples=load_examples(),
                        inputs=[batch_example_label, txt_input, batch_ref_audio, batch_ref_text, batch_language],
                        label="Examples (Click to load Reference Audio)"
                    )
                    
                with gr.Column():
                    batch_status = gr.Textbox(label="Processing Log", lines=20, interactive=False)
            
            batch_btn.click(
                fn=batch_generate_speech,
                inputs=[txt_input, batch_ref_audio, batch_ref_text, batch_language, batch_output_dir],
                outputs=[batch_status]
            )

if __name__ == "__main__":
    app.launch()
