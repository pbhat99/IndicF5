import gradio as gr
from transformers import AutoModel
import numpy as np
import soundfile as sf
import os
import sys

# Define the model repository ID
REPO_ID = "ai4bharat/IndicF5"

print(f"Loading {REPO_ID} model... This may take a while significantly for the first time.")
try:
    # Load the model with trust_remote_code=True as required
    model = AutoModel.from_pretrained(REPO_ID, trust_remote_code=True)
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    model = None

def generate_speech(text, ref_audio, ref_text):
    if model is None:
        return None, "Model failed to load. Please check console logs."
    
    if not text:
        return None, "Please enter the text to synthesize."
    if not ref_audio:
        return None, "Please upload a reference audio."
    if not ref_text:
        return None, "Please enter the reference text."

    print(f"Synthesizing: {text}")
    print(f"Reference Audio: {ref_audio}")
    print(f"Reference Text: {ref_text}")

    try:
        # Generate audio
        # The model expects ref_audio_path as a string path to the wav file
        audio = model(
            text,
            ref_audio_path=ref_audio,
            ref_text=ref_text
        )

        # The output audio might be raw samples. 
        # The README says:
        # if audio.dtype == np.int16: audio = audio.astype(np.float32) / 32768.0
        # samplerate=24000
        
        target_sr = 24000
        
        if hasattr(audio, 'dtype') and audio.dtype == np.int16:
             audio = audio.astype(np.float32) / 32768.0
        elif isinstance(audio, np.ndarray) and audio.dtype == np.int16: # Double check
             audio = audio.astype(np.float32) / 32768.0
             
        # Convert to numpy if it's a tensor (just in case)
        if hasattr(audio, 'cpu'):
            audio = audio.cpu().numpy()
            
        return (target_sr, audio), "Synthesis Successful"

    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, f"Error during synthesis: {str(e)}"

# Create the Gradio Interface
with gr.Blocks(title="IndicF5 TTS") as app:
    gr.Markdown(
        """
        # IndicF5 Text-to-Speech
        
        **IndicF5** is a near-human polyglot TTS model supporting 11 Indian languages:
        Assamese, Bengali, Gujarati, Hindi, Kannada, Malayalam, Marathi, Odia, Punjabi, Tamil, Telugu.
        """
    )
    
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
                placeholder="Enter the exact text spoken in the reference audio...",
                lines=3
            )
            
            generate_btn = gr.Button("Generate Speech", variant="primary")
            
        with gr.Column():
            audio_output = gr.Audio(label="Generated Speech")
            status_output = gr.Textbox(label="Status", interactive=False)
            
            gr.Markdown(
                """
                ### Instructions:
                1. **Text to Synthesize**: Enter the text you want the model to speak.
                2. **Reference Audio**: Upload a short audio clip (voice prompt) of the speaker you want to clone or use as a style reference.
                3. **Reference Text**: Transcribe exactly what is spoken in the reference audio.
                """
            )

    generate_btn.click(
        fn=generate_speech,
        inputs=[text_input, ref_audio_input, ref_text_input],
        outputs=[audio_output, status_output]
    )

if __name__ == "__main__":
    app.launch()