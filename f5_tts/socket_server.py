import socket
import struct
import torch
import torchaudio
import soundfile as sf
from threading import Thread


import gc
import traceback


from infer.utils_infer import infer_batch_process, preprocess_ref_audio_text, load_vocoder, load_model
from model.backbones.dit import DiT


class TTSStreamingProcessor:
    def __init__(self, ckpt_file, vocab_file, ref_audio, ref_text, device=None, dtype=torch.float32, hf_cache_dir=None, asr_model_path=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.asr_model_path = asr_model_path
        self.hf_cache_dir = hf_cache_dir

        # Load the model using the provided checkpoint and vocab files
        self.model = load_model(
            model_cls=DiT,
            model_cfg=dict(dim=1024, depth=22, heads=16, ff_mult=2, text_dim=512, conv_layers=4),
            ckpt_path=ckpt_file,
            mel_spec_type="vocos",  # or "bigvgan" depending on vocoder
            vocab_file=vocab_file,
            ode_method="euler",
            use_ema=True,
            device=self.device,
        ).to(self.device, dtype=dtype)

        # Load the vocoder
        is_local_vocoder = False
        if hf_cache_dir and os.path.exists(os.path.join(hf_cache_dir, "config.yaml")):
            is_local_vocoder = True
        
        self.vocoder = load_vocoder(is_local=is_local_vocoder, local_path=hf_cache_dir if is_local_vocoder else None, device=self.device, hf_cache_dir=hf_cache_dir)

        # Set sampling rate for streaming
        self.sampling_rate = 24000  # Consistency with client

        # Set reference audio and text
        self.ref_audio = ref_audio
        self.ref_text = ref_text

        # Warm up the model
        self._warm_up()

    def _warm_up(self):
        """Warm up the model with a dummy input to ensure it's ready for real-time processing."""
        print("Warming up the model...")
        ref_audio, ref_text = preprocess_ref_audio_text(self.ref_audio, self.ref_text, hf_cache_dir=self.hf_cache_dir, model_path=self.asr_model_path)
        
        audio_np, sr = sf.read(ref_audio)
        if len(audio_np.shape) == 1:
            audio_np = audio_np[None, :]
        else:
            audio_np = audio_np.T
        audio = torch.from_numpy(audio_np).float()
        
        gen_text = "Warm-up text for the model."

        # Pass the vocoder as an argument here
        infer_batch_process((audio, sr), ref_text, [gen_text], self.model, self.vocoder, device=self.device)
        print("Warm-up completed.")

    def generate_stream(self, text, play_steps_in_s=0.5):
        """Generate audio in chunks and yield them in real-time."""
        # Preprocess the reference audio and text
        ref_audio, ref_text = preprocess_ref_audio_text(self.ref_audio, self.ref_text, hf_cache_dir=self.hf_cache_dir, model_path=self.asr_model_path)

        # Load reference audio
        audio_np, sr = sf.read(ref_audio)
        if len(audio_np.shape) == 1:
            audio_np = audio_np[None, :]
        else:
            audio_np = audio_np.T
        audio = torch.from_numpy(audio_np).float()

        # Run inference for the input text
        audio_chunk, final_sample_rate, _ = infer_batch_process(
            (audio, sr),
            ref_text,
            [text],
            self.model,
            self.vocoder,
            device=self.device,  # Pass vocoder here
        )

        # Break the generated audio into chunks and send them
        chunk_size = int(final_sample_rate * play_steps_in_s)

        if len(audio_chunk) < chunk_size:
            packed_audio = struct.pack(f"{len(audio_chunk)}f", *audio_chunk)
            yield packed_audio
            return

        for i in range(0, len(audio_chunk), chunk_size):
            chunk = audio_chunk[i : i + chunk_size]

            # Check if it's the final chunk
            if i + chunk_size >= len(audio_chunk):
                chunk = audio_chunk[i:]

            # Send the chunk if it is not empty
            if len(chunk) > 0:
                packed_audio = struct.pack(f"{len(chunk)}f", *chunk)
                yield packed_audio


def handle_client(client_socket, processor):
    try:
        while True:
            # Receive data from the client
            data = client_socket.recv(1024).decode("utf-8")
            if not data:
                break

            try:
                # The client sends the text input
                text = data.strip()

                # Generate and stream audio chunks
                for audio_chunk in processor.generate_stream(text):
                    client_socket.sendall(audio_chunk)

                # Send end-of-audio signal
                client_socket.sendall(b"END_OF_AUDIO")

            except Exception as inner_e:
                print(f"Error during processing: {inner_e}")
                traceback.print_exc()  # Print the full traceback to diagnose the issue
                break

    except Exception as e:
        print(f"Error handling client: {e}")
        traceback.print_exc()
    finally:
        client_socket.close()


def start_server(host, port, processor):
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind((host, port))
    server.listen(5)
    print(f"Server listening on {host}:{port}")

    while True:
        client_socket, addr = server.accept()
        print(f"Accepted connection from {addr}")
        client_handler = Thread(target=handle_client, args=(client_socket, processor))
        client_handler.start()


if __name__ == "__main__":
    try:
        import os
        
        REPO_ID = "ai4bharat/IndicF5"
        MODEL_DIR = "model"
        os.makedirs(MODEL_DIR, exist_ok=True)
        
        ckpt_file = os.path.join(MODEL_DIR, "model.safetensors")
        vocab_file = os.path.join(MODEL_DIR, "vocab.txt")
        if not os.path.exists(vocab_file):
             vocab_file = os.path.join(MODEL_DIR, "checkpoints", "vocab.txt")

        if not os.path.exists(ckpt_file) or not os.path.exists(vocab_file):
            print(f"Downloading/Locating model files in {MODEL_DIR}...")
            try:
                from huggingface_hub import hf_hub_download
                ckpt_file = hf_hub_download(repo_id=REPO_ID, filename="model.safetensors", local_dir=MODEL_DIR)
                vocab_file = hf_hub_download(repo_id=REPO_ID, filename="checkpoints/vocab.txt", local_dir=MODEL_DIR)
            except ImportError:
                 print("huggingface_hub not installed. Please ensure models are present in 'model' directory.")
        
        # Check for local ASR model (Whisper)
        asr_model_path = None
        # Common local paths for whisper
        possible_asr_paths = [os.path.join(MODEL_DIR, "whisper-large-v3-turbo"), os.path.join(MODEL_DIR, "asr")]
        for p in possible_asr_paths:
            if os.path.exists(p):
                asr_model_path = p
                print(f"Found local ASR model at {p}")
                break

        # Initialize the processor with the model and vocoder
        processor = TTSStreamingProcessor(
            ckpt_file=ckpt_file,
            vocab_file=vocab_file,
            ref_audio="",  # add ref audio"./tests/ref_audio/reference.wav"
            ref_text="",
            device=None,
            dtype=torch.float32,
            hf_cache_dir=os.path.join(MODEL_DIR, "vocos"),
            asr_model_path=asr_model_path
        )
        # Note: TTSStreamingProcessor doesn't currently take hf_cache_dir, 
        # it would need modification if we want vocoder also in 'model/'
        
        # Start the server
        start_server("0.0.0.0", 9998, processor)
    except KeyboardInterrupt:
        gc.collect()
