import os
import uvicorn
import asyncio
import torch
import numpy as np
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from huggingface_hub import hf_hub_download
import sentencepiece
import time
import os
import sys
import tarfile
import secrets
import argparse
from pathlib import Path
from typing import Optional
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware



# Add local vendor directory for websockets/uvicorn dependencies
vendor_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_vendor")
if os.path.exists(vendor_path):
    sys.path.insert(0, vendor_path)



# Imports for WebUI protocol
try:
    import sphn
except ImportError:
    print("WARNING: 'sphn' not found. WebUI audio will not work. Install with 'pip install sphn'")
    sphn = None

# Import moshi internals directly
from moshi.models import loaders, LMGen

app = FastAPI()

# Add CORS to allow cross-origin requests if needed
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_wrapper = None

def get_static_path(hf_repo="nvidia/personaplex-7b-v1"):
    print("Retrieving WebUI static content...")
    try:
        dist_tgz = hf_hub_download(hf_repo, "dist.tgz")
        dist_tgz = Path(dist_tgz)
        dist = dist_tgz.parent / "dist"
        if not dist.exists():
            print(f"Extracting WebUI to {dist}")
            with tarfile.open(dist_tgz, "r:gz") as tar:
                tar.extractall(path=dist_tgz.parent)
        return str(dist)
    except Exception as e:
        print(f"Failed to download/extract WebUI: {e}")
        return None

def get_voices_dir(hf_repo="nvidia/personaplex-7b-v1"):
    print("Retrieving Voice Prompts...")
    try:
        voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
        voices_tgz = Path(voices_tgz)
        voices_dir = voices_tgz.parent / "voices"
        if not voices_dir.exists():
            print(f"Extracting Voices to {voices_dir}")
            with tarfile.open(voices_tgz, "r:gz") as tar:
                tar.extractall(path=voices_tgz.parent)
        return str(voices_dir)
    except Exception as e:
        print(f"Failed to download/extract Voices: {e}")
        return None

class ChatSession:
    """
    Session handler for the WebUI (/api/chat).
    Replicates the logic from moshi.server handling Opus streams and specific control bytes.
    """
    def __init__(self, wrapper, device, sample_rate, frame_rate):
        self.wrapper = wrapper
        self.device = device
        self.sample_rate = int(sample_rate)
        self.frame_rate = frame_rate
        self.frame_size = int(self.sample_rate / self.frame_rate)
        
        self.lm_gen = LMGen(
            self.wrapper.lm,
            audio_silence_frame_cnt=int(0.5 * self.frame_rate),
            sample_rate=self.sample_rate,
            device=self.device,
            frame_rate=self.frame_rate,
        )
        self.lm_gen.streaming_forever(1)
        
        # NOTE: moshi.server uses two mimi instances. We replicate this pattern.
        self.mimi = self.wrapper.mimi
        self.other_mimi = self.wrapper.other_mimi
        
        self.mimi.reset_streaming()
        self.other_mimi.reset_streaming()

    async def run(self, websocket: WebSocket, text_prompt: str = "", voice_prompt_name: str = ""):
        # Acquire lock to ensure exclusive access to the model
        async with self.wrapper.lock:
            print("[Session] Acquired lock. Starting session.")
            
            # Set text prompt
            if text_prompt:
                print(f"[Session] Setting text prompt: {text_prompt}")
                self.lm_gen.text_prompt_tokens = self.wrapper.text_tokenizer.encode(
                    f"<system> {text_prompt} <system>"
                )
            else:
                self.lm_gen.text_prompt_tokens = None

            # Set Voice Prompt
            if voice_prompt_name and self.wrapper.voice_prompt_dir:
                vp_path = os.path.join(self.wrapper.voice_prompt_dir, voice_prompt_name)
                if os.path.exists(vp_path):
                    print(f"[Session] Loading voice prompt: {vp_path}")
                    if vp_path.endswith('.pt'):
                        self.lm_gen.load_voice_prompt_embeddings(vp_path)
                    else:
                        self.lm_gen.load_voice_prompt(vp_path)
                else:
                    print(f"[Session] Warning: Voice prompt not found: {vp_path}")
            
            # Initialize Opus streams
            if sphn is None:
                print("[Session] Error: sphn library missing")
                await websocket.close(reason="sphn library missing")
                return
    
            opus_reader = sphn.OpusStreamReader(self.sample_rate)
            opus_writer = sphn.OpusStreamWriter(self.sample_rate)
            
            # Reset Streaming State under lock
            self.mimi.reset_streaming()
            self.other_mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            
            # Feed System Prompts
            print("[Session] Stepping system prompts...")
            async def is_alive():
                return not (websocket.client_state == WebSocketDisconnect)
            
            await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
            self.mimi.reset_streaming()
            print("[Session] System prompts done. Ready for loop.")
            
            # Flags
            close = False
            
            async def recv_loop():
                nonlocal close
                print("[RecvLoop] Started.")
                try:
                    while not close:
                        # Expecting binary messages
                        data = await websocket.receive_bytes()
                        if not data:
                            continue
                            
                        kind = data[0]
                        if kind == 1: # Audio
                            payload = data[1:]
                            opus_reader.append_bytes(payload)
                        else:
                            print(f"[RecvLoop] Unknown message kind: {kind}")
                except WebSocketDisconnect:
                    print("[RecvLoop] WebSocket Disconnected.")
                    close = True
                except Exception as e:
                    print(f"[RecvLoop] Error: {e}")
                    close = True
                    
            async def opus_loop():
                print("[OpusLoop] Started.")
                all_pcm_data = None
                while not close:
                    await asyncio.sleep(0.001)
                    pcm = opus_reader.read_pcm()
                    
                    if pcm.shape[-1] == 0:
                        continue
                    
                    # print(f"[OpusLoop] Got PCM data: {pcm.shape}")
                        
                    if all_pcm_data is None:
                        all_pcm_data = pcm
                    else:
                        all_pcm_data = np.concatenate((all_pcm_data, pcm))
                    
                    while all_pcm_data.shape[-1] >= self.frame_size:
                        chunk = all_pcm_data[:self.frame_size]
                        all_pcm_data = all_pcm_data[self.frame_size:]
                        
                        chunk_tensor = torch.from_numpy(chunk).to(self.device).view(1, 1, -1)
                        
                        codes = self.mimi.encode(chunk_tensor)
                        _ = self.other_mimi.encode(chunk_tensor)
                        
                        for c in range(codes.shape[-1]):
                            tokens = self.lm_gen.step(codes[:, :, c : c + 1])
                            if tokens is None:
                                continue
                            
                            # Log every ~100 tokens to avoid spam but confirm activity
                            # if secrets.randbelow(100) == 0:
                            #    print(f"[OpusLoop] Generated tokens: {tokens[0,0,0]} (Audio: {tokens[0,1,0]})")

                            main_pcm = self.mimi.decode(tokens[:, 1:9])
                            _ = self.other_mimi.decode(tokens[:, 1:9])
                            
                            main_pcm = main_pcm.cpu()
                            opus_writer.append_pcm(main_pcm[0, 0].detach().numpy())
                            
                            text_token = tokens[0, 0, 0].item()
                            if text_token not in (0, 3):
                                 _text = self.wrapper.text_tokenizer.id_to_piece(text_token)
                                 _text = _text.replace("▁", " ")
                                 msg = b"\x02" + bytes(_text, encoding="utf8")
                                 await websocket.send_bytes(msg)
    
            async def send_loop():
                print("[SendLoop] Started.")
                while not close:
                    await asyncio.sleep(0.001)
                    msg = opus_writer.read_bytes()
                    if len(msg) > 0:
                        try:
                            # print(f"[SendLoop] Sending {len(msg)} bytes")
                            await websocket.send_bytes(b"\x01" + msg)
                        except Exception as e:
                            print(f"[SendLoop] Error: {e}")
                            break
    
            # Send Handshake
            print("[Session] Sending handshake.")
            await websocket.send_bytes(b"\x00")
            
            # Run loops
            tasks = [
                asyncio.create_task(recv_loop()),
                asyncio.create_task(opus_loop()),
                asyncio.create_task(send_loop())
            ]
            
            try:
                done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                for t in pending:
                    t.cancel()
            except:
                pass


class PhoneSession:
    """
    Session handler for RAW PCM Telephony (/api/phone).
    """
    def __init__(self, wrapper, device):
        self.wrapper = wrapper
        self.device = device
        self.sample_rate = wrapper.mimi.sample_rate
        self.frame_rate = wrapper.mimi.frame_rate
        self.frame_size = int(self.sample_rate / self.frame_rate)
        
        self.lm_gen = LMGen(
            self.wrapper.lm,
            audio_silence_frame_cnt=int(0.5 * self.frame_rate),
            sample_rate=self.sample_rate,
            device=self.device,
            frame_rate=self.frame_rate,
        )
        self.lm_gen.streaming_forever(1)
        self.mimi = wrapper.mimi

    async def process_audio_stream(self, websocket: WebSocket):
        print("[PhoneSession] Starting.")
        async with self.wrapper.lock:
            print("[PhoneSession] Acquired lock.")
            # Reset Streaming
            self.wrapper.mimi.reset_streaming()
            self.wrapper.other_mimi.reset_streaming()
            self.lm_gen.reset_streaming()
            
            # Warmup/System Prompts (if needed for phone too, usually yes)
            async def is_alive():
                return not (websocket.client_state == WebSocketDisconnect)
            
            print("[PhoneSession] Stepping system prompts...")
            await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
            self.wrapper.mimi.reset_streaming() # Reset mimi context after prompt encoding
            print("[PhoneSession] Ready for loop.")

            incoming_buffer = np.zeros(0, dtype=np.float32)
            try:
                while True:
                    data = await websocket.receive_bytes()
                    try:
                        new_audio = np.frombuffer(data, dtype=np.float32)
                    except ValueError:
                         # Attempt int16 conversion if float32 fails or looks wrong
                         new_audio = np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0

                    incoming_buffer = np.concatenate((incoming_buffer, new_audio))
                    
                    while len(incoming_buffer) >= self.frame_size:
                        chunk = incoming_buffer[:self.frame_size]
                        incoming_buffer = incoming_buffer[self.frame_size:]
                        
                        chunk_tensor = torch.from_numpy(chunk).to(self.device).view(1, 1, -1)
                        codes = self.mimi.encode(chunk_tensor)
                        
                        for c in range(codes.shape[-1]):
                            tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                            if tokens is None:
                                continue
                            
                            audio_tokens = tokens[:, 1:9]
                            audio_output = self.mimi.decode(audio_tokens)
                            audio_output_np = audio_output.cpu().detach().numpy()[0, 0]
                            await websocket.send_bytes(audio_output_np.tobytes())
                    await asyncio.sleep(0.001)
            except WebSocketDisconnect:
                print("[PhoneSession] Disconnected.")
            except Exception as e:
                print(f"[PhoneSession] Error: {e}")


class ModelWrapper:
    def __init__(self, hf_repo="nvidia/personaplex-7b-v1", device="cuda"):
        self.device = torch.device(device)
        self.hf_repo = hf_repo
        self.lock = asyncio.Lock()
        
        print(f"Initializing on {self.device}...")
        
        print("Downloading/Loading Mimi (1)...")
        mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, self.device)
        self.mimi.streaming_forever(1)
        
        print("Downloading/Loading Mimi (2)...")
        self.other_mimi = loaders.get_mimi(mimi_weight, self.device)
        self.other_mimi.streaming_forever(1)
        
        print("Downloading/Loading Moshi...")
        moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)
        self.lm = loaders.get_moshi_lm(moshi_weight, device=self.device)
        self.lm.eval()
        
        print("Loading Tokenizer...")
        tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)

        # Voice Prompts
        self.voice_prompt_dir = get_voices_dir(hf_repo)

@app.on_event("startup")
async def startup_event():
    global model_wrapper
    print("Loading PersonaPlex/Moshi weights...")
    model_wrapper = ModelWrapper(
        hf_repo="nvidia/personaplex-7b-v1",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
    
    # Mount WebUI static files
    static_path = get_static_path()
    if static_path and os.path.exists(static_path):
        app.mount("/", StaticFiles(directory=static_path, html=True), name="static")
        print(f"WebUI serving from {static_path}")
    else:
        print("WARNING: WebUI not found.")

    print("Server Ready. Listening on /api/chat (WebUI) and /api/phone (Telephony)")

@app.websocket("/api/chat")
async def chat_endpoint(websocket: WebSocket):
    await websocket.accept()
    # Read query params
    text_prompt = websocket.query_params.get("text_prompt", "")
    voice_prompt = websocket.query_params.get("voice_prompt", "")
    
    session = ChatSession(
        model_wrapper, 
        model_wrapper.device, 
        model_wrapper.mimi.sample_rate, 
        model_wrapper.mimi.frame_rate
    )
    await session.run(websocket, text_prompt, voice_prompt)

@app.websocket("/api/phone")
async def phone_endpoint(websocket: WebSocket):
    await websocket.accept()
    session = PhoneSession(model_wrapper, model_wrapper.device)
    await session.process_audio_stream(websocket)


if __name__ == "__main__":
    # Optional: SSL Config (can be removed if handled by a reverse proxy)
    SSL_CERT_PATH = os.path.expanduser("~/ssl_certs/cert.pem")
    SSL_KEY_PATH = os.path.expanduser("~/ssl_certs/key.pem")
    
    ssl_config = {}
    if os.path.exists(SSL_CERT_PATH) and os.path.exists(SSL_KEY_PATH):
        ssl_config["ssl_keyfile"] = SSL_KEY_PATH
        ssl_config["ssl_certfile"] = SSL_CERT_PATH
        print("SSL Enabled.")
    else:
        print("SSL Certificates not found. Running in HTTP mode.")

    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8998,
        **ssl_config
    )
