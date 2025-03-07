import os
from dotenv import load_dotenv, find_dotenv
from elevenlabs import stream
from elevenlabs.client import ElevenLabs
from groq import Groq
import pygame
import tempfile
from deepgram import DeepgramClient, LiveTranscriptionEvents, LiveOptions
import threading
import pyaudio
import wave

# Load environment variables from .env file
load_dotenv(find_dotenv())

class AI_Assistant:
    def __init__(self):
        self.deepgram = DeepgramClient(api_key=os.getenv("DEEPGRAM_API_KEY"))
        self.client = Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.elevenlabs_api_key = os.getenv("ELEVENLABS_API_KEY")
        self.elevenlabs_client = ElevenLabs(api_key=self.elevenlabs_api_key)
        
        self.recording = False
        self.audio_thread = None
        self.dg_connection = None
        
        self.interaction = [
            {"role": "system", "content": "You are a helpful travel guide in London, UK, helping a tourist plan their trip. Be conversational and concise in your responses."},
        ]

    def on_transcript(self, result):
        """Handle transcription results from Deepgram."""
        if result.channel.alternatives:
            transcript = result.channel.alternatives[0].transcript
            if transcript:
                self.generate_ai_response(transcript)

    def start_recording(self):
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 16000

        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT,
                       channels=CHANNELS,
                       rate=RATE,
                       input=True,
                       frames_per_buffer=CHUNK)

        while self.recording:
            data = stream.read(CHUNK)
            if self.dg_connection:
                self.dg_connection.send(data)

        stream.stop_stream()
        stream.close()
        p.terminate()

    def start_transcription(self):
        self.recording = True
        self.dg_connection = self.deepgram.listen.websocket.v("1")
        self.dg_connection.on(LiveTranscriptionEvents.Transcript, self.on_transcript)

        options = LiveOptions(
            model="nova-3",
            language="en-US",
            smart_format=True,
            interim_results=False
        )

        if self.dg_connection.start(options):
            self.audio_thread = threading.Thread(target=self.start_recording)
            self.audio_thread.start()
        else:
            print("Failed to start Deepgram connection")

    def stop_transcription(self):
        if self.recording:
            self.recording = False
            if self.audio_thread:
                self.audio_thread.join()
            if self.dg_connection:
                self.dg_connection.finish()

    def generate_ai_response(self, text):
        self.interaction.append({"role": "user", "content": text})
        print(f"\nTourist: {text}", end="\r\n")

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=self.interaction,
                max_tokens=150,
                n=1,
                stop=None,
                temperature=0.7,
            )
            ai_message = response.choices[0].message["content"]
            self.interaction.append({"role": "assistant", "content": ai_message})
            print(f"\nAI Guide: {ai_message}", end="\r\n")
            self.generate_audio(ai_message)
        except Exception as e:
            print("Error generating AI response:", e)

    def generate_audio(self, text):
        self.interaction.append({"role": "assistant", "content": text})
        print(f"\nAI Guide: {text}")
        temp_path = None

        try:
            # Generate audio and convert to bytes
            audio_generator = self.elevenlabs_client.generate(
                text=text,
                voice="Rachel",
                stream=False
            )
            audio_bytes = b"".join(audio_generator)
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as temp_file:
                temp_file.write(audio_bytes)
                temp_path = temp_file.name
            
            # Initialize pygame mixer if not already initialized
            if not pygame.mixer.get_init():
                pygame.mixer.init()
            
            # Play the audio file using pygame
            pygame.mixer.music.load(temp_path)
            pygame.mixer.music.play()
            
            # Wait for playback to finish
            while pygame.mixer.music.get_busy():
                pygame.time.Clock().tick(10)
            
            # Unload the music and quit pygame mixer
            pygame.mixer.music.unload()
            pygame.mixer.quit()
            
            # Clean up the temporary file
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except Exception as e:
                    print(f"Warning: Could not delete temporary file {temp_path}: {e}")
            
        except Exception as e:
            print("Error generating audio:", e)
            # Cleanup in case of error
            if temp_path and os.path.exists(temp_path):
                try:
                    os.unlink(temp_path)
                except:
                    pass

greeting = "Thank you for calling London Travel Guide. My name is Rachel, how may I assist you?"
ai_assistant = AI_Assistant()
ai_assistant.generate_audio(greeting)
ai_assistant.start_transcription()