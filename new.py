import os
import torch
import whisper
from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import librosa
import numpy as np
import matplotlib.pyplot as plt
import json
import time
import gradio as gr
from pathlib import Path
import tempfile
import traceback

# Dictionary mapping ISO language codes to full names
LANGUAGE_MAP = {
    "hi": "Hindi",
    "mr": "Marathi",
    "bn": "Bengali",
    "ta": "Tamil",
    "te": "Telugu",
    "ml": "Malayalam",
    "kn": "Kannada",
    "gu": "Gujarati",
    "pa": "Punjabi",
    "or": "Odia",
    "as": "Assamese",
    "sa": "Sanskrit",
    # Tribal languages
    "sat": "Santali",
    "kok": "Konkani",
    "gon": "Gondi",
    "bho": "Bhojpuri",
    "mai": "Maithili",
    "doi": "Dogri",
    "awa": "Awadhi",
    "mag": "Magahi",
    "mni": "Manipuri",
    "auto": "Auto-detect"
}

# Mapping language codes to their script systems
SCRIPT_MAP = {
    "hi": sanscript.DEVANAGARI,
    "mr": sanscript.DEVANAGARI,
    "bn": sanscript.BENGALI,
    "ta": sanscript.TAMIL,
    "te": sanscript.TELUGU,
    "ml": sanscript.MALAYALAM,
    "kn": sanscript.KANNADA,
    "gu": sanscript.GUJARATI,
    "pa": sanscript.GURMUKHI,
    "or": sanscript.ORIYA,
    "as": sanscript.BENGALI,  # Assamese uses Bengali script with variations
    "sa": sanscript.DEVANAGARI,
    # For tribal languages that may use adapted scripts
    "sat": sanscript.DEVANAGARI,  # Santali has its own script
    "kok": sanscript.DEVANAGARI,
    "gon": sanscript.DEVANAGARI,  # Some use Devanagari
    "bho": sanscript.DEVANAGARI,
    "mai": sanscript.DEVANAGARI,
    "doi": sanscript.DEVANAGARI,
    "awa": sanscript.DEVANAGARI,
    "mag": sanscript.DEVANAGARI,
    "mni": sanscript.BENGALI,  # Manipuri uses Bengali script
}

class IndianLanguageTranscriber:
    def __init__(self, model_size="small", device=None):
        """
        Initialize the transcriber with the specified Whisper model.

        Args:
            model_size: Size of the Whisper model (tiny, base, small, medium, large)
            device: Device to run the model on (cuda, cpu)
        """
        try:
            if device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"
            else:
                self.device = device

            print(f"Loading Whisper {model_size} model on {self.device}...")
            # Try loading with retries
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    self.model = whisper.load_model(model_size, device=self.device)
                    print("Model loaded successfully")
                    break
                except ConnectionError as e:
                    if attempt < max_retries - 1:
                       print(f"Connection error, retrying ({attempt+1}/{max_retries})...")
                       time.sleep(5)  # Wait 5 seconds before retry
                    else:
                       raise
            
            # Store model size for reference
            self.model_size = model_size
        except Exception as e:
            print(f"Error initializing transcriber: {e}")
            print(traceback.format_exc())
            raise

    def preprocess_audio(self, audio_path, target_sr=16000):
        """
        Preprocess audio file - load and resample if necessary.

        Args:
            audio_path: Path to the audio file
            target_sr: Target sample rate

        Returns:
            Preprocessed audio array
        """
        try:
            print(f"Preprocessing audio: {audio_path}")
            # Use a more direct approach with just NumPy and torch
            import soundfile as sf
        
            # Try with soundfile first
            try:
                audio, sr = sf.read(audio_path)
                # Make sure audio is float32
                audio = audio.astype(np.float32)
                # Convert to mono if stereo
                if len(audio.shape) > 1:
                   audio = audio.mean(axis=1).astype(np.float32)
            except Exception as e:
                print(f"Soundfile error: {e}, trying torchaudio...")
                # If soundfile fails, try torch's torchaudio
                import torchaudio
                audio, sr = torchaudio.load(audio_path)
                audio = audio.mean(dim=0).numpy().astype(np.float32)
        
            # Resample if needed
            if sr != target_sr:
               # Use a simpler resampling method
               import resampy
               audio = resampy.resample(audio, sr, target_sr)
               audio = audio.astype(np.float32)  # Ensure float32 after resampling
            return audio
        except Exception as e:
           print(f"Error preprocessing audio: {e}")
           print(traceback.format_exc())
           raise

    def transcribe(self, audio_path, language=None):
        """
        Transcribe audio file to text in native and Roman scripts.

        Args:
            audio_path: Path to the audio file
            language: Language code (e.g., 'hi' for Hindi, 'mr' for Marathi)

        Returns:
            Dictionary with transcriptions in both scripts
        """
        try:
            # If language not specified, try to detect
            if language == "auto" or language is None:
                print("No language specified, will try to auto-detect")
                language_option = None
            else:
                print(f"Transcribing in {LANGUAGE_MAP.get(language, language)} language")
                language_option = language

            # Preprocess audio
            audio = self.preprocess_audio(audio_path)
            # Ensure audio is the correct shape and dtype expected by whisper
            if len(audio.shape) > 1:
                audio = audio.mean(axis=1)
            audio = audio.astype(np.float32)

            # Set transcription options
            options = {
                "task": "transcribe",
                "verbose": True
            }

            # If language is specified, add it to options
            if language_option:
                options["language"] = language_option

            # Perform transcription
            print("Starting transcription...")
            result = self.model.transcribe(audio, **options)

            # Get native text (Whisper typically outputs in native script)
            native_text = result["text"].strip()
            detected_lang = result.get("language", "auto-detected")
            
            # Convert to Roman script if possible
            try:
                # Check if detected language is in our script map
                if detected_lang in SCRIPT_MAP:
                    source_script = SCRIPT_MAP[detected_lang]
                    roman_text = transliterate(native_text, source_script, sanscript.IAST)
                else:
                    # If script mapping unknown, just return the original text
                    roman_text = native_text
                    print(f"No script mapping for {detected_lang}, cannot transliterate")
            except Exception as e:
                print(f"Error in transliteration: {e}")
                roman_text = native_text
                
            return {
                "native": native_text,
                "roman": roman_text,
                "language": detected_lang,
                "segments": result.get("segments", [])
            }
        except Exception as e:
            print(f"Error in transcription: {e}")
            print(traceback.format_exc())
            raise

    def save_transcription(self, result, output_dir="./transcriptions", base_filename=None):
        """
        Save transcription results to files.

        Args:
            result: Transcription result dictionary
            output_dir: Directory to save output files
            base_filename: Base name for output files
        """
        try:
            # Create output directory if it doesn't exist
            os.makedirs(output_dir, exist_ok=True)

            # Generate base filename if not provided
            if base_filename is None:
                base_filename = f"transcription_{int(time.time())}"

            # Save native script text
            native_path = os.path.join(output_dir, f"{base_filename}_native.txt")
            with open(native_path, "w", encoding="utf-8") as f:
                f.write(result["native"])

            # Save Roman text
            roman_path = os.path.join(output_dir, f"{base_filename}_roman.txt")
            with open(roman_path, "w", encoding="utf-8") as f:
                f.write(result["roman"])

            # Save detailed results (including segments) as JSON
            json_path = os.path.join(output_dir, f"{base_filename}_full.json")
            with open(json_path, "w", encoding="utf-8") as f:
                # Convert segments to serializable format
                result_copy = result.copy()
                if "segments" in result_copy:
                    result_copy["segments"] = [dict(s) for s in result_copy["segments"]]

                json.dump(result_copy, f, ensure_ascii=False, indent=2)

            print(f"Transcription saved to {output_dir}/{base_filename}_*.txt")

            return {
                "native_path": native_path,
                "roman_path": roman_path,
                "json_path": json_path
            }
        except Exception as e:
            print(f"Error saving transcription: {e}")
            print(traceback.format_exc())
            raise

    def unload_model(self):
        """Unload model to free up memory"""
        try:
            if hasattr(self, 'model'):
                del self.model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print("Model unloaded")
        except Exception as e:
            print(f"Error unloading model: {e}")


def create_gradio_interface():
    """Create and launch Gradio interface for the transcriber."""
    
    # Global variables to store transcriber and results
    transcriber = None
    current_model_size = "small"  # Default model size
    
    def load_transcriber(model_size):
        nonlocal transcriber, current_model_size
        
        try:
            # Unload existing model if any
            if transcriber:
                transcriber.unload_model()
                transcriber = None
            
            # Load new model
            transcriber = IndianLanguageTranscriber(model_size=model_size)
            current_model_size = model_size
            return f"Loaded transcriber with {model_size} model on {transcriber.device}"
        except Exception as e:
            error_msg = f"Failed to load transcriber: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            return error_msg
    
    def display_waveform(audio_file):
        """Generate and return a waveform plot for the audio file"""
        try:
            # Load audio to display waveform without relying on librosa
            import soundfile as sf
            try:
               y, sr = sf.read(audio_file)
            except Exception:
               import torchaudio
               y, sr = torchaudio.load(audio_file)
               y = y.mean(dim=0).numpy()
        
            duration = len(y) / sr
            
            # Create waveform plot
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(np.linspace(0, duration, len(y)), y)
            ax.set_title('Audio Waveform')
            ax.set_xlabel('Time (s)')
            ax.set_ylabel('Amplitude')
            ax.set_xlim(0, duration)
            plt.tight_layout()
            
            return fig
        except Exception as e:
            print(f"Error creating waveform: {e}")
            print(traceback.format_exc())
            return None
    
    def process_audio(audio_file, model_size, language):
        """Process uploaded audio file and perform transcription"""
        try:
            if audio_file is None:
                return "Please upload an audio file first.", None, "", "", "", None, None, None
                
            # Load transcriber if needed or if model size changed
            nonlocal transcriber, current_model_size
            if transcriber is None or current_model_size != model_size:
                status = load_transcriber(model_size)
                print(status)
                if "Failed" in status:
                    return status, None, "", "", "", None, None, None
            
            # Generate waveform
            waveform = display_waveform(audio_file)
            
            # Check if audio file exists and is readable
            if not os.path.exists(audio_file):
                return f"Audio file not found: {audio_file}", waveform, "", "", "", None, None, None
                
            # Run transcription
            result = transcriber.transcribe(audio_file, language=language)
            
            # Extract results
            native_text = result["native"]
            roman_text = result["roman"]
            detected_lang = result["language"]
            
            # Format segments for display
            segments_text = ""
            for i, seg in enumerate(result.get("segments", [])):
                start = seg.get("start", 0)
                end = seg.get("end", 0)
                text = seg.get("text", "")
                segments_text += f"[{start:.2f}s - {end:.2f}s] {text}\n"
                
            status = f"Transcription completed. Detected language: {detected_lang}"
            
            # Save results
            temp_dir = tempfile.gettempdir()
            base_filename = Path(audio_file).stem
            file_paths = transcriber.save_transcription(
                result,
                output_dir=temp_dir,
                base_filename=base_filename
            )
            
            # Return results
            return (
                status,
                waveform,
                native_text,
                roman_text,
                segments_text,
                file_paths["native_path"],
                file_paths["roman_path"],
                file_paths["json_path"]
            )
                
        except Exception as e:
            error_msg = f"Error during transcription: {str(e)}"
            print(error_msg)
            print(traceback.format_exc())
            return error_msg, None, "", "", "", None, None, None
    
    # Create Gradio interface
    with gr.Blocks(title="Indian & Tribal Language Transcriber") as interface:
        gr.Markdown("# Indian & Tribal Language Transcription Tool")
        gr.Markdown("Upload audio files in Indian languages for automatic transcription and detection")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Input section
                audio_input = gr.Audio(type="filepath", label="Upload Audio File")
                
                with gr.Row():
                    model_size = gr.Dropdown(
                        choices=["tiny", "base", "small", "medium", "large"],
                        value="small",
                        label="Model Size"
                    )
                
                language = gr.Dropdown(
                    choices=list(LANGUAGE_MAP.keys()),
                    value="auto",
                    label="Language (select 'auto' for automatic detection)"
                )
                
                transcribe_btn = gr.Button("Transcribe", variant="primary")
                
                status_text = gr.Textbox(label="Status", interactive=False)
                
            with gr.Column(scale=2):
                # Visualization and results
                waveform_plot = gr.Plot(label="Audio Waveform")
                
                with gr.Tabs():
                    with gr.TabItem("Native Script"):
                        native_text = gr.Textbox(
                            label="Transcription (Native Script)",
                            interactive=False,
                            lines=10
                        )
                        native_file = gr.File(label="Download Native Script", visible=True)
                    
                    with gr.TabItem("Roman Script"):
                        roman_text = gr.Textbox(
                            label="Transcription (Roman Script)",
                            interactive=False,
                            lines=10
                        )
                        roman_file = gr.File(label="Download Roman Script", visible=True)
                        
                    with gr.TabItem("Segments"):
                        segments_text = gr.Textbox(
                            label="Segments with Timestamps",
                            interactive=False,
                            lines=10
                        )
                        json_file = gr.File(label="Download Full Results (JSON)", visible=True)
        
        # Add debugging information
        with gr.Accordion("Debug Info", open=False):
            debug_text = gr.Textbox(
                    label="System Information",
                value=f"Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}\n"
                      f"CUDA available: {torch.cuda.is_available()}\n"
                      f"PyTorch version: {torch.__version__}\n",
                interactive=False
            )
        
        # Connect the button to the processing function
        transcribe_btn.click(
            fn=process_audio,
            inputs=[audio_input, model_size, language],
            outputs=[status_text, waveform_plot, native_text, roman_text, segments_text, native_file, roman_file, json_file]
        )
        
        # Load model on startup
        interface.load(
            fn=lambda: load_transcriber("small"),
            inputs=None,
            outputs=None
        )
    
    return interface

if __name__ == "__main__":
    # Create and launch the interface
    interface = create_gradio_interface()
    interface.launch(share=True)  # Set share=False if you don't want a public link