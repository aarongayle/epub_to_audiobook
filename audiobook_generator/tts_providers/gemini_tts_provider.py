import io
import logging
import math
import os
import wave

from google import genai
from google.genai import types

from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.utils.utils import split_text, set_audio_tags, merge_audio_segments
from audiobook_generator.tts_providers.base_tts_provider import BaseTTSProvider


logger = logging.getLogger(__name__)


def get_gemini_supported_output_formats():
    # Native Gemini TTS only outputs PCM/WAV
    return ["wav"]


def get_gemini_supported_voices():
    return [
        # Female voices
        "Achernar",
        "Aoede",
        "Autonoe",
        "Callirrhoe",
        "Despina",
        "Erinome",
        "Gacrux",
        "Kore",
        "Laomedeia",
        "Leda",
        "Pulcherrima",
        "Sulafat",
        "Vindemiatrix",
        "Zephyr",
        # Male voices
        "Achird",
        "Algenib",
        "Algieba",
        "Alnilam",
        "Charon",
        "Enceladus",
        "Fenrir",
        "Iapetus",
        "Orus",
        "Puck",
        "Rasalgethi",
        "Sadachbia",
        "Sadaltager",
        "Schedar",
        "Umbriel",
        "Zubenelgenubi",
    ]


def get_gemini_supported_models():
    return ["gemini-2.5-flash-preview-tts", "gemini-2.5-pro-preview-tts"]


def get_price(model):
    # Pricing TBD - using placeholder values
    if model == "gemini-2.5-flash-preview-tts":
        return 0.0  # Update with actual pricing when available
    elif model == "gemini-2.5-pro-preview-tts":
        return 0.0  # Update with actual pricing when available
    else:
        logger.warning(f"Gemini: Unsupported model name: {model}, unable to retrieve the price")
        return 0.0


def pcm_to_wav_bytes(pcm_data: bytes, channels: int = 1, rate: int = 24000, sample_width: int = 2) -> bytes:
    """Convert raw PCM data to WAV format in memory."""
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(sample_width)
        wf.setframerate(rate)
        wf.writeframes(pcm_data)
    wav_buffer.seek(0)
    return wav_buffer.read()


class GeminiTTSProvider(BaseTTSProvider):
    def __init__(self, config: GeneralConfig):
        config.model_name = config.model_name or "gemini-2.5-flash-preview-tts"
        config.voice_name = config.voice_name or "Kore"
        config.output_format = config.output_format or "wav"

        self.price = get_price(config.model_name)
        
        # Check for API key before calling super().__init__ which calls validate_config
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")

        # Initialize the Google GenAI client
        self.client = genai.Client(api_key=api_key)
        
        super().__init__(config)

    def __str__(self) -> str:
        return super().__str__()

    def text_to_speech(self, text: str, output_file: str, audio_tags: AudioTags):
        max_chars = 1800

        text_chunks = split_text(text, max_chars, self.config.language)

        audio_segments = []
        chunk_ids = []

        for i, chunk in enumerate(text_chunks, 1):
            chunk_id = f"chapter-{audio_tags.idx}_{audio_tags.title}_chunk_{i}_of_{len(text_chunks)}"
            logger.info(
                f"Processing {chunk_id}, length={len(chunk)}"
            )
            logger.debug(
                f"Processing {chunk_id}, length={len(chunk)}, text=[{chunk}]"
            )

            response = self.client.models.generate_content(
                model=self.config.model_name,
                contents=chunk,
                config=types.GenerateContentConfig(
                    response_modalities=["AUDIO"],
                    speech_config=types.SpeechConfig(
                        voice_config=types.VoiceConfig(
                            prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                voice_name=self.config.voice_name,
                            )
                        )
                    ),
                )
            )

            # Extract PCM audio data from response
            pcm_data = response.candidates[0].content.parts[0].inline_data.data
            
            # Convert PCM to WAV format
            wav_data = pcm_to_wav_bytes(pcm_data)
            
            logger.debug(f"Received audio chunk: {len(wav_data)} bytes")

            audio_segments.append(io.BytesIO(wav_data))
            chunk_ids.append(chunk_id)

        merge_audio_segments(audio_segments, output_file, self.config.output_format, chunk_ids, self.config.use_pydub_merge)

        set_audio_tags(output_file, audio_tags)

    def get_break_string(self):
        return "   "

    def get_output_file_extension(self):
        return self.config.output_format

    def validate_config(self):
        if self.config.output_format not in get_gemini_supported_output_formats():
            raise ValueError(f"Gemini: Unsupported output format: {self.config.output_format}. Only 'wav' is supported.")
        if self.config.voice_name not in get_gemini_supported_voices():
            raise ValueError(f"Gemini: Unsupported voice: {self.config.voice_name}. Voice names are case-sensitive.")

    def estimate_cost(self, total_chars):
        return math.ceil(total_chars / 1000) * self.price
