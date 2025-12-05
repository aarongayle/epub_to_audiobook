import io
import logging
import math
import os

from openai import OpenAI

from audiobook_generator.core.audio_tags import AudioTags
from audiobook_generator.config.general_config import GeneralConfig
from audiobook_generator.utils.utils import split_text, set_audio_tags, merge_audio_segments
from audiobook_generator.tts_providers.base_tts_provider import BaseTTSProvider


logger = logging.getLogger(__name__)


def get_gemini_supported_output_formats():
    return ["mp3", "aac", "flac", "opus", "wav"]


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
    return ["gemini-2.5-flash-tts", "gemini-2.5-pro-tts"]


def get_price(model):
    # Pricing TBD - using placeholder values
    if model == "gemini-2.5-flash-tts":
        return 0.0  # Update with actual pricing when available
    elif model == "gemini-2.5-pro-tts":
        return 0.0  # Update with actual pricing when available
    else:
        logger.warning(f"Gemini: Unsupported model name: {model}, unable to retrieve the price")
        return 0.0


class GeminiTTSProvider(BaseTTSProvider):
    def __init__(self, config: GeneralConfig):
        config.model_name = config.model_name or "gemini-2.5-flash-tts"
        config.voice_name = config.voice_name or "Kore"
        config.speed = config.speed or 1.0
        config.output_format = config.output_format or "mp3"

        self.price = get_price(config.model_name)
        super().__init__(config)

        # Gemini uses OpenAI-compatible API with custom base URL
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set")

        self.client = OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
            max_retries=4,
        )

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

            response = self.client.audio.speech.create(
                model=self.config.model_name,
                voice=self.config.voice_name,
                speed=self.config.speed,
                input=chunk,
                response_format=self.config.output_format,
            )

            logger.debug(f"Remote server response: status_code={response.response.status_code}, "
                         f"size={len(response.content)} bytes, "
                         f"content={response.content[:128]}...")

            audio_segments.append(io.BytesIO(response.content))
            chunk_ids.append(chunk_id)

        merge_audio_segments(audio_segments, output_file, self.config.output_format, chunk_ids, self.config.use_pydub_merge)

        set_audio_tags(output_file, audio_tags)

    def get_break_string(self):
        return "   "

    def get_output_file_extension(self):
        return self.config.output_format

    def validate_config(self):
        if self.config.output_format not in get_gemini_supported_output_formats():
            raise ValueError(f"Gemini: Unsupported output format: {self.config.output_format}")
        if self.config.speed < 0.25 or self.config.speed > 4.0:
            raise ValueError(f"Gemini: Unsupported speed: {self.config.speed}")
        if self.config.voice_name not in get_gemini_supported_voices():
            raise ValueError(f"Gemini: Unsupported voice: {self.config.voice_name}. Voice names are case-sensitive.")

    def estimate_cost(self, total_chars):
        return math.ceil(total_chars / 1000) * self.price

