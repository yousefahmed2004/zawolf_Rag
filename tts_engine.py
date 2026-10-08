import os
import re
import shutil
import tempfile
import threading
import time
import wave
from typing import Optional, Iterator, Tuple, List

from gradio_client import Client, handle_file


# ============================================================
# CONFIG
# ============================================================
# الموديل مش بيتحمل محليًا خالص. بنكلم Hugging Face Space
# (Gradio + ZeroGPU) عبر gradio_client.
# ============================================================

TTS_SPACE_ID = os.getenv("TTS_SPACE_ID", "mohammedaly22/VoiceTut-TTS")

# اختياري: بيديك quota أعلى على ZeroGPU من الاستخدام المجهول
HF_TOKEN = os.getenv("HF_TOKEN") or None

TTS_DEFAULT_SPEAKER = os.getenv("TTS_DEFAULT_SPEAKER", "Mohamed")
TTS_LANGUAGE = os.getenv("TTS_LANGUAGE", "العربية (Egyptian)")  # أو "English"

# عدد الـsteps: كل ما قل كل ما كان أسرع (الـSpace بيقبل 8..64)
# 16-24 كفاية جدًا للمكالمة.
TTS_NUM_STEP = int(os.getenv("TTS_NUM_STEP", "20"))
TTS_GUIDANCE_SCALE = float(os.getenv("TTS_GUIDANCE_SCALE", "2.5"))
TTS_SPEED = float(os.getenv("TTS_SPEED", "1.0"))

# أقصى عدد حروف بيتنطق في الرد الواحد
TTS_MAX_CHARS = int(os.getenv("TTS_MAX_CHARS", "300"))

# حجم الجزء الواحد اللي بيتبعت للـSpace
TTS_CHUNK_CHARS = int(os.getenv("TTS_CHUNK_CHARS", "200"))

# حجم أول جزء (قصير عشان الصوت يبدأ بسرعة)
TTS_FIRST_CHUNK_CHARS = int(os.getenv("TTS_FIRST_CHUNK_CHARS", "90"))

# أسماء الـendpoints (Gradio بيسميها من اسم الدالة في app.py بتاع الـSpace)
TTS_API_BUILTIN = os.getenv("TTS_API_BUILTIN", "/run_b_oneshot")
TTS_API_CLONE = os.getenv("TTS_API_CLONE", "/run_c_oneshot")

SILENCE_BETWEEN_CHUNKS_SEC = 0.15


# ============================================================
# CLIENT (lazy singleton)
# ============================================================

_client = None
_client_lock = threading.Lock()

# ZeroGPU quota محدود، فبنبعت طلب واحد في المرة
_synth_lock = threading.Lock()


def get_tts() -> Client:
    """يرجّع Gradio Client متصل بالـSpace. مفيش تحميل موديل."""
    global _client

    if _client is None:
        with _client_lock:
            if _client is None:
                kwargs = {"verbose": False}

                if HF_TOKEN:
                    # gradio_client الجديد: token / القديم: hf_token
                    try:
                        _client = Client(
                            TTS_SPACE_ID, token=HF_TOKEN, **kwargs
                        )
                    except TypeError:
                        _client = Client(
                            TTS_SPACE_ID, hf_token=HF_TOKEN, **kwargs
                        )
                else:
                    _client = Client(TTS_SPACE_ID, **kwargs)

    return _client


# ============================================================
# RETRY
# ============================================================

_TRANSIENT_MARKERS = (
    "503", "502", "504", "timeout", "timed out",
    "connection", "temporarily", "queue",
)


def _call_with_retry(fn, *args, max_retries: int = 3,
                     base_delay: float = 2.0, **kwargs):

    last_error = None

    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)

        except Exception as e:
            last_error = e
            msg = str(e).lower()

            transient = any(m in msg for m in _TRANSIENT_MARKERS)

            if transient and attempt < max_retries - 1:
                wait = base_delay * (2 ** attempt)
                print(
                    f"TTS Space transient error "
                    f"(attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {wait:.1f}s: {e}"
                )
                time.sleep(wait)
                continue

            raise

    raise last_error


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_for_tts(text: str) -> str:
    """يشيل markdown والإيموجي والرموز اللي الموديل مش هيعرف ينطقها."""
    if not text:
        return ""

    text = re.sub(r"```.*?```", " ", text, flags=re.S)
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = re.sub(r"[#*_`>|]+", " ", text)
    text = re.sub(r"[\U0001F300-\U0001FAFF\u2600-\u27BF]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def truncate_for_speech(text: str, max_chars: int = TTS_MAX_CHARS) -> str:
    """يقص النص عند آخر نهاية جملة قبل الحد الأقصى."""
    if len(text) <= max_chars:
        return text

    cut = text[:max_chars]

    last_end = max(
        cut.rfind("."), cut.rfind("!"),
        cut.rfind("؟"), cut.rfind("?"),
    )

    if last_end > max_chars * 0.4:
        return cut[: last_end + 1]

    return cut


_SENT_SPLIT_RE = re.compile(r"(?<=[.!?؟…])\s+")


def split_into_chunks(text: str, max_chars: int = TTS_CHUNK_CHARS) -> List[str]:
    """يقسّم النص لأجزاء عند حدود الجمل، كل جزء <= max_chars."""
    sentences = [s.strip() for s in _SENT_SPLIT_RE.split(text) if s.strip()]

    chunks: List[str] = []
    cur = ""

    for s in sentences:

        # جملة أطول من الحد: نقطعها عند مسافة
        while len(s) > max_chars:
            cut = s.rfind(" ", 0, max_chars)
            if cut <= 0:
                cut = max_chars
            piece, s = s[:cut].strip(), s[cut:].strip()

            if cur:
                chunks.append(cur)
                cur = ""
            if piece:
                chunks.append(piece)

        if not s:
            continue

        if cur and len(cur) + 1 + len(s) > max_chars:
            chunks.append(cur)
            cur = s
        else:
            cur = f"{cur} {s}".strip()

    if cur:
        chunks.append(cur)

    return chunks


def split_for_streaming(
    text: str,
    max_chars: int = TTS_CHUNK_CHARS,
    first_chars: int = TTS_FIRST_CHUNK_CHARS,
) -> List[str]:
    """
    زي split_into_chunks بس أول جزء قصير (~first_chars)
    عشان أول صوت يطلع بسرعة، والباقي بحجم عادي.
    """
    chunks = split_into_chunks(text, max_chars)

    if not chunks or len(chunks[0]) <= first_chars:
        return chunks

    pieces = split_into_chunks(chunks[0], first_chars)

    if len(pieces) <= 1:
        return chunks

    rest = " ".join(pieces[1:]).strip()

    result = [pieces[0]]

    if rest:
        result.append(rest)

    result.extend(chunks[1:])

    return result


# ============================================================
# ONE CALL TO THE SPACE -> local wav path
# ============================================================

def _extract_path(result) -> str:
    audio = result[0] if isinstance(result, (list, tuple)) else result

    if isinstance(audio, dict):
        audio = audio.get("path") or audio.get("value")

    if not isinstance(audio, str) or not os.path.exists(audio):
        raise RuntimeError(f"Unexpected TTS Space response: {result!r}")

    return audio


def _predict_chunk(
    text: str,
    speaker: Optional[str],
    ref_audio: Optional[str],
    ref_text: Optional[str],
    num_step: int,
    guidance_scale: float,
    speed: float,
) -> str:

    client = get_tts()

    if ref_audio:
        # استنساخ صوت: ref_audio, ref_text, text, language, steps, guidance, speed, normalize
        result = _call_with_retry(
            client.predict,
            handle_file(ref_audio),
            ref_text or "",
            text,
            TTS_LANGUAGE,
            num_step,
            guidance_scale,
            speed,
            True,
            api_name=TTS_API_CLONE,
        )
    else:
        # صوت جاهز: speaker, text, language, steps, guidance, speed, normalize
        result = _call_with_retry(
            client.predict,
            speaker or TTS_DEFAULT_SPEAKER,
            text,
            TTS_LANGUAGE,
            num_step,
            guidance_scale,
            speed,
            True,
            api_name=TTS_API_BUILTIN,
        )

    return _extract_path(result)


# ============================================================
# WAV HELPERS (بدون dependencies إضافية)
# ============================================================

def _read_pcm(path: str):
    """يرجّع (sample_rate, channels, sampwidth, frames_bytes)."""
    try:
        with wave.open(path, "rb") as w:
            return (
                w.getframerate(), w.getnchannels(),
                w.getsampwidth(), w.readframes(w.getnframes()),
            )
    except wave.Error:
        # لو الملف float مش PCM: نحوله عبر soundfile
        import soundfile as sf

        data, sr = sf.read(path, dtype="int16")
        channels = 1 if data.ndim == 1 else data.shape[1]
        return sr, channels, 2, data.tobytes()


def _write_pcm(path: str, sr: int, channels: int, sampwidth: int, frames: bytes):
    with wave.open(path, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(sampwidth)
        w.setframerate(sr)
        w.writeframes(frames)


# ============================================================
# TEXT -> SPEECH (FILE) - جزء واحد
# ============================================================

def synthesize_speech(
    text: str,
    output_path: str,
    speaker: Optional[str] = None,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
    num_step: int = TTS_NUM_STEP,
    guidance_scale: float = TTS_GUIDANCE_SCALE,
    speed: float = TTS_SPEED,
) -> str:

    if not text or not text.strip():
        raise ValueError("Cannot synthesize empty text")

    with _synth_lock:
        src = _predict_chunk(
            text.strip(), speaker, ref_audio, ref_text,
            num_step, guidance_scale, speed,
        )

    shutil.copyfile(src, output_path)

    return output_path


# ============================================================
# TEXT -> SPEECH (BYTES)
# ============================================================

def synthesize_speech_bytes(
    text: str,
    speaker: Optional[str] = None,
    ref_audio: Optional[str] = None,
    ref_text: Optional[str] = None,
) -> bytes:

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        synthesize_speech(
            text, output_path=path, speaker=speaker,
            ref_audio=ref_audio, ref_text=ref_text,
        )

        with open(path, "rb") as f:
            return f.read()

    finally:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass


# ============================================================
# CHUNK (جملة واحدة) -> WAV bytes
# ============================================================
# بيستخدمه /speak-one في المكالمة الصوتية (streaming جملة بجملة).
# ============================================================

def synthesize_chunk_bytes(
    text: str,
    speaker: Optional[str] = None,
) -> Tuple[bytes, str]:
    """يرجّع (wav bytes, media_type) من موديلك على الـSpace."""

    return synthesize_speech_bytes(text, speaker), "audio/wav"


# ============================================================
# LONG TEXT -> FILE / BYTES
# ============================================================
# بنقسّم النص جمل، نبعت كل جزء للـSpace، وندمج الـWAVs
# مع فاصل صمت صغير بينهم.
# ============================================================

def synthesize_long_speech(
    text: str,
    output_path: str,
    speaker: Optional[str] = None,
) -> str:

    if not text or not text.strip():
        raise ValueError("Cannot synthesize empty text")

    chunks = split_into_chunks(text.strip())

    if not chunks:
        raise ValueError("Cannot synthesize empty text")

    parts = []

    with _synth_lock:
        for chunk in chunks:
            path = _predict_chunk(
                chunk, speaker, None, None,
                TTS_NUM_STEP, TTS_GUIDANCE_SCALE, TTS_SPEED,
            )
            parts.append(_read_pcm(path))

    sr, channels, sampwidth, _ = parts[0]

    silence = b"\x00" * (
        int(sr * SILENCE_BETWEEN_CHUNKS_SEC) * channels * sampwidth
    )

    frames = silence.join(p[3] for p in parts)

    _write_pcm(output_path, sr, channels, sampwidth, frames)

    return output_path


def synthesize_long_speech_bytes(
    text: str,
    speaker: Optional[str] = None,
) -> bytes:

    fd, path = tempfile.mkstemp(suffix=".wav")
    os.close(fd)

    try:
        synthesize_long_speech(text, path, speaker=speaker)

        with open(path, "rb") as f:
            return f.read()

    finally:
        if os.path.exists(path):
            try:
                os.remove(path)
            except Exception:
                pass


# ============================================================
# STREAMING (كل جزء لما يخلص)
# ============================================================

def stream_speech(
    text: str,
    speaker: Optional[str] = None,
) -> Iterator[Tuple[int, bytes]]:

    if not text or not text.strip():
        raise ValueError("Cannot synthesize empty text")

    for chunk in split_for_streaming(text.strip()):
        with _synth_lock:
            path = _predict_chunk(
                chunk, speaker, None, None,
                TTS_NUM_STEP, TTS_GUIDANCE_SCALE, TTS_SPEED,
            )

        sr, _, _, frames = _read_pcm(path)

        yield sr, frames