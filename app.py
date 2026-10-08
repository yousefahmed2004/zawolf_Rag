import os
import tempfile
import threading
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from pypdf import PdfReader
from starlette.concurrency import run_in_threadpool

from rag import (
    init_qdrant,
    add_document_structured,
    ask_rag,
    analyze_image_with_context,
    ask_rag_with_image,
    transcribe_audio
)

from ocr_utils import (
    needs_ocr,
    ocr_pdf_page,
    OCR_AVAILABLE
)

from tts_engine import (
    get_tts,
    clean_for_tts,
    truncate_for_speech,
    split_for_streaming,
    synthesize_speech_bytes,
    synthesize_long_speech_bytes
)


# ============================================================
# FASTAPI
# ============================================================

app = FastAPI(
    title="Zawolf Industrial Multimodal RAG",
    description="Industrial Document-Aware Multimodal RAG",
    version="2.3.0"
)


# ============================================================
# STATIC FILES
# ============================================================

app.mount(
    "/static",
    StaticFiles(directory="static"),
    name="static"
)


# ============================================================
# REQUEST MODELS
# ============================================================

class Question(BaseModel):
    question: str
    # True = مكالمة صوتية: رد قصير + context أصغر
    voice: bool = False


class SpeakRequest(BaseModel):
    text: str
    speaker: Optional[str] = None


# ============================================================
# ALLOWED FILE TYPES
# ============================================================

ALLOWED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp"
}

ALLOWED_DOCUMENT_EXTENSIONS = {
    ".pdf",
    ".txt"
}


# ============================================================
# STARTUP
# ============================================================

@app.on_event("startup")
def startup():

    print("=" * 60)
    print("Starting Zawolf Industrial Multimodal RAG...")
    print("=" * 60)

    try:

        init_qdrant()

        print("Qdrant initialized successfully.")

    except Exception as e:

        print(f"Qdrant initialization failed: {e}")

        raise

    # تحميل موديل الـ TTS في الخلفية عشان أول رد صوتي ميستناش
    def _preload_tts():

        try:

            get_tts()

            print("TTS model loaded.")

        except Exception as e:

            print(f"TTS preload failed: {e}")

    threading.Thread(
        target=_preload_tts,
        daemon=True
    ).start()


# ============================================================
# HOME
# ============================================================

@app.get("/")
def home():

    return FileResponse(
        "static/index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "ok",
        "service": "Zawolf Industrial Multimodal RAG",
        "version": "2.3.0"
    }


# ============================================================
# UPLOAD PDF / TXT
# ============================================================

@app.post("/upload")
async def upload_document(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="اسم الملف غير موجود"
        )

    filename = file.filename

    extension = os.path.splitext(
        filename
    )[1].lower()

    if extension not in ALLOWED_DOCUMENT_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail="مسموح فقط برفع ملفات PDF أو TXT"
        )

    content = await file.read()

    if not content:

        raise HTTPException(
            status_code=400,
            detail="الملف فارغ"
        )

    total_chunks = 0

    # ========================================================
    # PDF
    # ========================================================

    if extension == ".pdf":

        if not content.startswith(b"%PDF"):

            raise HTTPException(
                status_code=400,
                detail="الملف المرفوع ليس PDF صالحًا"
            )

        temp_path = None

        try:

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=".pdf"
            ) as temp_file:

                temp_path = temp_file.name

                temp_file.write(content)

            reader = PdfReader(
                temp_path
            )

            total_pages = len(
                reader.pages
            )

            print(
                f"Processing PDF: {filename}"
            )

            print(
                f"Total pages: {total_pages}"
            )

            for page_number, page in enumerate(
                reader.pages,
                start=1
            ):

                try:

                    text = page.extract_text() or ""

                    # OCR fallback
                    if needs_ocr(text):

                        print(
                            f"Page {page_number}: "
                            "Weak text detected."
                        )

                        if OCR_AVAILABLE:

                            try:

                                ocr_text = ocr_pdf_page(
                                    temp_path,
                                    page_number
                                )

                                if ocr_text:

                                    text = ocr_text

                                    print(
                                        f"Page {page_number}: "
                                        "OCR successful."
                                    )

                            except Exception as e:

                                print(
                                    f"Page {page_number}: "
                                    f"OCR failed: {e}"
                                )

                        else:

                            print(
                                f"Page {page_number}: "
                                "OCR unavailable."
                            )

                    if not text.strip():

                        print(
                            f"Page {page_number}: "
                            "No usable text."
                        )

                        continue

                    chunks = add_document_structured(
                        text=text,
                        filename=filename,
                        page=page_number
                    )

                    total_chunks += chunks

                    print(
                        f"Page {page_number}: "
                        f"{chunks} chunks added."
                    )

                except Exception as e:

                    print(
                        f"Error processing page "
                        f"{page_number}: {e}"
                    )

                    continue

        except Exception as e:

            print(
                f"PDF processing error: {e}"
            )

            raise HTTPException(
                status_code=400,
                detail="تعذر قراءة ملف PDF. تأكد أن الملف PDF سليم وغير تالف."
            )

        finally:

            if (
                temp_path
                and os.path.exists(temp_path)
            ):

                try:

                    os.remove(
                        temp_path
                    )

                except Exception as e:

                    print(
                        f"Failed to delete temp PDF: {e}"
                    )

    # ========================================================
    # TXT
    # ========================================================

    elif extension == ".txt":

        text = content.decode(
            "utf-8",
            errors="ignore"
        )

        if not text.strip():

            raise HTTPException(
                status_code=400,
                detail="ملف TXT فارغ"
            )

        try:

            total_chunks = add_document_structured(
                text=text,
                filename=filename
            )

        except Exception as e:

            print(
                f"TXT processing error: {e}"
            )

            raise HTTPException(
                status_code=500,
                detail="حدث خطأ أثناء معالجة الملف"
            )

    return {
        "success": True,
        "filename": filename,
        "chunks": total_chunks
    }


# ============================================================
# TEXT RAG CHAT
# ============================================================

@app.post("/chat")
def chat(
    data: Question
):

    question = data.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="اكتب السؤال"
        )

    try:

        result = ask_rag(
            question,
            voice=data.voice
        )

        return result

    except Exception as e:

        print(
            f"Chat error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="حدث خطأ أثناء معالجة السؤال. جرب تاني بعد شوية."
        )


# ============================================================
# IMAGE ANALYSIS
# ============================================================

@app.post("/analyze-image")
async def analyze_image(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="اسم الصورة غير موجود"
        )

    if file.content_type not in ALLOWED_IMAGE_TYPES:

        raise HTTPException(
            status_code=400,
            detail="مسموح فقط بصور JPG أو PNG أو WEBP"
        )

    image_bytes = await file.read()

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="الصورة فارغة"
        )

    try:

        answer = await run_in_threadpool(
            analyze_image_with_context,
            image_bytes=image_bytes,
            mime_type=file.content_type
        )

        return {
            "success": True,
            "filename": file.filename,
            "answer": answer
        }

    except Exception as e:

        print(
            f"Image analysis error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="حدث خطأ أثناء تحليل الصورة"
        )


# ============================================================
# IMAGE + RAG
# ============================================================

@app.post("/chat-with-image")
async def chat_with_image(
    question: str,
    file: UploadFile = File(...)
):

    question = question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="اكتب السؤال المرتبط بالصورة"
        )

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="اسم الصورة غير موجود"
        )

    if file.content_type not in ALLOWED_IMAGE_TYPES:

        raise HTTPException(
            status_code=400,
            detail="مسموح فقط بصور JPG أو PNG أو WEBP"
        )

    image_bytes = await file.read()

    if not image_bytes:

        raise HTTPException(
            status_code=400,
            detail="الصورة فارغة"
        )

    try:

        result = await run_in_threadpool(
            ask_rag_with_image,
            question=question,
            image_bytes=image_bytes,
            mime_type=file.content_type
        )

        return {
            "success": True,
            "filename": file.filename,
            **result
        }

    except Exception as e:

        print(
            f"Image RAG error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="حدث خطأ أثناء تحليل الصورة والبحث في قاعدة المعرفة"
        )


# ============================================================
# AUDIO TRANSCRIPTION
# ============================================================

@app.post("/transcribe")
async def transcribe(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="اسم الملف غير موجود"
        )

    content_type = file.content_type or ""

    if not content_type.startswith("audio/"):

        raise HTTPException(
            status_code=400,
            detail="مسموح فقط بملفات صوتية"
        )

    audio_bytes = await file.read()

    if not audio_bytes:

        raise HTTPException(
            status_code=400,
            detail="التسجيل الصوتي فارغ"
        )

    try:

        text = await run_in_threadpool(
            transcribe_audio,
            audio_bytes=audio_bytes,
            mime_type=content_type
        )

        return {
            "success": True,
            "filename": file.filename,
            "text": text
        }

    except Exception as e:

        print(
            f"Transcription error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="حدث خطأ أثناء تحويل الصوت إلى نص"
        )


# ============================================================
# TEXT TO SPEECH (used by Live Voice Call in index.html)
# ============================================================
# التوليد blocking، فبنشغله في threadpool
# عشان ميجمدش الـ event loop.
#
# فيه طريقتين:
#   1) /speak                  : بيرجّع الرد كله wav واحد (القديم)
#   2) /speak-chunks + /speak-one : streaming - الفرونت بيطلب كل
#      جزء لوحده ويشغّله أول ما يجهز (أسرع بكتير في المكالمة)
# ============================================================

@app.post("/speak")
async def speak(
    data: SpeakRequest
):

    text = truncate_for_speech(
        clean_for_tts(data.text)
    )

    if not text:

        raise HTTPException(
            status_code=400,
            detail="مفيش نص للنطق"
        )

    try:

        audio = await run_in_threadpool(
            synthesize_long_speech_bytes,
            text,
            data.speaker
        )

        return Response(
            content=audio,
            media_type="audio/wav"
        )

    except Exception as e:

        print(
            f"TTS error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="حدث خطأ أثناء تحويل النص لصوت"
        )


@app.post("/speak-chunks")
async def speak_chunks(
    data: SpeakRequest
):
    """يرجّع النص منضّف ومقسّم لأجزاء (أول جزء قصير)."""

    text = truncate_for_speech(
        clean_for_tts(data.text)
    )

    if not text:

        raise HTTPException(
            status_code=400,
            detail="مفيش نص للنطق"
        )

    return {
        "chunks": split_for_streaming(text)
    }


@app.post("/speak-one")
async def speak_one(
    data: SpeakRequest
):
    """يولّد صوت جزء واحد (wav)."""

    text = clean_for_tts(data.text)

    if not text:

        raise HTTPException(
            status_code=400,
            detail="مفيش نص للنطق"
        )

    try:

        audio = await run_in_threadpool(
            synthesize_speech_bytes,
            text,
            data.speaker
        )

        return Response(
            content=audio,
            media_type="audio/wav"
        )

    except Exception as e:

        print(
            f"TTS error: {e}"
        )

        raise HTTPException(
            status_code=500,
            detail="حدث خطأ أثناء تحويل النص لصوت"
        )


# ============================================================
# RUN
# ============================================================
# reload=False: الـ reload بيعيد تحميل موديل الـ TTS
# مع كل تعديل في الكود.
# ============================================================

if __name__ == "__main__":

    import uvicorn

    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False
    )