import os
import re
import io
import json
import time
import uuid
import base64
import threading
from typing import Optional, List, Dict, Any

from dotenv import load_dotenv
from groq import Groq

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    Filter,
    FieldCondition,
    MatchValue,
    Range,
    FilterSelector,
    PayloadSchemaType,
)

load_dotenv()


# ============================================================
# GROQ
# ============================================================
# Groq بيوفر: Chat (Llama / GPT-OSS) + Vision + Whisper (تفريغ صوت).
# Groq مفيهوش Embeddings، فالـEmbeddings بقت محلية (تحت).
# ============================================================

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# موديل الشات والإجابة (بيتقرا من .env دلوقتي)
CHAT_MODEL = os.getenv("CHAT_MODEL", "openai/gpt-oss-120b")

# موديل خطوة "فهم السؤال" (JSON بسيط) - موديل خفيف وسريع.
# سيبه فاضي لو عايزه يستخدم نفس CHAT_MODEL.
UNDERSTAND_MODEL = os.getenv("UNDERSTAND_MODEL", "openai/gpt-oss-20b") or None

# مجهود التفكير لموديلات gpt-oss فقط: low / medium / high
# low = أسرع بكتير. (بيتجاهل لو الموديل مش gpt-oss)
REASONING_EFFORT = os.getenv("REASONING_EFFORT", "low")

# موديل تحليل الصور (لازم يدعم vision)
# ملحوظة: llama-4-scout اتشال من Groq (17 يوليو 2026). البديل الموصى بيه
# موديل Qwen متعدد الوسائط (multimodal). اتأكد من الاسم من قايمة الموديلات
# المتاحة على حسابك (شوف الأمر اللي في الشرح).
VISION_MODEL = os.getenv(
    "VISION_MODEL",
    "qwen/qwen3.8-27b"
)

# موديل تحويل الصوت لنص
STT_MODEL = os.getenv("STT_MODEL", "whisper-large-v3")
STT_LANGUAGE = os.getenv("STT_LANGUAGE", "ar")
STT_PROMPT = os.getenv("STT_PROMPT", "")

# إعدادات وضع المكالمة الصوتية
VOICE_RETRIEVAL_LIMIT = int(os.getenv("VOICE_RETRIEVAL_LIMIT", "4"))
VOICE_MAX_TOKENS = int(os.getenv("VOICE_MAX_TOKENS", "1200"))

# موديل مخصص للمكالمة الصوتية: لازم يكون سريع في أول كلمة.
# llama-3.x اتشالت من Groq (16 أغسطس 2026). gpt-oss-20b هو الأسرع المتاح
# (حوالي 1000 token/s) ومعاه reasoning_effort=low.
VOICE_CHAT_MODEL = os.getenv("VOICE_CHAT_MODEL", "openai/gpt-oss-20b")

# أقل طول للجملة قبل ما نبعتها للـTTS (عشان منبعتش جمل صغيرة جدًا)
VOICE_MIN_SENTENCE = int(os.getenv("VOICE_MIN_SENTENCE", "15"))

# ------------------------------------------------------------
# ذاكرة المكالمة الصوتية (متخزنة في Qdrant)
# ------------------------------------------------------------
# عدد الأدوار (سؤال + رد) اللي بنفتكرها لكل جلسة
VOICE_MEMORY_TURNS = int(os.getenv("VOICE_MEMORY_TURNS", "8"))

# بعد كام ثانية من آخر كلام الجلسة بتتنسي (افتراضي 30 دقيقة)
VOICE_MEMORY_TTL = int(os.getenv("VOICE_MEMORY_TTL", "1800"))

# أقصى طول لرد المساعد المحفوظ في الذاكرة (عشان الـprompt ميكبرش)
VOICE_MEMORY_ANSWER_CHARS = int(
    os.getenv("VOICE_MEMORY_ANSWER_CHARS", "350")
)

# عدد النتايج المرشّحة من Qdrant لكل استعلام (كان 30)
SEMANTIC_CANDIDATES = int(os.getenv("SEMANTIC_CANDIDATES", "15"))

if not GROQ_API_KEY:
    raise RuntimeError(
        "GROQ_API_KEY is missing from .env"
    )

groq_client = Groq(
    api_key=GROQ_API_KEY,
    max_retries=0  # الـretry بنتحكم فيه إحنا تحت
)


# ============================================================
# RETRY / BACKOFF WRAPPER FOR GROQ CALLS
# ============================================================
# بيعيد المحاولة بس مع الأخطاء المؤقتة (ضغط / rate limit / شبكة)
# وأي خطأ تاني (مفتاح غلط، مدخل غلط...) بيتعمله raise فورًا.
# ============================================================

TRANSIENT_ERROR_MARKERS = (
    "503",
    "502",
    "500",
    "429",
    "rate limit",
    "rate_limit",
    "overloaded",
    "unavailable",
    "timeout",
    "timed out",
    "connection",
)


def _call_with_retry(
    fn,
    *args,
    max_retries: int = 3,
    base_delay: float = 2.0,
    **kwargs
):

    last_error = None

    for attempt in range(max_retries):

        try:

            return fn(*args, **kwargs)

        except Exception as e:

            last_error = e

            error_str = str(e)

            is_transient = any(
                marker.lower() in error_str.lower()
                for marker in TRANSIENT_ERROR_MARKERS
            )

            if is_transient and attempt < max_retries - 1:

                wait = base_delay * (2 ** attempt)

                print(
                    f"Groq transient error "
                    f"(attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {wait:.1f}s: {error_str}"
                )

                time.sleep(wait)

                continue

            raise

    raise last_error


# ============================================================
# QDRANT
# ============================================================
# ملحوظة: الـEmbeddings الجديدة أبعادها مختلفة عن Gemini (1536)،
# فالـcollection القديمة مش هتشتغل. الاسم الافتراضي اتغير
# عشان تتعمل collection جديدة، وارفع المستندات تاني.
# ============================================================

QDRANT_URL = os.getenv(
    "QDRANT_URL",
    "http://127.0.0.1:6333"
)

COLLECTION_NAME = os.getenv(
    "QDRANT_COLLECTION",
    "zawolf_documents_groq"
)

# ------------------------------------------------------------
# SOLVED CASE MEMORY (الذاكرة طويلة المدى للحالات المحلولة)
# ------------------------------------------------------------

SOLVED_CASES_COLLECTION = os.getenv(
    "QDRANT_CASES_COLLECTION",
    "zawolf_solved_cases"
)

CASE_SAVE_ENABLED = os.getenv("CASE_SAVE_ENABLED", "1") == "1"

# لو في حالة محفوظة بنفس الدرجة دي أو أعلى، منحفظش نسخة تانية
CASE_DUPLICATE_THRESHOLD = float(
    os.getenv("CASE_DUPLICATE_THRESHOLD", "0.97")
)

# ------------------------------------------------------------
# CHAT MEMORY (المحادثات + رسايلها + ذاكرة المكالمة) في Qdrant
# ------------------------------------------------------------

CHATS_COLLECTION = os.getenv(
    "QDRANT_CHATS_COLLECTION",
    "cupii_chats"
)

qdrant = QdrantClient(
    url=QDRANT_URL,
    check_compatibility=False,
    timeout=30
)


# ============================================================
# CHUNK CONFIGURATION
# ============================================================

DEFAULT_MAX_CHARS = int(
    os.getenv("CHUNK_MAX_CHARS", "1200")
)

DEFAULT_MIN_CHARS = int(
    os.getenv("CHUNK_MIN_CHARS", "80")
)

MAX_CONTEXT_CHARS = int(
    os.getenv("CHUNK_CONTEXT_CHARS", "500")
)


# ============================================================
# LOCAL EMBEDDINGS (بديل Gemini Embeddings)
# ============================================================
# multilingual-e5-large: بيدعم العربي والإنجليزي كويس، أبعاده 1024.
# بيتحمل أول مرة بس (بيتنزل ~2GB) وبعدها بيشتغل محلي بدون API.
# بيتحدد حجم الـvector أوتوماتيك من الموديل.
# ============================================================

EMBEDDING_MODEL = os.getenv(
    "EMBEDDING_MODEL",
    "intfloat/multilingual-e5-large"
)

# فاضي = يختار GPU لو موجود وإلا CPU. أو اكتب "cpu" / "cuda"
EMBEDDING_DEVICE = os.getenv("EMBEDDING_DEVICE", "") or None

EMBEDDING_BATCH_SIZE = int(
    os.getenv("EMBEDDING_BATCH_SIZE", "16")
)

_embed_model = None
_embed_load_lock = threading.Lock()
_embed_encode_lock = threading.Lock()


def get_embedding_model():

    global _embed_model

    if _embed_model is None:

        with _embed_load_lock:

            if _embed_model is None:

                from sentence_transformers import SentenceTransformer

                print(
                    f"Loading embedding model: {EMBEDDING_MODEL}"
                )

                _embed_model = SentenceTransformer(
                    EMBEDDING_MODEL,
                    device=EMBEDDING_DEVICE
                )

                print("Embedding model loaded.")

    return _embed_model


def get_vector_size() -> int:

    return int(
        get_embedding_model()
        .get_sentence_embedding_dimension()
    )


def _embedding_prefix(task_type: str) -> str:
    """
    موديلات E5 محتاجة prefix عشان تفرق بين المستند والسؤال:
      passage: للنصوص المخزنة
      query:   للأسئلة
    """

    if "e5" in EMBEDDING_MODEL.lower():

        if task_type == "RETRIEVAL_QUERY":
            return "query: "

        return "passage: "

    return ""


def create_embeddings_batch(
    texts: List[str],
    task_type: str = "RETRIEVAL_DOCUMENT"
) -> List[List[float]]:

    if not texts:
        return []

    for t in texts:

        if not t or not t.strip():
            raise ValueError(
                "Cannot create embedding from empty text"
            )

    prefix = _embedding_prefix(task_type)

    model = get_embedding_model()

    with _embed_encode_lock:

        vectors = model.encode(
            [prefix + t for t in texts],
            batch_size=EMBEDDING_BATCH_SIZE,
            normalize_embeddings=True,
            show_progress_bar=False
        )

    return vectors.tolist()


def create_embedding(
    text: str,
    task_type: str = "RETRIEVAL_DOCUMENT"
) -> List[float]:

    return create_embeddings_batch(
        [text],
        task_type=task_type
    )[0]


# ============================================================
# QDRANT INITIALIZATION
# ============================================================

def init_qdrant():

    vector_size = get_vector_size()

    # collection الحالات المحلولة (لازم الأول لأن الدالة بتعمل return بدري)
    init_cases_collection(vector_size)

    # collection المحادثات (لازم الأول برضه لنفس السبب)
    init_chats_collection()

    if not qdrant.collection_exists(
        collection_name=COLLECTION_NAME
    ):
        qdrant.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(
                size=vector_size,
                distance=Distance.COSINE
            )
        )

        return

    # الـcollection موجودة: اتأكد إن أبعادها مطابقة للموديل الحالي
    try:

        info = qdrant.get_collection(
            collection_name=COLLECTION_NAME
        )

        existing_size = getattr(
            info.config.params.vectors,
            "size",
            None
        )

    except Exception:

        existing_size = None

    if existing_size and existing_size != vector_size:

        raise RuntimeError(
            f"Collection '{COLLECTION_NAME}' أبعادها {existing_size} "
            f"لكن موديل الـembedding الحالي أبعاده {vector_size}. "
            f"غيّر QDRANT_COLLECTION لاسم جديد أو امسح الـcollection "
            f"القديمة وارفع المستندات تاني."
        )


# ============================================================
# SOLVED CASE MEMORY
# ============================================================
# ملحوظة مهمة: بنعمل embedding للسؤال بـ task_type="RETRIEVAL_QUERY"
# في الحفظ والبحث الاتنين، لأن e5 بيقارن سؤال بسؤال، فالـ prefix
# لازم يبقى "query:" في الناحيتين (مش "passage:").
# ============================================================

def init_cases_collection(vector_size: int):

    if qdrant.collection_exists(
        collection_name=SOLVED_CASES_COLLECTION
    ):
        return

    qdrant.create_collection(
        collection_name=SOLVED_CASES_COLLECTION,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE
        )
    )


def search_solved_cases(question: str, limit: int = 1):

    try:

        vector = create_embedding(
            question,
            task_type="RETRIEVAL_QUERY"
        )

        return qdrant.query_points(
            collection_name=SOLVED_CASES_COLLECTION,
            query=vector,
            limit=limit,
            with_payload=True
        ).points

    except Exception as e:

        print(f"Case search failed: {e}")

        return []


def save_solved_case(
    problem: str,
    solution: str,
    sources: list,
    intent: str
):

    if not CASE_SAVE_ENABLED:
        return

    try:

        vector = create_embedding(
            problem,
            task_type="RETRIEVAL_QUERY"
        )

        # منكررش نفس الحالة
        existing = qdrant.query_points(
            collection_name=SOLVED_CASES_COLLECTION,
            query=vector,
            limit=1,
            with_payload=False
        ).points

        if (
            existing
            and existing[0].score >= CASE_DUPLICATE_THRESHOLD
        ):
            return

        payload = {
            "problem": problem,
            "solution": solution,
            "error_code": extract_error_code(problem),
            "intent": intent,
            "sources": sources,
            "created_at": int(time.time()),
            "hits": 0,
        }

        qdrant.upsert(
            collection_name=SOLVED_CASES_COLLECTION,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload=payload
                )
            ]
        )

        print(f"[case] saved: {problem[:60]}")

    except Exception as e:

        print(f"Case save failed: {e}")


# ============================================================
# CHAT MEMORY IN QDRANT (المحادثات + الرسايل + ذاكرة المكالمة)
# ============================================================
# collection واحدة (CHATS_COLLECTION) فيها نوعين من الـpoints:
#
#   kind="chat"     : معلومات المحادثة (العنوان، وقت الإنشاء والتحديث).
#                     الـid بتاعها ثابت (uuid5 من chat_id) فالـupsert بيحدّثها.
#   kind="message"  : رسالة واحدة (role, content, ts).
#
# الاتنين بيشتركوا في payload.chat_id، فمسح المحادثة بيمسح الاتنين مرة واحدة.
#
# مفيش بحث دلالي هنا، بنقرا بالفلتر والترتيب بس. فالـvector مجرد
# قيمة شكلية (حجمها 1) لأن Qdrant بيطلب vector لكل point.
# مفيش حاجة بتتعمل لها embedding، فالحفظ سريع جدًا.
#
# ذاكرة المكالمة الصوتية بتتخزن هنا برضه تحت chat_id = "voice:<session>"
# من غير point من نوع "chat"، فمبتظهرش في قايمة المحادثات.
# ============================================================

_CHAT_DUMMY_VECTOR = [1.0]

_chats_ready = False

_chats_init_lock = threading.Lock()


def _chat_point_id(chat_id: str) -> str:

    return str(
        uuid.uuid5(
            uuid.NAMESPACE_URL,
            f"cupii-chat:{chat_id}"
        )
    )


def init_chats_collection():

    global _chats_ready

    if _chats_ready:
        return

    with _chats_init_lock:

        if _chats_ready:
            return

        if not qdrant.collection_exists(
            collection_name=CHATS_COLLECTION
        ):

            qdrant.create_collection(
                collection_name=CHATS_COLLECTION,
                vectors_config=VectorParams(
                    size=1,
                    distance=Distance.COSINE
                )
            )

        for field in ("kind", "chat_id"):

            try:

                qdrant.create_payload_index(
                    collection_name=CHATS_COLLECTION,
                    field_name=field,
                    field_schema=PayloadSchemaType.KEYWORD
                )

            except Exception:
                # الـindex موجود قبل كده
                pass

        _chats_ready = True


def _chat_filter(
    chat_id: Optional[str] = None,
    kind: Optional[str] = None
) -> Filter:

    must = []

    if kind:
        must.append(
            FieldCondition(
                key="kind",
                match=MatchValue(value=kind)
            )
        )

    if chat_id:
        must.append(
            FieldCondition(
                key="chat_id",
                match=MatchValue(value=chat_id)
            )
        )

    return Filter(must=must)


def _scroll_all(flt: Filter):

    init_chats_collection()

    points = []

    offset = None

    while True:

        batch, offset = qdrant.scroll(
            collection_name=CHATS_COLLECTION,
            scroll_filter=flt,
            limit=256,
            offset=offset,
            with_payload=True,
            with_vectors=False
        )

        points.extend(batch)

        if offset is None:
            break

    return points


def _clean_chat_id(chat_id: str) -> str:

    chat_id = (chat_id or "").strip()

    if not chat_id or len(chat_id) > 120:
        raise ValueError("invalid chat id")

    return chat_id


def _get_chat_payload(chat_id: str) -> Optional[dict]:

    init_chats_collection()

    found = qdrant.retrieve(
        collection_name=CHATS_COLLECTION,
        ids=[_chat_point_id(chat_id)],
        with_payload=True,
        with_vectors=False
    )

    if not found:
        return None

    return found[0].payload or {}


def _upsert_chat_point(
    chat_id: str,
    title: Optional[str] = None,
    created_at: Optional[int] = None,
    updated_at: Optional[int] = None
):
    """ينشئ أو يحدّث point المحادثة (العنوان والأوقات)."""

    existing = _get_chat_payload(chat_id) or {}

    now = int(time.time() * 1000)

    payload = {
        "kind": "chat",
        "chat_id": chat_id,
        "title": (
            title
            or existing.get("title")
            or "محادثة جديدة"
        ),
        "created_at": (
            existing.get("created_at")
            or created_at
            or now
        ),
        "updated_at": updated_at or now,
    }

    qdrant.upsert(
        collection_name=CHATS_COLLECTION,
        points=[
            PointStruct(
                id=_chat_point_id(chat_id),
                vector=_CHAT_DUMMY_VECTOR,
                payload=payload
            )
        ]
    )

    return payload


def _message_point(
    chat_id: str,
    role: str,
    content: str,
    ts: int
) -> PointStruct:

    return PointStruct(
        id=str(uuid.uuid4()),
        vector=_CHAT_DUMMY_VECTOR,
        payload={
            "kind": "message",
            "chat_id": chat_id,
            "role": role,
            "content": content,
            "ts": ts,
        }
    )


def chat_list() -> List[dict]:
    """كل المحادثات (من الأحدث للأقدم)، من غير الرسايل."""

    points = _scroll_all(_chat_filter(kind="chat"))

    chats = []

    for p in points:

        payload = p.payload or {}

        chats.append({
            "id": payload.get("chat_id"),
            "title": payload.get("title") or "محادثة جديدة",
            "created_at": payload.get("created_at") or 0,
            "updated_at": payload.get("updated_at") or 0,
        })

    chats.sort(key=lambda c: c["updated_at"], reverse=True)

    return chats


def chat_get_messages(
    chat_id: str,
    limit: Optional[int] = None
) -> List[dict]:
    """رسايل محادثة مرتبة بالوقت. limit = آخر N رسالة."""

    points = _scroll_all(
        _chat_filter(chat_id=chat_id, kind="message")
    )

    messages = [
        {
            "role": (p.payload or {}).get("role", "user"),
            "content": (p.payload or {}).get("content", ""),
            "ts": (p.payload or {}).get("ts", 0),
        }
        for p in points
    ]

    messages.sort(key=lambda m: m["ts"])

    if limit:
        messages = messages[-limit:]

    return messages


def chat_add_message(
    chat_id: str,
    role: str,
    content: str,
    ts: Optional[int] = None,
    title: Optional[str] = None
):
    """يحفظ رسالة ويحدّث المحادثة (ينشئها لو مش موجودة)."""

    chat_id = _clean_chat_id(chat_id)

    if role not in ("user", "assistant"):
        raise ValueError("role must be user or assistant")

    content = (content or "").strip()

    if not content:
        raise ValueError("empty message")

    init_chats_collection()

    now = int(time.time() * 1000)

    ts = int(ts or now)

    _upsert_chat_point(chat_id, title=title, updated_at=now)

    qdrant.upsert(
        collection_name=CHATS_COLLECTION,
        points=[_message_point(chat_id, role, content, ts)]
    )

    return {"role": role, "content": content, "ts": ts}


def chat_set_title(chat_id: str, title: str):

    chat_id = _clean_chat_id(chat_id)

    title = (title or "").strip()[:120]

    if not title:
        raise ValueError("empty title")

    existing = _get_chat_payload(chat_id)

    if existing is None:
        raise KeyError(chat_id)

    qdrant.set_payload(
        collection_name=CHATS_COLLECTION,
        payload={"title": title},
        points=[_chat_point_id(chat_id)]
    )


def chat_delete(chat_id: str):
    """يمسح المحادثة ورسايلها."""

    chat_id = _clean_chat_id(chat_id)

    init_chats_collection()

    qdrant.delete(
        collection_name=CHATS_COLLECTION,
        points_selector=FilterSelector(
            filter=_chat_filter(chat_id=chat_id)
        )
    )


def chat_import(chats: List[dict]) -> int:
    """
    استيراد محادثات قديمة (من localStorage مثلًا).
    المحادثة الموجودة بالفعل على السيرفر بتتخطى.
    """

    init_chats_collection()

    imported = 0

    for chat in chats:

        try:

            chat_id = _clean_chat_id(str(chat.get("id", "")))

            if _get_chat_payload(chat_id) is not None:
                continue

            raw_messages = chat.get("messages") or []

            points = []

            last_ts = 0

            for index, m in enumerate(raw_messages):

                role = m.get("role")

                content = str(m.get("content") or "").strip()

                if role not in ("user", "assistant") or not content:
                    continue

                ts = int(
                    m.get("ts")
                    or (int(time.time() * 1000) + index)
                )

                last_ts = max(last_ts, ts)

                points.append(
                    _message_point(chat_id, role, content, ts)
                )

            if not points:
                continue

            _upsert_chat_point(
                chat_id,
                title=str(chat.get("title") or "")[:120] or None,
                created_at=int(chat.get("createdAt") or last_ts or 0) or None,
                updated_at=int(chat.get("updatedAt") or last_ts or 0) or None
            )

            qdrant.upsert(
                collection_name=CHATS_COLLECTION,
                points=points
            )

            imported += 1

        except Exception as e:

            print(f"Chat import skipped one chat: {e}")

    return imported


# ------------------------------------------------------------
# REST API للمحادثات (اربطه في main.py بسطر واحد):
#     from rag import chat_router
#     app.include_router(chat_router)
# ------------------------------------------------------------

chat_router = APIRouter(prefix="/api", tags=["chats"])


class _ChatCreateIn(BaseModel):
    id: Optional[str] = None
    title: Optional[str] = None


class _MessageIn(BaseModel):
    role: str
    content: str
    ts: Optional[int] = None
    title: Optional[str] = None


class _TitleIn(BaseModel):
    title: str


class _ImportIn(BaseModel):
    chats: List[Dict[str, Any]]


@chat_router.get("/chats")
def api_list_chats():

    try:
        return {"chats": chat_list()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.post("/chats")
def api_create_chat(body: _ChatCreateIn):

    try:

        chat_id = _clean_chat_id(body.id or str(uuid.uuid4()))

        init_chats_collection()

        payload = _upsert_chat_point(chat_id, title=body.title)

        return {
            "id": chat_id,
            "title": payload["title"],
            "updated_at": payload["updated_at"],
        }

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.post("/chats/import")
def api_import_chats(body: _ImportIn):

    try:
        return {"imported": chat_import(body.chats)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.get("/chats/{chat_id}/messages")
def api_get_messages(chat_id: str):

    try:
        return {"messages": chat_get_messages(chat_id)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.post("/chats/{chat_id}/messages")
def api_add_message(chat_id: str, body: _MessageIn):

    try:

        message = chat_add_message(
            chat_id,
            body.role,
            body.content,
            ts=body.ts,
            title=body.title
        )

        return {"message": message}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.put("/chats/{chat_id}")
def api_rename_chat(chat_id: str, body: _TitleIn):

    try:

        chat_set_title(chat_id, body.title)

        return {"ok": True}

    except KeyError:
        raise HTTPException(status_code=404, detail="chat not found")

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@chat_router.delete("/chats/{chat_id}")
def api_delete_chat(chat_id: str):

    try:

        chat_delete(chat_id)

        return {"ok": True}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================
# BASIC CHUNKING FALLBACK
# ============================================================

def chunk_text(
    text: str,
    chunk_size: int = 1000,
    overlap: int = 150
):

    text = text.strip()

    if not text:
        return []

    if overlap >= chunk_size:
        overlap = int(chunk_size * 0.15)

    chunks = []

    start = 0

    while start < len(text):

        end = start + chunk_size

        chunk = text[start:end]

        if chunk.strip():
            chunks.append(
                chunk.strip()
            )

        start += chunk_size - overlap

    return chunks


# ============================================================
# REGEX PATTERNS
# ============================================================

ARABIC_SECTION_RE = re.compile(
    r"^\s*(?:الفصل|الباب|القسم|الجزء|فصل|باب)"
    r"\s+\S.*$",
    re.IGNORECASE
)

CAPS_HEADING_RE = re.compile(
    r"^\s*[A-Z][A-Za-z0-9 \-_/&]{3,70}\s*$"
)

NUMBERED_HEADING_RE = re.compile(
    r"^\s*(\d{1,2}(?:\.\d{1,2}){0,3})"
    r"[.\)]?\s+(\S.{0,80})\s*$"
)

SHORT_LABEL_RE = re.compile(
    r"^\s*\S.{0,58}[:：]\s*$"
)

ERROR_CODE_RE = re.compile(
    r"(?:كود الخطأ|رمز الخطأ|Error Code|Fault Code|"
    r"Alarm|Error|Err)"
    r"\s*[:#\-]?\s*"
    r"([A-Za-z]{0,6}-?\d{1,6})",
    re.IGNORECASE
)

ERROR_CODE_LOOSE_RE = re.compile(
    r"\b([A-Z]{1,5}-?\d{2,6})\b"
)

CAUSE_RE = re.compile(
    r"^\s*(?:السبب|الأسباب|Cause|Causes)\s*[:：]",
    re.IGNORECASE
)

SOLUTION_RE = re.compile(
    r"^\s*(?:الحل|الحلول|الإصلاح|طريقة الإصلاح|"
    r"Solution|Fix|Remedy|Corrective Action)"
    r"\s*[:：]",
    re.IGNORECASE
)

DIAGNOSIS_RE = re.compile(
    r"^\s*(?:التشخيص|Diagnosis|Troubleshooting)"
    r"\s*[:：]",
    re.IGNORECASE
)

SPEC_RE = re.compile(
    r"^\s*(?:المواصفات|Specifications?|Pressure|"
    r"Temperature|Voltage|Current|Torque|Dimensions)"
    r"\s*[:：]",
    re.IGNORECASE
)

PROCEDURE_RE = re.compile(
    r"^\s*(?:خطوات|إجراء|طريقة الاستبدال|"
    r"Procedure|Replacement Procedure|Steps?)"
    r"\s*[:：]",
    re.IGNORECASE
)

SYMPTOM_RE = re.compile(
    r"^\s*(?:العَرَض|الأعراض|العرض|Symptoms?|"
    r"Symptom|Observation|ملاحظة|المشكلة)"
    r"\s*[:：]",
    re.IGNORECASE
)

COMPONENT_RE = re.compile(
    r"^\s*(?:المكون|المكونات|Component|Components?|"
    r"Part|Parts|الجزء)"
    r"\s*[:：]",
    re.IGNORECASE
)

SAFETY_RE = re.compile(
    r"^\s*(?:تحذير|تحذيرات|تنبيه|السلامة|"
    r"Warning|Warnings|Caution|Safety)"
    r"\s*[:：]",
    re.IGNORECASE
)

TABLE_ROW_RE = re.compile(
    r"(\|.*\|)"
    r"|(\t\S+\t)"
    r"|( {3,}\S+ {3,}\S+)"
)


# ============================================================
# ERROR CODE EXTRACTION
# ============================================================

def extract_error_code(
    text: str
) -> Optional[str]:

    if not text:
        return None

    match = ERROR_CODE_RE.search(text)

    if match:
        code = match.group(1).strip("-").upper()

        if code and any(
            char.isdigit()
            for char in code
        ):
            return code

    matches = ERROR_CODE_LOOSE_RE.findall(
        text
    )

    for code in matches:

        code = code.upper()

        if any(
            char.isdigit()
            for char in code
        ):
            return code

    return None


# ============================================================
# LINE CLASSIFICATION
# ============================================================

def classify_line(
    line: str
):

    stripped = line.strip()

    if not stripped:
        return "blank", None

    # Tables first
    if TABLE_ROW_RE.search(stripped):
        return "table", None

    # Technical semantic labels
    if SAFETY_RE.match(stripped):
        return "safety", None

    if SYMPTOM_RE.match(stripped):
        return "symptom", None

    if COMPONENT_RE.match(stripped):
        return "component", None

    if CAUSE_RE.match(stripped):
        return "cause", None

    if SOLUTION_RE.match(stripped):
        return "solution", None

    if DIAGNOSIS_RE.match(stripped):
        return "diagnosis", None

    if PROCEDURE_RE.match(stripped):
        return "procedure", None

    if SPEC_RE.match(stripped) and len(stripped) <= 90:
        return "specification", None

    # Error code
    if (
        ERROR_CODE_RE.search(stripped)
        or ERROR_CODE_LOOSE_RE.search(stripped)
    ):
        return "error_code", None

    # Arabic section
    if (
        len(stripped) <= 90
        and ARABIC_SECTION_RE.match(stripped)
    ):
        return "heading", "section"

    # Numbered hierarchy
    numbered_match = NUMBERED_HEADING_RE.match(
        stripped
    )

    if numbered_match and len(stripped) <= 90:

        depth = (
            numbered_match.group(1).count(".")
            + 1
        )

        if depth == 1:
            return "heading", "section"

        elif depth == 2:
            return "heading", "subsection"

        else:
            return "heading", "topic"

    # English uppercase heading
    if (
        len(stripped) <= 70
        and CAPS_HEADING_RE.match(stripped)
    ):
        return "heading", "section"

    # Short technical label
    if (
        len(stripped) <= 60
        and SHORT_LABEL_RE.match(stripped)
    ):
        return "heading", "topic"

    return "text", None


# ============================================================
# TABLE HEADER
# ============================================================

def table_header_lines(
    lines
):

    header_lines = []

    for line in lines:

        if not line.strip():
            continue

        header_lines.append(
            line
        )

        if TABLE_ROW_RE.search(line):
            break

        if len(header_lines) >= 3:
            break

    return header_lines


# ============================================================
# BUILD HIERARCHICAL CONTEXT
# ============================================================

def build_parent_context(
    machine=None,
    section=None,
    subsection=None,
    topic=None,
    error_code=None,
    content_type=None
):

    parts = []

    if machine:
        parts.append(f"Machine: {machine}")

    if section:
        parts.append(f"Section: {section}")

    if subsection:
        parts.append(f"Subsection: {subsection}")

    if topic:
        parts.append(f"Topic: {topic}")

    if error_code:
        parts.append(f"Error Code: {error_code}")

    if content_type:
        parts.append(f"Content Type: {content_type}")

    return "\n".join(parts)


# ============================================================
# SPLIT OVERSIZED BLOCK
# ============================================================

def split_oversized_block(
    block_text,
    block_types,
    max_chars,
    min_chars,
    parent_context=None
):

    lines = block_text.split("\n")

    context_lines = []

    # Preserve table header
    if "table" in block_types:

        context_lines = table_header_lines(
            lines
        )

    # Preserve error code
    elif block_types & {
        "error_code",
        "cause",
        "solution",
        "diagnosis",
        "symptom",
        "procedure"
    }:

        for line in lines[:6]:

            if (
                ERROR_CODE_RE.search(line)
                or ERROR_CODE_LOOSE_RE.search(line)
            ):
                context_lines = [line]
                break

    chunks = []

    current = []

    current_len = 0

    def get_context_length():

        if not context_lines:
            return 0

        return sum(
            len(line) + 1
            for line in context_lines
        )

    context_length = get_context_length()

    def flush():

        nonlocal current
        nonlocal current_len

        joined = "\n".join(
            current
        ).strip()

        if joined:
            chunks.append(
                joined
            )

        current = list(
            context_lines
        )

        current_len = context_length

    for line in lines:

        line_len = len(line) + 1

        if (
            current
            and current_len + line_len > max_chars
        ):

            flush()

        current.append(line)

        current_len += line_len

    joined = "\n".join(
        current
    ).strip()

    if joined:
        chunks.append(
            joined
        )

    # Merge tiny final chunk
    if (
        len(chunks) > 1
        and len(chunks[-1]) < min_chars
    ):

        chunks[-2] += (
            "\n" + chunks[-1]
        )

        chunks.pop()

    return chunks


# ============================================================
# HIERARCHICAL STRUCTURE-AWARE CHUNKING
# ============================================================

def chunk_text_structure_aware(
    text,
    max_chars=None,
    min_chars=None
):

    max_chars = (
        max_chars
        or DEFAULT_MAX_CHARS
    )

    min_chars = (
        min_chars
        or DEFAULT_MIN_CHARS
    )

    text = text.strip()

    if not text:
        return []

    lines = text.split("\n")

    raw_blocks = []

    # Current hierarchy
    current_section = None
    current_subsection = None
    current_topic = None
    current_error_code = None

    buffer_lines = []
    buffer_types = set()

    # --------------------------------------------------------
    # Flush buffer
    # --------------------------------------------------------

    def flush_buffer():

        nonlocal buffer_lines
        nonlocal buffer_types

        joined = "\n".join(
            buffer_lines
        ).strip()

        if joined:

            detected_error = extract_error_code(
                joined
            )

            raw_blocks.append({

                "text": joined,

                "types": set(
                    buffer_types
                ),

                "section":
                    current_section,

                "subsection":
                    current_subsection,

                "topic":
                    current_topic,

                "error_code":
                    detected_error
                    or current_error_code
            })

        buffer_lines = []
        buffer_types = set()

    # --------------------------------------------------------
    # Main parser
    # --------------------------------------------------------

    for raw_line in lines:

        line_type, heading_level = classify_line(
            raw_line
        )

        # Blank
        if line_type == "blank":

            buffer_lines.append(
                raw_line
            )

            continue

        # ----------------------------------------------------
        # Heading
        # ----------------------------------------------------

        if line_type == "heading":

            flush_buffer()

            clean_title = (
                raw_line
                .strip()
                .rstrip(":：")
                .strip()
            )

            if heading_level == "section":

                current_section = clean_title

                current_subsection = None
                current_topic = None
                current_error_code = None

            elif heading_level == "subsection":

                current_subsection = clean_title

                current_topic = None

            else:

                current_topic = clean_title

            buffer_lines.append(
                raw_line
            )

            buffer_types.add(
                "heading"
            )

            continue

        # ----------------------------------------------------
        # Error code
        # ----------------------------------------------------

        if line_type == "error_code":

            detected_code = extract_error_code(
                raw_line
            )

            if detected_code:

                # If a new error appears,
                # separate the previous error block.
                if (
                    current_error_code
                    and detected_code
                    != current_error_code
                ):
                    flush_buffer()

                current_error_code = detected_code

            buffer_lines.append(
                raw_line
            )

            buffer_types.add(
                "error_code"
            )

            continue

        # ----------------------------------------------------
        # Tables
        # ----------------------------------------------------

        if line_type == "table":

            if (
                buffer_types
                and "table"
                not in buffer_types
            ):

                flush_buffer()

            buffer_lines.append(
                raw_line
            )

            buffer_types.add(
                "table"
            )

            continue

        # ----------------------------------------------------
        # Technical semantic blocks
        # ----------------------------------------------------

        if line_type in (
            "cause",
            "solution",
            "diagnosis",
            "procedure",
            "specification",
            "symptom",
            "component",
            "safety"
        ):

            # If moving from one major technical
            # block to another, preserve separation.
            if (
                line_type
                in (
                    "cause",
                    "solution",
                    "diagnosis",
                    "procedure"
                )
                and buffer_types
                and any(
                    x in buffer_types
                    for x in (
                        "cause",
                        "solution",
                        "diagnosis",
                        "procedure"
                    )
                )
                and line_type
                not in buffer_types
            ):

                flush_buffer()

            buffer_lines.append(
                raw_line
            )

            buffer_types.add(
                line_type
            )

            continue

        # ----------------------------------------------------
        # Text after table
        # ----------------------------------------------------

        if "table" in buffer_types:

            flush_buffer()

        buffer_lines.append(
            raw_line
        )

        buffer_types.add(
            "text"
        )

        # Normal text size boundary
        if (
            buffer_types == {"text"}
            and len(
                "\n".join(
                    buffer_lines
                )
            ) >= max_chars
        ):

            flush_buffer()

    flush_buffer()

    # ========================================================
    # MERGE SMALL BLOCKS
    # ========================================================

    merged_blocks = []

    for block in raw_blocks:

        previous = (
            merged_blocks[-1]
            if merged_blocks
            else None
        )

        can_merge = (

            previous is not None

            and len(
                block["text"]
            ) < min_chars

            and previous["section"]
            == block["section"]

            and previous["subsection"]
            == block["subsection"]

            and previous["topic"]
            == block["topic"]

            and previous["error_code"]
            == block["error_code"]

            and "table"
            not in block["types"]

            and "table"
            not in previous["types"]

            and (
                len(
                    previous["text"]
                )
                + len(
                    block["text"]
                )
                <= max_chars * 1.5
            )
        )

        if can_merge:

            previous["text"] += (
                "\n"
                + block["text"]
            )

            previous["types"] |= (
                block["types"]
            )

        else:

            merged_blocks.append(
                block
            )

    # ========================================================
    # FINAL CHUNKS
    # ========================================================

    final_chunks = []

    for block in merged_blocks:

        block_text = (
            block["text"].strip()
        )

        if not block_text:
            continue

        error_code = (
            block.get("error_code")
            or extract_error_code(
                block_text
            )
        )

        block_types = block["types"]

        # ----------------------------------------------------
        # Content type
        # ----------------------------------------------------

        if "table" in block_types:

            content_type = "table"

        elif "safety" in block_types:

            content_type = "safety"

        elif (
            "cause" in block_types
            or "solution" in block_types
        ):

            content_type = "cause_solution"

        elif "diagnosis" in block_types:

            content_type = "diagnosis"

        elif "procedure" in block_types:

            content_type = "procedure"

        elif "specification" in block_types:

            content_type = "specification"

        elif "symptom" in block_types:

            content_type = "symptom"

        elif "component" in block_types:

            content_type = "component"

        elif error_code:

            content_type = "error_code"

        elif (
            "heading" in block_types
            and len(block_types) == 1
        ):

            content_type = "heading"

        else:

            content_type = "text"

        # ----------------------------------------------------
        # Keep chunk
        # ----------------------------------------------------

        if len(block_text) <= max_chars:

            final_chunks.append({

                "text": block_text,

                "section": block["section"],

                "subsection": block["subsection"],

                "topic": block["topic"],

                "error_code": error_code,

                "content_type": content_type
            })

            continue

        # ----------------------------------------------------
        # Split oversized
        # ----------------------------------------------------

        sub_texts = split_oversized_block(
            block_text,
            block_types,
            max_chars,
            min_chars
        )

        for sub in sub_texts:

            final_chunks.append({

                "text": sub,

                "section": block["section"],

                "subsection": block["subsection"],

                "topic": block["topic"],

                "error_code": (
                    error_code
                    or extract_error_code(sub)
                ),

                "content_type": content_type
            })

    return final_chunks


# ============================================================
# ADD STRUCTURED DOCUMENT
# ============================================================

def add_document_structured(
    text,
    filename,
    page=None,
    machine=None
):

    chunks = chunk_text_structure_aware(
        text
    )

    if not chunks:
        return 0

    prepared = []

    embedding_texts = []

    for index, chunk in enumerate(chunks):

        error_code = chunk.get("error_code")

        content_type = chunk.get(
            "content_type",
            "text"
        )

        section = chunk.get("section")

        subsection = chunk.get("subsection")

        topic = chunk.get("topic")

        # ====================================================
        # HIERARCHICAL CONTEXT
        # ====================================================

        parent_context = build_parent_context(
            machine=machine,
            section=section,
            subsection=subsection,
            topic=topic,
            error_code=error_code,
            content_type=content_type
        )

        # ====================================================
        # ENRICHED EMBEDDING TEXT
        # ====================================================

        embedding_texts.append(
            parent_context
            + "\n\n"
            + chunk["text"]
        )

        # ====================================================
        # PAYLOAD
        # ====================================================

        payload = {

            "text": chunk["text"],

            "filename": filename,

            "chunk": index,

            "content_type": content_type,

            "parent_context": parent_context
        }

        if machine:
            payload["machine"] = machine

        if section:
            payload["section"] = section

        if subsection:
            payload["subsection"] = subsection

        if topic:
            payload["topic"] = topic

        if error_code:
            payload["error_code"] = error_code

        if page is not None:
            payload["page"] = page

        prepared.append(payload)

    # ========================================================
    # EMBED (batch واحد لكل الصفحة)
    # ========================================================

    embeddings = create_embeddings_batch(
        embedding_texts,
        task_type="RETRIEVAL_DOCUMENT"
    )

    points = [

        PointStruct(

            id=str(uuid.uuid4()),

            vector=embedding,

            payload=payload
        )

        for embedding, payload
        in zip(embeddings, prepared)
    ]

    # ========================================================
    # UPSERT
    # ========================================================

    if points:

        qdrant.upsert(

            collection_name=COLLECTION_NAME,

            points=points
        )

    return len(points)


# ============================================================
# EXACT ERROR CODE SEARCH
# ============================================================

def search_by_error_code(
    error_code: str,
    limit: int = 5
):

    try:

        normalized_code = (
            error_code
            .strip()
            .upper()
        )

        results = qdrant.scroll(

            collection_name=COLLECTION_NAME,

            scroll_filter=Filter(

                must=[

                    FieldCondition(

                        key="error_code",

                        match=MatchValue(
                            value=normalized_code
                        )
                    )
                ]
            ),

            limit=limit,

            with_payload=True
        )

        return results[0]

    except Exception:

        return []


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(
    question: str,
    limit: int = 10
):

    query_vector = create_embedding(
        question,
        task_type="RETRIEVAL_QUERY"
    )

    results = (
        qdrant.query_points(

            collection_name=COLLECTION_NAME,

            query=query_vector,

            limit=limit,

            with_payload=True
        )
        .points
    )

    return results


# ============================================================
# KEYWORD EXTRACTION
# ============================================================

def extract_keywords(
    text: str
):

    if not text:
        return set()

    words = re.findall(
        r"[\w\u0600-\u06FF]+",
        text.lower()
    )

    # Common Arabic stop words
    stop_words = {
        "ما",
        "هو",
        "هي",
        "هل",
        "كيف",
        "ماذا",
        "لماذا",
        "في",
        "من",
        "على",
        "الى",
        "إلى",
        "عن",
        "مع",
        "هذا",
        "هذه",
        "ذلك",
        "تلك",
        "عند",
        "عندما",
        "تم",
        "يتم",
        "أي",
        "او",
        "أو",
        "و",
        "ف",
        "ثم",
        "the",
        "is",
        "are",
        "what",
        "how",
        "why"
    }

    return {
        word
        for word in words
        if len(word) > 1
        and word not in stop_words
    }


# ============================================================
# RERANKING
# ============================================================

def rerank_results(
    question: str,
    results,
    limit: int = 5
):

    detected_code = extract_error_code(
        question
    )

    question_keywords = extract_keywords(
        question
    )

    scored = []

    for result in results:

        payload = result.payload or {}

        # Record من qdrant.scroll مالوش .score
        base_score = float(
            getattr(result, "score", 0)
            or 0
        )

        text = str(
            payload.get("text", "")
        ).lower()

        error_code = str(
            payload.get("error_code", "")
        ).lower()

        content_type = payload.get(
            "content_type",
            ""
        )

        section = str(
            payload.get("section", "")
        ).lower()

        subsection = str(
            payload.get("subsection", "")
        ).lower()

        topic = str(
            payload.get("topic", "")
        ).lower()

        # ====================================================
        # START SCORE
        # ====================================================

        score = base_score

        # ====================================================
        # EXACT ERROR CODE
        # ====================================================

        if (
            detected_code
            and error_code
            == detected_code.lower()
        ):

            score += 1.5

        # ====================================================
        # ERROR CODE PRESENT IN TEXT
        # ====================================================

        elif (
            detected_code
            and detected_code.lower()
            in text
        ):

            score += 0.8

        # ====================================================
        # KEYWORD OVERLAP
        # ====================================================

        document_keywords = extract_keywords(
            text
            + " "
            + section
            + " "
            + subsection
            + " "
            + topic
        )

        if question_keywords:

            overlap = (
                len(
                    question_keywords
                    & document_keywords
                )
                / len(question_keywords)
            )

            score += overlap * 0.35

        # ====================================================
        # METADATA MATCH
        # ====================================================

        metadata_text = (
            section
            + " "
            + subsection
            + " "
            + topic
        )

        metadata_keywords = extract_keywords(
            metadata_text
        )

        if question_keywords:

            metadata_overlap = (
                len(
                    question_keywords
                    & metadata_keywords
                )
                / len(question_keywords)
            )

            score += metadata_overlap * 0.20

        # ====================================================
        # INDUSTRIAL CONTENT BOOST
        # ====================================================

        industrial_types = {

            "cause_solution": 0.18,

            "diagnosis": 0.18,

            "procedure": 0.16,

            "error_code": 0.20,

            "symptom": 0.14,

            "component": 0.10,

            "specification": 0.08,

            "table": 0.08,

            "safety": 0.10
        }

        score += industrial_types.get(
            content_type,
            0
        )

        scored.append(
            (score, result)
        )

    # ========================================================
    # SORT
    # ========================================================

    scored.sort(
        key=lambda item: item[0],
        reverse=True
    )

    return [
        result
        for score, result
        in scored[:limit]
    ]


# ============================================================
# SEARCH DOCUMENTS (استعلام واحد - بيُستخدم في مسار الصور)
# ============================================================

def search_documents(
    question: str,
    limit: int = 7
):

    detected_code = extract_error_code(
        question
    )

    candidates = []

    seen_ids = set()

    # 1. EXACT ERROR CODE
    if detected_code:

        exact_results = search_by_error_code(
            detected_code,
            limit=10
        )

        for result in exact_results:

            if result.id not in seen_ids:

                candidates.append(result)

                seen_ids.add(result.id)

    # 2. SEMANTIC SEARCH
    semantic_results = semantic_search(
        question,
        limit=SEMANTIC_CANDIDATES * 2
    )

    for result in semantic_results:

        if result.id not in seen_ids:

            candidates.append(result)

            seen_ids.add(result.id)

    # 3. RERANK
    final_results = rerank_results(
        question,
        candidates,
        limit=limit
    )

    return final_results


# ============================================================
# BUILD RAG CONTEXT
# ============================================================

def build_rag_context(
    results
):

    context_parts = []

    sources = []

    for result in results:

        payload = result.payload or {}

        text = payload.get("text", "")

        filename = payload.get("filename", "unknown")

        page = payload.get("page")

        machine = payload.get("machine")

        section = payload.get("section")

        subsection = payload.get("subsection")

        topic = payload.get("topic")

        error_code = payload.get("error_code")

        content_type = payload.get("content_type")

        # ====================================================
        # INTERNAL CONTEXT
        # ====================================================

        # These are INTERNAL ONLY.
        # They help the LLM understand context.
        # They can be hidden from the final answer.
        #
        # ملحوظة: شلنا "السياق الهيكلي" (parent_context) لأنه
        # تكرار لنفس الحقول اللي تحت، وكان بيكبّر الـprompt
        # (وبالتالي بيبطّأ الرد ويقرّب من حد الـTPM) من غير فايدة.

        metadata = []

        if filename:
            metadata.append(
                f"المصدر الداخلي: {filename}"
            )

        if page:
            metadata.append(
                f"الصفحة الداخلية: {page}"
            )

        if machine:
            metadata.append(
                f"الماكينة: {machine}"
            )

        if section:
            metadata.append(
                f"القسم: {section}"
            )

        if subsection:
            metadata.append(
                f"الفرع: {subsection}"
            )

        if topic:
            metadata.append(
                f"الموضوع: {topic}"
            )

        if error_code:
            metadata.append(
                f"كود الخطأ: {error_code}"
            )

        if content_type:
            metadata.append(
                f"نوع المحتوى: {content_type}"
            )

        context_parts.append(

            "\n".join(metadata)

            + "\n\n"

            + "المحتوى:\n"

            + text
        )

        # ====================================================
        # SOURCES
        # ====================================================

        sources.append({

            "filename": filename,

            "page": page,

            "machine": machine,

            "section": section,

            "subsection": subsection,

            "topic": topic,

            "error_code": error_code,

            "content_type": content_type,

            "score": round(
                float(
                    getattr(result, "score", 0)
                    or 0
                ),
                4
            )
        })

    context = "\n\n---\n\n".join(
        context_parts
    )

    return context, sources


# ============================================================
# GROQ CHAT HELPERS
# ============================================================

def _chat(
    system_prompt: str,
    user_content,
    model: Optional[str] = None,
    temperature: float = 0.1,
    json_mode: bool = False,
    max_tokens: Optional[int] = None
) -> str:
    """
    نداء واحد لـ Groq Chat Completions.
    user_content: نص عادي، أو list (نص + صورة) للـvision.
    """

    model_name = model or CHAT_MODEL

    kwargs: Dict[str, Any] = {

        "model": model_name,

        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_content
            }
        ],

        "temperature": temperature,
    }

    if json_mode:
        kwargs["response_format"] = {
            "type": "json_object"
        }

    if max_tokens:
        kwargs["max_tokens"] = max_tokens

    # موديلات gpt-oss بتفكر قبل ما ترد. low = تفكير أقل = رد أسرع.
    # بنبعته في extra_body عشان يشتغل مع أي إصدار من مكتبة groq.
    if "gpt-oss" in model_name.lower() and REASONING_EFFORT:
        kwargs["extra_body"] = {
            "reasoning_effort": REASONING_EFFORT
        }

    response = _call_with_retry(
        groq_client.chat.completions.create,
        **kwargs
    )

    return (
        response.choices[0].message.content
        or ""
    ).strip()


# Groq بيقبل صور base64 لحد ~4MB فقط
MAX_IMAGE_B64_BYTES = 4 * 1024 * 1024


def _prepare_image(image_bytes: bytes, mime_type: str):
    """
    لو الصورة كبيرة (صور الموبايل غالبًا كبيرة)، نصغّرها
    ونحولها JPEG عشان تعدي حد Groq. محتاج Pillow.
    """

    if len(image_bytes) * 4 / 3 <= MAX_IMAGE_B64_BYTES:
        return image_bytes, mime_type

    try:

        from PIL import Image

    except ImportError:

        raise ValueError(
            "الصورة كبيرة، ثبّت Pillow عشان يتم تصغيرها: "
            "pip install pillow"
        )

    img = Image.open(io.BytesIO(image_bytes))

    img = img.convert("RGB")

    size = 2000

    while True:

        copy = img.copy()

        copy.thumbnail((size, size))

        buffer = io.BytesIO()

        copy.save(buffer, format="JPEG", quality=85)

        data = buffer.getvalue()

        if (
            len(data) * 4 / 3 <= MAX_IMAGE_B64_BYTES
            or size <= 600
        ):
            return data, "image/jpeg"

        size = int(size * 0.75)


def _image_content(
    text: str,
    image_bytes: bytes,
    mime_type: str
):

    image_bytes, mime_type = _prepare_image(
        image_bytes,
        mime_type
    )

    b64 = base64.b64encode(
        image_bytes
    ).decode("utf-8")

    return [
        {
            "type": "text",
            "text": text
        },
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{b64}"
            }
        }
    ]


# ============================================================
# IMAGE ANALYSIS
# ============================================================

def analyze_image_with_context(
    image_bytes,
    mime_type
):

    system_prompt = """

أنت مهندس صيانة صناعية وخبير في تحليل صور
الماكينات والمعدات.

حلل الصورة بدقة وبدون اختلاق معلومات.

حدد قدر الإمكان:

1. الماكينة أو الجزء الظاهر.
2. المكونات الظاهرة.
3. أي عطل أو تلف أو تسريب أو كسر أو تشوه.
4. أي Error Code أو Alarm ظاهر.
5. أي Sensor أو جزء كهربائي ظاهر.
6. العلامات البصرية التي اعتمدت عليها.
7. وصف تقني يمكن استخدامه للبحث داخل كتالوجات الصيانة.

مهم جدًا:

- لا تخترع معلومات غير واضحة.
- إذا لم تكن متأكدًا استخدم "قد يكون".
- فرّق بين ما هو ظاهر فعلًا وما هو استنتاج.
- لا تؤكد وجود عطل من الصورة وحدها إذا لم توجد أدلة كافية.
- ركز على المصطلحات الصناعية والفنية.

في النهاية أعطني وصفًا تقنيًا مختصرًا
يمكن استخدامه كـ retrieval query داخل قاعدة المعرفة.

أجب باللغة العربية.
"""

    user_text = """
حلل الصورة وأعطني وصفًا تقنيًا
يمكن استخدامه للبحث داخل كتالوجات
صيانة الماكينات.
"""

    return _chat(
        system_prompt,
        _image_content(
            user_text,
            image_bytes,
            mime_type
        ),
        model=VISION_MODEL,
        temperature=0.1
    )


# ============================================================
# MULTIMODAL RAG
# ============================================================

def ask_rag_with_image(
    question: str,
    image_bytes,
    mime_type,
    limit: int = 7
):

    # ========================================================
    # STEP 1 - IMAGE ANALYSIS
    # ========================================================

    image_analysis = analyze_image_with_context(
        image_bytes,
        mime_type
    )

    # ========================================================
    # STEP 2 - RETRIEVAL QUERY
    # ========================================================

    retrieval_query = f"""

سؤال المستخدم:

{question}

تحليل الصورة:

{image_analysis}

ابحث عن المعلومات الفنية المرتبطة
بالمشكلة أو المكون أو Error Code أو
العطل المحتمل.
"""

    # ========================================================
    # STEP 3 - RETRIEVAL
    # ========================================================

    results = search_documents(
        retrieval_query,
        limit=limit
    )

    # ========================================================
    # STEP 4 - NO RESULTS
    # ========================================================

    if not results:

        return {

            "answer":
                "لم أجد معلومات مرتبطة "
                "بالصورة والسؤال داخل "
                "قاعدة المعرفة.",

            "image_analysis":
                image_analysis,

            "sources":
                []
        }

    # ========================================================
    # STEP 5 - BUILD CONTEXT
    # ========================================================

    context, sources = build_rag_context(
        results
    )

    # ========================================================
    # STEP 6 - FINAL ANSWER
    # ========================================================

    system_prompt = """

أنت مهندس صيانة صناعية ومساعد RAG متخصص
في كتالوجات الماكينات والمعدات.

لديك:

1. سؤال المستخدم.
2. تحليل الصورة.
3. معلومات مسترجعة من قاعدة المعرفة.

قواعد مهمة جدًا:

- استخدم قاعدة المعرفة كمرجع أساسي.
- لا تخترع معلومات غير موجودة في السياق.
- لا تعتبر تحليل الصورة حقيقة مؤكدة إذا كان مجرد احتمال.
- فرّق بوضوح بين "الملاحظ" و"التشخيص المحتمل".
- إذا كانت المعلومات غير كافية، قل ذلك بوضوح.
- لا تعطِ إجراءً خطيرًا أو غير مدعوم بالمعلومات.
- إذا كان هناك Error Code استخدمه كعامل رئيسي في التشخيص.

رتب الإجابة بهذا الشكل:

### الحالة الظاهرة
ما يظهر في الصورة.

### التشخيص المحتمل
التشخيص المدعوم بالمعلومات المتاحة.

### الأدلة الفنية
المعلومات الموجودة في قاعدة المعرفة
التي تدعم التشخيص.

### خطوات الفحص
خطوات الفحص الموجودة أو المدعومة
من الكتالوج.

### الحل أو الإجراء المقترح
الإجراء الفني المدعوم بالمعلومات.

مهم جدًا:

- لا تذكر اسم الملف.
- لا تذكر رقم الصفحة.
- لا تكتب قسم "المصدر ورقم الصفحة".
- المصادر تستخدم داخليًا للتحقق فقط.

أجب باللغة العربية.
"""

    user_prompt = f"""

سؤال المستخدم:

{question}

تحليل الصورة:

{image_analysis}

المعلومات المسترجعة من قاعدة المعرفة:

{context}

استخدم هذه المعلومات فقط عند إعطاء
التشخيص والإجراء الفني.
"""

    answer = _chat(
        system_prompt,
        user_prompt,
        temperature=0.1
    )

    return {

        "answer": answer,

        "image_analysis": image_analysis,

        "sources": sources
    }


# ============================================================
# SMART ASK_RAG
# ============================================================

HISTORY_HEADER = (
    "سياق المحادثة السابق "
    "(للفهم فقط، جاوب على السؤال الحالي بالأسفل):"
)

CURRENT_MARKER = "السؤال الحالي:"

CASUAL_PATTERNS = [
    r"^\s*السلام عليكم\s*$",
    r"^\s*السلام عليكم ورحمة الله\s*$",
    r"^\s*اهلا\s*$",
    r"^\s*أهلا\s*$",
    r"^\s*اهلا وسهلا\s*$",
    r"^\s*أهلاً وسهلاً\s*$",
    r"^\s*هاي\s*$",
    r"^\s*hello\s*$",
    r"^\s*hi\s*$",
    r"^\s*مرحبا\s*$",
    r"^\s*مراحب\s*$",
]

GREETING_ANSWER = (
    "أهلاً بيك 👋\n"
    "أنا مساعد CUPII للصيانة الصناعية. "
    "ابعتلي مشكلة الماكينة أو Error Code "
    "أو صورة للعطل وأنا أساعدك في التشخيص "
    "بالاعتماد على قاعدة المعرفة."
)

# نسخة بدون إيموجي للمكالمة الصوتية
GREETING_ANSWER_VOICE = (
    "أهلاً بيك. "
    "أنا مساعد CUPII للصيانة الصناعية. "
    "قولي مشكلة الماكينة أو كود الخطأ "
    "وأنا أساعدك."
)

VALID_INTENTS = {
    "chitchat",
    "about_assistant",
    "document_question",
    "general_question",
    "unclear",
}


# ============================================================
# PROMPTS
# ============================================================

UNDERSTAND_PROMPT = """
أنت وحدة "فهم الأسئلة" داخل مساعد صيانة صناعية اسمه CUPII،
وعنده قاعدة معرفة من كتالوجات ومستندات فنية.

مهمتك: تفهم رسالة المستخدم مهما كانت (عامية مصرية، ناقصة،
فيها أخطاء إملائية، بتشير لكلام سابق، أو مزيج عربي/إنجليزي)
وترجع JSON فقط بالحقول التالية:

- intent: واحدة بالظبط من:
  "chitchat"          تحية أو شكر أو كلام عادي
  "about_assistant"   سؤال عن المساعد نفسه (بتفهمني؟ بتعمل إيه؟ قدراتك؟ مين انت؟)
  "document_question" سؤال فني ممكن إجابته في الكتالوجات
                      (عطل، Error Code، مواصفات، صيانة، قطعة، إجراء)
  "general_question"  سؤال عام أو فني مش مرتبط بمستند معين
  "unclear"           مفهوش تفسير معقول حتى مع السياق

- standalone_question: السؤال بعد حل أي إشارة لكلام سابق
  ("هو"، "ده"، "والحل؟"، "طب ليه؟") بحيث يتفهم لوحده، بالعربية.

- search_queries: من 1 إلى 3 استعلامات بحث قصيرة ومختلفة
  (مصطلحات فنية بالعربي والإنجليزي، Error Code لو موجود،
  اسم الماكينة أو القطعة، مرادفات). فاضية لو النية chitchat
  أو about_assistant.

- clarifying_question: سؤال توضيحي واحد قصير ومفيد لو ناقص
  معلومة مهمة (موديل الماكينة، كود الخطأ، متى بيحصل العطل)،
  وإلا "".

قواعد:
- لا تخترع معلومات.
- لو الرسالة غامضة اختار أقرب تفسير معقول بدل unclear،
  واستخدم unclear فقط لو مفيش أي تفسير.
- أرجع JSON صالح فقط، بدون شرح وبدون markdown.
"""

ASSISTANT_SYSTEM_PROMPT = """
أنت CUPII، مساعد ذكي للصيانة الصناعية.

اللي تقدر تعمله فعلاً:
- تجاوب من الكتالوجات والمستندات اللي المستخدم يرفعها (PDF أو TXT).
- تشرح Error Codes والأعطال وخطوات الفحص والإصلاح.
- تحلل صور الأعطال والمكونات.
- تفهم الرسائل الصوتية وتتكلم في المكالمة الصوتية المباشرة.
- تفتكر سياق المحادثة الحالية (في الشات وفي المكالمة الصوتية كمان).
- لو المعلومة مش في المستندات، تساعد بمعلومات هندسية عامة
  مع تنبيه واضح إنها مش من الكتالوج.

القواعد:
- جاوب بطبيعية على أسئلة زي "بتفهمني؟" و"تقدر تعمل إيه؟" و"مين انت؟".
- متدّعيش قدرات مش موجودة فوق.
- لو المستخدم قال حاجة عن نفسه أو عن شغله في سياق المحادثة
  (اسمه، ماكينته، مشكلته)، افتكرها واستخدمها في ردك.
- جاوب بلهجة المستخدم (مصري لو كتب مصري)، قصير ومباشر وودود.
- من غير markdown: مفيش # ولا ** ولا جداول.
- في الآخر، لو مناسب، اقترح للمستخدم يسأل عن إيه.
"""

ANSWER_SYSTEM_PROMPT = """
أنت CUPII، مساعد صيانة صناعية بيجاوب من قاعدة معرفة
(كتالوجات ومستندات فنية) وبيفهم أي سؤال بأي صياغة.

ستستلم: سياق المحادثة، سؤال المستخدم، فهمنا للسؤال،
و CONTEXT من قاعدة المعرفة (ممكن يكون مرتبط أو لأ).

اللي تعمله حسب حالة الـCONTEXT:

1) CONTEXT مرتبط ويجاوب السؤال:
   جاوب اعتمادًا عليه، وميّز بين السبب والتشخيص والحل.
   لو فيه Error Code اعتبره عنصر أساسي.

2) CONTEXT مرتبط جزئيًا:
   جاوب باللي مدعوم بيه، وقل بوضوح إيه اللي ناقص.

3) CONTEXT مش مرتبط أو فاضي:
   ممنوع تقول "البيانات غير كافية" وتقف. اعمل التالي:
   - جملة قصيرة: مالقيتش الموضوع ده في المستندات المرفوعة.
   - جاوب من معرفتك الهندسية العامة بشكل مفيد، وابدأه بعبارة
     "معلومة عامة (مش من الكتالوجات):" ونبّه إنه يتأكد من كتالوج الماكينة.
   - اقترح إيه اللي يساعدك تدقق: موديل الماكينة، كود الخطأ،
     صورة، أو رفع الكتالوج.

4) السؤال غامض:
   قل فهمك الأقرب في جملة، جاوب عليه، وبعدين اسأل سؤال
   توضيحي واحد بس. متردش بـ"مفهمتش" لوحدها أبدًا.

قواعد عامة:
- استخدم سياق المحادثة عشان تفهم الإشارات (ده، هو، والحل؟)،
  وافتكر أي معلومة المستخدم قالها قبل كده (اسمه، الماكينة، الكود،
  اللي جربه) ومتسألوش عليها تاني.
- لا تخترع أرقام أو مواصفات أو أكواد. لو مش متأكد قول كده.
- لا تقدم احتمال كأنه مؤكد.
- في أي إجراء كهربائي أو ضغط أو أجزاء متحركة، نبّه على فصل الطاقة
  وإجراءات الأمان قبل الفحص.
- لو السؤال مش فني (عام أو ودي) جاوب عليه طبيعي.
- لا تذكر اسم الملف ولا رقم الصفحة.
- جاوب بلهجة المستخدم (مصري لو كتب مصري)، قصير ومنظم.
- من غير markdown: مفيش # ولا ** ولا جداول. استخدم فقرات
  قصيرة وقوائم بسيطة بشرطة (-).
"""

# بيتضاف لآخر الـprompt في وضع المكالمة الصوتية
VOICE_STYLE_SUFFIX = """

وضع المكالمة الصوتية (مهم جدًا):
- الرد هيتنطق بصوت عالي، فخليه قصير: من 2 إلى 3 جمل فقط (حوالي 40 كلمة).
- ابدأ بالمعلومة المفيدة على طول بدون مقدمات.
- جمل قصيرة وسهلة النطق.
- ممنوع القوايم والشرطات والرموز والإيموجي والجداول.
- لو الموضوع طويل، اديه أهم خطوة أو أهم سبب، واسأل لو عايز تكمل بالتفاصيل.
- لو في إجراء فيه خطر (كهرباء/ضغط)، جملة تنبيه قصيرة بس.
- إنت فاكر كل اللي اتقال في المكالمة دي، فلو المستخدم سأل عن حاجة قالها
  قبل كده جاوب عليها من سياق المحادثة، ومتقولش إنك مش فاكر.
"""


# ============================================================
# HELPERS
# ============================================================

def split_history(raw: str):
    """
    الفرونت بيبعت: [تاريخ المحادثة] + 'السؤال الحالي:' + السؤال.
    هنا بنفصلهم عشان البحث يتعمل على السؤال بس، والتاريخ
    يتستخدم لفهم الإشارات.
    """

    raw = (raw or "").strip()

    if CURRENT_MARKER in raw:

        history, current = raw.rsplit(CURRENT_MARKER, 1)

        history = history.replace(HISTORY_HEADER, "").strip()

        return history, current.strip()

    return "", raw


# ============================================================
# VOICE CONVERSATION MEMORY (ذاكرة المكالمة الصوتية في Qdrant)
# ============================================================
# المشكلة: في الشات الفرونت بيبعت تاريخ المحادثة مع كل سؤال،
# لكن في المكالمة الصوتية ممكن يبعت السؤال لوحده. الحل: السيرفر
# نفسه بيحتفظ بآخر أدوار المكالمة لكل session_id، ويحطها في
# الـprompt تلقائي.
#
# - لو الفرونت بعت تاريخ (HISTORY_HEADER + السؤال الحالي) بنستخدمه.
# - لو مبعتش، بنقرا الذاكرة المحفوظة في Qdrant.
# - الذاكرة بقت دايمة: بتفضل بعد restart للسيرفر.
# - "النسيان" بعد VOICE_MEMORY_TTL بيتم وقت القراءة (بنتجاهل
#   الأقدم)، وبنمسح الأقدم فعليًا مع كل حفظ جديد.
# ============================================================

def _voice_key(session_id: Optional[str]) -> str:
    """
    لو مفيش session_id بنستخدم مفتاح واحد مشترك "default".
    ده كفاية لمستخدم واحد، لكن لو في أكتر من مستخدم لازم
    الراوت يبعت session_id مختلف لكل مكالمة.
    """

    return (session_id or "").strip() or "default"


def _voice_chat_id(session_id: Optional[str]) -> str:

    return "voice:" + _voice_key(session_id)


def get_voice_history(session_id: Optional[str] = None) -> str:
    """يرجّع تاريخ المكالمة كنص بنفس شكل الشات (المستخدم/المساعد)."""

    try:

        messages = chat_get_messages(_voice_chat_id(session_id))

    except Exception as e:

        print(f"Voice memory read failed: {e}")

        return ""

    cutoff = int((time.time() - VOICE_MEMORY_TTL) * 1000)

    messages = [
        m for m in messages
        if m["ts"] >= cutoff
    ][-VOICE_MEMORY_TURNS * 2:]

    lines = []

    for m in messages:

        label = "المستخدم" if m["role"] == "user" else "المساعد"

        lines.append(f"{label}: {m['content']}")

    return "\n".join(lines)


def add_voice_turn(
    session_id: Optional[str],
    user_text: str,
    answer: str
):
    """يسجّل دور جديد (سؤال + رد) في ذاكرة المكالمة على Qdrant."""

    user_text = (user_text or "").strip()

    answer = (answer or "").strip()

    if not user_text or not answer:
        return

    # نقصّر الرد المحفوظ عشان الـprompt ميكبرش
    if len(answer) > VOICE_MEMORY_ANSWER_CHARS:
        answer = answer[:VOICE_MEMORY_ANSWER_CHARS].rstrip() + "..."

    chat_id = _voice_chat_id(session_id)

    try:

        init_chats_collection()

        now = int(time.time() * 1000)

        qdrant.upsert(
            collection_name=CHATS_COLLECTION,
            points=[
                _message_point(chat_id, "user", user_text, now),
                _message_point(chat_id, "assistant", answer, now + 1),
            ]
        )

        # نمسح الرسايل الأقدم من TTL فعليًا عشان الـcollection متكبرش
        cutoff = now - VOICE_MEMORY_TTL * 1000

        qdrant.delete(
            collection_name=CHATS_COLLECTION,
            points_selector=FilterSelector(
                filter=Filter(
                    must=[
                        FieldCondition(
                            key="chat_id",
                            match=MatchValue(value=chat_id)
                        ),
                        FieldCondition(
                            key="ts",
                            range=Range(lt=cutoff)
                        ),
                    ]
                )
            )
        )

    except Exception as e:

        print(f"Voice memory save failed: {e}")


def clear_voice_memory(session_id: Optional[str] = None):
    """يمسح ذاكرة جلسة (استخدمها لما المكالمة تخلص أو يبدأ المستخدم من جديد)."""

    try:

        chat_delete(_voice_chat_id(session_id))

    except Exception as e:

        print(f"Voice memory clear failed: {e}")


def _generate(
    system_prompt: str,
    prompt: str,
    temperature: float = 0.1,
    max_tokens: Optional[int] = None
):

    text = _chat(
        system_prompt,
        prompt,
        temperature=temperature,
        max_tokens=max_tokens
    )

    if not text:
        return "معرفتش أطلّع رد المرة دي. جرّب تعيد صياغة السؤال بشكل تاني."

    return text


def understand_question(history: str, question: str) -> dict:
    """
    خطوة فهم: نية السؤال + سؤال مستقل + استعلامات بحث.
    بتشتغل على UNDERSTAND_MODEL (موديل خفيف وسريع).
    لو فشلت لأي سبب، بنرجع لسلوك آمن (بحث عادي بالسؤال نفسه).
    """

    fallback = {
        "intent": "document_question",
        "standalone_question": question,
        "search_queries": [],
        "clarifying_question": "",
    }

    prompt = (
        f"سياق المحادثة السابق:\n{history or '(لا يوجد)'}\n\n"
        f"رسالة المستخدم الحالية:\n{question}\n\n"
        "أرجع الرد كـ JSON فقط."
    )

    try:

        raw = _chat(
            UNDERSTAND_PROMPT,
            prompt,
            model=UNDERSTAND_MODEL,
            temperature=0.0,
            json_mode=True,
            max_tokens=1500
        )

        raw = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw).strip()

        data = json.loads(raw)

        intent = data.get("intent")

        if intent not in VALID_INTENTS:
            intent = "document_question"

        standalone = (
            str(data.get("standalone_question") or "").strip()
            or question
        )

        queries = [
            str(q).strip()
            for q in (data.get("search_queries") or [])
            if str(q).strip()
        ][:3]

        return {
            "intent": intent,
            "standalone_question": standalone,
            "search_queries": queries,
            "clarifying_question": str(
                data.get("clarifying_question") or ""
            ).strip(),
        }

    except Exception as e:

        print(f"Understanding step failed, using fallback: {e}")

        return fallback


def retrieve_for_queries(queries, question, limit: int = 7):
    """
    بحث متعدد الاستعلامات:
    - embedding لكل الاستعلامات في batch واحد (بدل واحد واحد)
    - بحث Error Code مرة واحدة
    - دمج بدون تكرار + rerank مرة واحدة
    """

    queries = [q for q in queries if q and q.strip()]

    if not queries:
        return []

    merged = {}

    # ---------- 1) EXACT ERROR CODE (من السؤال ومن الاستعلامات) ----------

    codes = []

    for text in [question] + queries:

        code = extract_error_code(text)

        if code and code not in codes:
            codes.append(code)

    for code in codes:

        for result in search_by_error_code(code, limit=10):

            merged.setdefault(result.id, result)

    # ---------- 2) SEMANTIC (batch embedding واحد) ----------

    try:

        vectors = create_embeddings_batch(
            queries,
            task_type="RETRIEVAL_QUERY"
        )

    except Exception as e:

        print(f"Embedding failed: {e}")

        vectors = []

    for q, vector in zip(queries, vectors):

        try:

            points = qdrant.query_points(
                collection_name=COLLECTION_NAME,
                query=vector,
                limit=SEMANTIC_CANDIDATES,
                with_payload=True
            ).points

            for result in points:

                merged.setdefault(result.id, result)

        except Exception as e:

            print(f"Search failed for query '{q}': {e}")

    if not merged:
        return []

    # ---------- 3) RERANK مرة واحدة ----------

    return rerank_results(
        question,
        list(merged.values()),
        limit=limit,
    )


# ============================================================
# TEXT RAG
# ============================================================

def ask_rag(
    question: str,
    voice: bool = False,
    session_id: Optional[str] = None
):
    """
    voice=True: وضع المكالمة الصوتية
      - رد قصير (2-3 جمل) عشان الـTTS يخلص بسرعة
      - context أصغر (VOICE_RETRIEVAL_LIMIT)
      - حد أقصى للـtokens
      - ذاكرة المكالمة في Qdrant (بحسب session_id)
    """

    history, current = split_history(question)

    # ذاكرة المكالمة: لو الفرونت مبعتش تاريخ، نستخدم المحفوظ في Qdrant
    if voice and not history:
        history = get_voice_history(session_id)

    # ========================================================
    # FAST PATH: تحيات صريحة (من غير استدعاء LLM)
    # ========================================================

    if any(
        re.match(pattern, current, re.IGNORECASE)
        for pattern in CASUAL_PATTERNS
    ):

        greeting = GREETING_ANSWER_VOICE if voice else GREETING_ANSWER

        if voice:
            add_voice_turn(session_id, current, greeting)

        return {
            "answer": greeting,
            "sources": [],
            "intent": "chitchat",
        }

    style_suffix = VOICE_STYLE_SUFFIX if voice else ""

    max_tokens = VOICE_MAX_TOKENS if voice else None

    retrieval_limit = VOICE_RETRIEVAL_LIMIT if voice else 7

    # ========================================================
    # UNDERSTAND
    # ========================================================

    understanding = understand_question(history, current)

    intent = understanding["intent"]

    standalone = understanding["standalone_question"]

    history_block = history if history else "(لا يوجد)"

    # ========================================================
    # CHITCHAT / ABOUT ASSISTANT: من غير بحث
    # ========================================================

    if intent in ("chitchat", "about_assistant"):

        answer = _generate(
            ASSISTANT_SYSTEM_PROMPT + style_suffix,
            (
                f"سياق المحادثة السابق:\n{history_block}\n\n"
                f"رسالة المستخدم:\n{current}"
            ),
            temperature=0.4,
            max_tokens=max_tokens,
        )

        if voice and not answer.startswith("معرفتش أطلّع"):
            add_voice_turn(session_id, current, answer)

        return {
            "answer": answer,
            "sources": [],
            "intent": intent,
        }

    # ========================================================
    # CASE MEMORY (shadow mode: بنسجل بس، مش بنغيّر الرد)
    # ========================================================

    if not voice:

        hits = search_solved_cases(standalone)

        if hits:
            print(
                f"[case-shadow] score={hits[0].score:.3f} "
                f"| new: {standalone[:50]} "
                f"| old: {hits[0].payload.get('problem', '')[:50]}"
            )

    # ========================================================
    # RETRIEVAL (سؤال مستقل + استعلامات مولّدة)
    # ========================================================

    queries = []

    for q in [standalone] + understanding["search_queries"]:

        if q and q not in queries:
            queries.append(q)

    queries = queries[:3]

    results = retrieve_for_queries(
        queries,
        standalone,
        limit=retrieval_limit
    )

    if results:
        context, sources = build_rag_context(results)
    else:
        context, sources = "", []

    # ========================================================
    # ANSWER
    # ========================================================

    clarify_line = ""

    if understanding["clarifying_question"]:

        clarify_line = (
            "سؤال توضيحي مقترح (استخدمه في آخر الرد لو مفيد): "
            f"{understanding['clarifying_question']}"
        )

    user_prompt = f"""
سياق المحادثة السابق:
{history_block}

سؤال المستخدم كما كتبه:
{current}

فهمي للسؤال:
{standalone}
{clarify_line}

CONTEXT (نتائج من قاعدة المعرفة، وممكن تكون غير مرتبطة بالسؤال):
{context if context else "(لم يتم العثور على نتائج)"}
"""

    answer = _generate(
        ANSWER_SYSTEM_PROMPT + style_suffix,
        user_prompt,
        temperature=0.2,
        max_tokens=max_tokens,
    )

    if voice and not answer.startswith("معرفتش أطلّع"):
        add_voice_turn(session_id, current, answer)

    # نحفظ بس لو: نص (مش صوت) + لقينا نتائج من الكتالوج + الرد مش الـ fallback
    if (
        not voice
        and results
        and intent == "document_question"
        and not answer.startswith("معرفتش أطلّع")
    ):
        save_solved_case(standalone, answer, sources, intent)

    return {
        "answer": answer,
        "sources": sources,
        "intent": intent,
    }


# ============================================================
# VOICE STREAMING RAG (جملة بجملة)
# ============================================================
# بدل ما نستنى الرد كله، بنبعت كل جملة أول ما تتكوّن، والفرونت
# بيبعتها للـTTS فورًا. وكمان بنتخطى خطوة "فهم السؤال" (نداء LLM
# كامل) ونعتمد على الـhistory جوه نفس الـprompt.
# ============================================================

_SENTENCE_END_CHARS = "!?؟…\n"


def _pop_sentence(buf: str, min_len: int):
    """يرجّع (جملة, الباقي) لو فيه جملة كاملة في buf، وإلا (None, buf)."""

    for i, ch in enumerate(buf):

        if ch in _SENTENCE_END_CHARS:
            is_end = True

        elif ch == ".":
            # النقطة بتكون نهاية جملة بس لو بعدها مسافة (عشان 3.5 مثلًا)
            is_end = i + 1 < len(buf) and buf[i + 1].isspace()

        else:
            is_end = False

        if is_end:

            sentence = buf[: i + 1].strip()

            if len(sentence) >= min_len:
                return sentence, buf[i + 1:].lstrip()

    return None, buf


def _extra_body(model_name: str):

    if "gpt-oss" in model_name.lower() and REASONING_EFFORT:
        return {"reasoning_effort": REASONING_EFFORT}

    return None


def ask_rag_voice_stream(
    question: str,
    session_id: Optional[str] = None
):
    """
    Generator بيطلّع events:
      {"type": "sentence", "text": "..."}   كل جملة أول ما تجهز
      {"type": "done", "answer": "...", "sources": [...]}

    session_id: معرّف المكالمة. بنستخدمه عشان نفتكر الكلام السابق.
    """

    t0 = time.time()

    history, current = split_history(question)

    # ذاكرة المكالمة: لو الفرونت مبعتش تاريخ، نستخدم المحفوظ في Qdrant
    if not history:
        history = get_voice_history(session_id)

    # تحيات صريحة: من غير LLM
    if any(
        re.match(pattern, current, re.IGNORECASE)
        for pattern in CASUAL_PATTERNS
    ):

        add_voice_turn(session_id, current, GREETING_ANSWER_VOICE)

        yield {"type": "sentence", "text": GREETING_ANSWER_VOICE}

        yield {
            "type": "done",
            "answer": GREETING_ANSWER_VOICE,
            "sources": [],
        }

        return

    history_block = history if history else "(لا يوجد)"

    # ---------- RETRIEVAL (من غير خطوة الفهم) ----------

    retrieval_query = current

    # سؤال قصير زي "والحل؟" : نضيف آخر سؤال للمستخدم عشان البحث يفهم
    if history and len(current.split()) < 7:

        for line in reversed(history.split("\n")):

            if line.startswith("المستخدم:"):

                last_user = line[len("المستخدم:"):].strip()

                if last_user:
                    retrieval_query = f"{last_user} {current}"

                break

    try:

        results = retrieve_for_queries(
            [retrieval_query],
            retrieval_query,
            limit=VOICE_RETRIEVAL_LIMIT
        )

    except Exception as e:

        print(f"Voice retrieval failed: {e}")

        results = []

    if results:
        context, sources = build_rag_context(results)
    else:
        context, sources = "", []

    t_retrieved = time.time()

    user_prompt = f"""
سياق المحادثة السابق:
{history_block}

سؤال المستخدم:
{current}

CONTEXT (نتائج من قاعدة المعرفة، وممكن تكون غير مرتبطة بالسؤال):
{context if context else "(لم يتم العثور على نتائج)"}
"""

    kwargs: Dict[str, Any] = {
        "model": VOICE_CHAT_MODEL,
        "messages": [
            {
                "role": "system",
                "content": ANSWER_SYSTEM_PROMPT + VOICE_STYLE_SUFFIX
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ],
        "temperature": 0.2,
        "max_tokens": VOICE_MAX_TOKENS,
        "stream": True,
    }

    extra = _extra_body(VOICE_CHAT_MODEL)

    if extra:
        kwargs["extra_body"] = extra

    buf = ""

    full_parts = []

    first_sentence_logged = False

    try:

        stream = _call_with_retry(
            groq_client.chat.completions.create,
            **kwargs
        )

        for event in stream:

            if not event.choices:
                continue

            delta = event.choices[0].delta.content or ""

            if not delta:
                continue

            buf += delta

            while True:

                sentence, buf = _pop_sentence(buf, VOICE_MIN_SENTENCE)

                if not sentence:
                    break

                if not first_sentence_logged:

                    first_sentence_logged = True

                    print(
                        f"[voice] retrieve={t_retrieved - t0:.2f}s "
                        f"first_sentence={time.time() - t0:.2f}s"
                    )

                full_parts.append(sentence)

                yield {"type": "sentence", "text": sentence}

    except Exception as e:

        print(f"Voice stream failed: {e}")

        if not full_parts and not buf.strip():

            fallback = "حصلت مشكلة وأنا بجهز الرد. جرّب تسأل تاني."

            yield {"type": "sentence", "text": fallback}

            yield {"type": "done", "answer": fallback, "sources": sources}

            return

    tail = buf.strip()

    if tail:

        full_parts.append(tail)

        yield {"type": "sentence", "text": tail}

    if not full_parts:

        fallback = "معرفتش أطلّع رد المرة دي. جرّب تعيد السؤال."

        full_parts.append(fallback)

        yield {"type": "sentence", "text": fallback}

    else:

        # نحفظ الدور في ذاكرة المكالمة (بس لو الرد حقيقي مش fallback)
        add_voice_turn(session_id, current, " ".join(full_parts))

    print(f"[voice] total={time.time() - t0:.2f}s")

    yield {
        "type": "done",
        "answer": " ".join(full_parts),
        "sources": sources,
    }


# ============================================================
# AUDIO TRANSCRIPTION (Groq Whisper)
# ============================================================

NO_SPEECH_TEXT = "(لم يتم التعرف على كلام واضح)"

_AUDIO_EXTENSIONS = {
    "audio/webm": "webm",
    "audio/ogg": "ogg",
    "audio/mp4": "mp4",
    "audio/x-m4a": "m4a",
    "audio/m4a": "m4a",
    "audio/mpeg": "mp3",
    "audio/mp3": "mp3",
    "audio/wav": "wav",
    "audio/x-wav": "wav",
    "audio/wave": "wav",
    "audio/flac": "flac",
}


def _audio_extension(mime_type: str) -> str:

    base = (mime_type or "").split(";")[0].strip().lower()

    return _AUDIO_EXTENSIONS.get(base, "webm")


def transcribe_audio(
    audio_bytes,
    mime_type
):

    kwargs: Dict[str, Any] = {

        "file": (
            f"audio.{_audio_extension(mime_type)}",
            audio_bytes
        ),

        "model": STT_MODEL,

        "language": STT_LANGUAGE,

        "temperature": 0.0,

        "response_format": "json",
    }

    if STT_PROMPT:
        kwargs["prompt"] = STT_PROMPT

    response = _call_with_retry(
        groq_client.audio.transcriptions.create,
        **kwargs
    )

    text = (
        getattr(response, "text", "")
        or ""
    ).strip()

    if not text:
        return NO_SPEECH_TEXT

    return text