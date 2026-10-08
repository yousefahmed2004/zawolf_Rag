"""
سكريبت لفحص الـvectors والـpayloads المخزنة في Qdrant.

طريقة التشغيل:

    python inspect_qdrant.py            # يعرض أول 5 points
    python inspect_qdrant.py 20         # يعرض أول 20 point
    python inspect_qdrant.py --filename machine.pdf   # يفلتر بملف معيّن
"""

import os
import sys

from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

load_dotenv()

QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")
COLLECTION_NAME = os.getenv("QDRANT_COLLECTION", "zawolf_documents")


def main():

    print(f"🔌 بحاول أتصل بـ Qdrant على {QDRANT_URL} ...")

    limit = 5
    filename_filter = None

    args = sys.argv[1:]

    if "--filename" in args:
        idx = args.index("--filename")
        filename_filter = args[idx + 1]
        args = args[:idx] + args[idx + 2:]

    if args and args[0].isdigit():
        limit = int(args[0])

    client = QdrantClient(url=QDRANT_URL)

    if not client.collection_exists(collection_name=COLLECTION_NAME):
        print(f"❌ الـcollection '{COLLECTION_NAME}' مش موجودة. اترفع مستند الأول.")
        return

    info = client.get_collection(collection_name=COLLECTION_NAME)
    print(f"📦 Collection: {COLLECTION_NAME}")
    print(f"   عدد الـpoints الكلي: {info.points_count}")
    print(f"   أبعاد الـvector: {info.config.params.vectors.size}")
    print(f"   نوع المسافة: {info.config.params.vectors.distance}")
    print("=" * 70)

    query_filter = None
    if filename_filter:
        query_filter = Filter(
            must=[FieldCondition(key="filename", match=MatchValue(value=filename_filter))]
        )
        print(f"🔍 فلترة على filename = {filename_filter}\n")

    points, _ = client.scroll(
        collection_name=COLLECTION_NAME,
        scroll_filter=query_filter,
        limit=limit,
        with_payload=True,
        with_vectors=True
    )

    if not points:
        print("مفيش points تتعرض (جرب من غير فلتر أو ارفع مستند الأول).")
        return

    for i, point in enumerate(points, start=1):

        payload = point.payload or {}
        vector = point.vector or []

        print(f"\n--- Point {i} | id: {point.id} ---")
        print(f"filename     : {payload.get('filename')}")
        print(f"page         : {payload.get('page')}")
        print(f"section      : {payload.get('section')}")
        print(f"subsection   : {payload.get('subsection')}")
        print(f"topic        : {payload.get('topic')}")
        print(f"error_code   : {payload.get('error_code')}")
        print(f"content_type : {payload.get('content_type')}")

        text = payload.get("text", "")
        preview = text[:150].replace("\n", " ")
        print(f"text preview : {preview}{'...' if len(text) > 150 else ''}")

        print(f"vector dims  : {len(vector)}")
        print(f"vector[:5]   : {[round(v, 4) for v in vector[:5]]}")

    print("\n" + "=" * 70)
    print(f"اتعرض {len(points)} من أصل {info.points_count} point.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        import traceback
        print("❌ حصل خطأ أثناء تشغيل السكريبت:")
        traceback.print_exc()
        print(f"\nنوع الخطأ: {type(e).__name__}")
        print(f"الرسالة: {e}")
        print(f"\nتأكد إن Qdrant شغال على {QDRANT_URL} (جرب: docker ps)")