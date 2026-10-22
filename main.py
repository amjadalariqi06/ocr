from __future__ import annotations

import io
import re
import unicodedata
from typing import Optional

import fitz
import numpy as np
import pytesseract
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, ImageEnhance, ImageFilter, UnidentifiedImageError

APP_VERSION = "3.0.0"
MAX_FILE_BYTES = 15 * 1024 * 1024
MAX_PDF_PAGES = 20
OCR_LANG = "ara+eng"

app = FastAPI(title="Umrah Track OCR API", version=APP_VERSION)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, set explicit frontend origins.
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

UNKNOWN = "غير متاح"

@app.get("/")
def home():
    return {"status": "Running", "service": "OCR API", "version": APP_VERSION}

@app.get("/health")
def health():
    try:
        languages = pytesseract.get_languages(config="")
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"تعذر تشغيل Tesseract: {exc}")
    return {"status": "ok", "ocr_languages": languages, "version": APP_VERSION}


def normalize_digits(text: str) -> str:
    table = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
    return text.translate(table)


def normalize_text(text: str) -> str:
    text = unicodedata.normalize("NFKC", text or "")
    text = text.replace("\u200e", "").replace("\u200f", "")
    text = text.replace("：", ":").replace("／", "/")
    text = normalize_digits(text)
    text = re.sub(r"[\t\xa0]+", " ", text)
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _image_variants(image: Image.Image) -> list[Image.Image]:
    """Generate a small set of conservative OCR inputs; keep original dimensions reasonable."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    image.thumbnail((2600, 2600), Image.Resampling.LANCZOS)
    gray = ImageOps.grayscale(image)
    gray = ImageOps.autocontrast(gray, cutoff=1)
    gray = ImageEnhance.Contrast(gray).enhance(1.25)
    if max(gray.size) < 1500:
        scale = min(2.0, 1500 / max(gray.size))
        gray = gray.resize((int(gray.width * scale), int(gray.height * scale)), Image.Resampling.LANCZOS)
    sharp = gray.filter(ImageFilter.UnsharpMask(radius=1.5, percent=150, threshold=3))
    # Adaptive threshold can help faded scans but may hurt colored/complex documents.
    arr = np.array(sharp)
    try:
        import cv2
        binary = cv2.adaptiveThreshold(arr, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                       cv2.THRESH_BINARY, 35, 11)
        thresholded = Image.fromarray(binary)
    except Exception:
        thresholded = sharp.point(lambda p: 255 if p > 170 else 0)
    return [image, sharp, thresholded]


def _ocr_score(text: str) -> int:
    """Rank OCR candidates for document-field usefulness, not for truth/authenticity."""
    t = normalize_text(text).lower()
    score = min(len(t), 1000) // 20
    cues = ["passport", "passport no", "ticket number", "passenger name", "visa no",
            "name", "الجواز", "رقم", "الاسم", "التأشيرة", "التذكرة", "تاريخ"]
    score += sum(12 for cue in cues if cue in t)
    score += min(len(re.findall(r"\d", t)), 40)
    # Penalize mostly symbol noise.
    useful = sum(ch.isalnum() or '\u0600' <= ch <= '\u06ff' for ch in t)
    score -= max(0, len(t) - useful) // 5
    return score


def ocr_image(image: Image.Image) -> str:
    variants = _image_variants(image)
    candidates = []
    configs = ("--oem 3 --psm 6", "--oem 3 --psm 11", "--oem 3 --psm 4")
    languages = [OCR_LANG, "eng"]
    # Prefer bilingual OCR; fall back to English if Arabic trained data is not installed.
    try:
        available = set(pytesseract.get_languages(config=""))
        if "ara" not in available:
            languages = ["eng"]
    except Exception:
        languages = ["eng"]
    for variant in variants:
        for config in configs:
            for lang in languages[:1]:
                try:
                    text = pytesseract.image_to_string(variant, lang=lang, config=config)
                    if text.strip():
                        candidates.append(text)
                except pytesseract.TesseractError:
                    continue
    if not candidates:
        raise RuntimeError("لم يتمكن OCR من قراءة الصورة. تحقق من تثبيت لغتي ara وeng في Tesseract.")
    return max(candidates, key=_ocr_score)


def pdf_page_text(page: fitz.Page) -> str:
    digital = page.get_text("text", sort=True).strip()
    # Use embedded text when substantial; OCR scanned pages or nearly-empty text layers.
    if len(re.sub(r"\s", "", digital)) >= 35:
        return digital
    pix = page.get_pixmap(dpi=300, alpha=False)
    image = Image.open(io.BytesIO(pix.tobytes("png")))
    return ocr_image(image)


def find_labeled_value(text: str, labels: list[str], value_pattern: str,
                       allow_next_line: bool = True) -> Optional[str]:
    """Extract only a value near a recognized field label; avoid unrelated-number fallbacks."""
    lines = [re.sub(r"\s+", " ", line).strip(" \t:：-|_") for line in text.splitlines()]
    label_alt = "|".join(labels)
    label_re = re.compile(rf"(?:{label_alt})", re.IGNORECASE)
    value_re = re.compile(value_pattern, re.IGNORECASE)
    for i, line in enumerate(lines):
        match = label_re.search(line)
        if not match:
            continue
        tail = line[match.end():].strip(" \t:：#-_|")
        # Don't accept a value if the label line contains another field label.
        if tail:
            value = value_re.search(tail)
            if value:
                return _clean_value(value.group(0))
        if allow_next_line and i + 1 < len(lines):
            next_line = lines[i + 1].strip()
            if next_line and not label_re.search(next_line) and len(next_line) < 100:
                value = value_re.search(next_line)
                if value:
                    return _clean_value(value.group(0))
    # Some OCR output merges label and value with unusual line breaks.
    flat = re.sub(r"\s+", " ", text)
    match = re.search(rf"(?:{label_alt})\s*[:：#-]?\s*({value_pattern})", flat, re.IGNORECASE)
    return _clean_value(match.group(1)) if match else None


def _clean_value(value: str) -> str:
    value = normalize_digits(value)
    value = re.sub(r"\s+", " ", value).strip(" :：#-_|/.,;")
    return value if value else ""


def _extract_name(text: str, doc_type: str) -> Optional[str]:
    # Airline tickets commonly use SURNAME/GIVEN NAME after PASSENGER NAME.
    if doc_type == "air_ticket":
        m = re.search(r"PASSENGER\s*NAME\s*[:：]?\s*([A-Z][A-Z /'-]{2,70}?)(?=\s{2,}|\b(?:BOOKING|TICKET|ISSUED|DATE|FLIGHT)\b|\n|$)", text, re.I)
        if m:
            value = re.sub(r"\s+", " ", m.group(1)).strip(" /-")
            if len(value) >= 3:
                return value
    # Explicit name labels only. Do not guess from arbitrary lines.
    value = find_labeled_value(text, [r"FULL\s*NAME", r"PASSENGER\s*NAME", r"(?:^|\b)NAME\b", r"الاسم الكامل", r"اسم المعتمر", r"الاسم"],
                               r"[A-Z][A-Z /'-]{2,70}|[\u0600-\u06ff][\u0600-\u06ff \u0640]{2,70}")
    if value:
        value = re.sub(r"\s+", " ", value).strip()
        # Avoid returning the label itself or generic OCR junk.
        if len(value) >= 3 and not re.fullmatch(r"(?:FULL\s*)?NAME|الاسم(?: الكامل)?", value, re.I):
            return value
    return None


def parse_document(text: str) -> dict:
    text = normalize_text(text)
    low = text.lower()
    is_air = any(x in low for x in ["yemenia", "passenger name", "ticket number", "e-ticket"])
    is_visa = any(x in low for x in ["visa no", "umrah operator", "تأشيرة السعودية", "تأشيرة العمرة", "رقم التأشيرة", "وزارة الحج"])
    is_local = any(x in low for x in ["سماء النخبة", "البركة", "رقم التذكرة", "رقم الجواز"])
    if is_air:
        kind = "تذكرة طيران"
        type_key = "air_ticket"
    elif is_visa:
        kind = "تأشيرة / وثيقة عمرة (تصنيف نصي فقط)"
        type_key = "visa"
    elif is_local:
        kind = "تذكرة سفر محلية / برية"
        type_key = "local_ticket"
    else:
        kind = "مستند غير معروف"
        type_key = "unknown"

    # Field-specific patterns: never assign an arbitrary long number as a ticket/visa number.
    passport = find_labeled_value(text,
        [r"PASSPORT\s*(?:NO\.?|NUMBER|#)?", r"رقم\s*جواز\s*السفر", r"رقم\s*الجواز"],
        r"[A-Z]{0,2}[0-9]{5,10}|[A-Z0-9]{6,12}")
    reference = find_labeled_value(text,
        [r"VISA\s*(?:NO\.?|NUMBER|#)", r"TICKET\s*(?:NO\.?|NUMBER|#)", r"E-?TICKET\s*(?:NO\.?|NUMBER|#)",
         r"رقم\s*التأشيرة", r"رقم\s*التذكرة", r"رقم\s*الحجز", r"BOOKING\s*REFERENCE", r"PNR"],
        r"[A-Z0-9][A-Z0-9 -]{4,19}")
    date_value = find_labeled_value(text,
        [r"DATE\s*OF\s*ISSUE", r"ISSUE\s*DATE", r"DEPARTURE\s*DATE", r"TRAVEL\s*DATE", r"DATE", r"تاريخ\s*الإصدار", r"تاريخ\s*الرحلة", r"تاريخ\s*السفر", r"تاريخ\s*بداية\s*الصلاحية"],
        r"(?:\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{1,2}\s+[A-Z]{3}\s+\d{4}|\d{4}[./-]\d{1,2}[./-]\d{1,2})")
    # For known air-ticket format, passenger name can be extracted from its explicit heading.
    name = _extract_name(text, type_key)

    return {
        "full_name": name or UNKNOWN,
        "passport_number": passport or UNKNOWN,
        "reference_number": reference or UNKNOWN,
        "date": date_value or UNKNOWN,
        "document_type": kind,
        "review_required": True,
        "extraction_note": "نتائج OCR آلية؛ يجب مطابقتها مع المستند الأصلي. الحقول غير المؤكدة تُترك غير متاحة بدل التخمين."
    }


@app.post("/extract-document/")
async def extract_document(file: UploadFile = File(...)):
    filename = (file.filename or "document").lower()
    if not filename.endswith((".pdf", ".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff")):
        raise HTTPException(status_code=400, detail="صيغة الملف غير مدعومة. استخدم PDF أو صورة PNG/JPG.")
    contents = await file.read(MAX_FILE_BYTES + 1)
    if not contents:
        raise HTTPException(status_code=400, detail="الملف فارغ.")
    if len(contents) > MAX_FILE_BYTES:
        raise HTTPException(status_code=413, detail="حجم الملف يتجاوز 15 ميجابايت.")

    try:
        page_texts = []
        if filename.endswith(".pdf"):
            try:
                pdf = fitz.open(stream=contents, filetype="pdf")
            except Exception as exc:
                raise HTTPException(status_code=400, detail="تعذر فتح ملف PDF؛ قد يكون تالفًا أو محميًا.") from exc
            with pdf:
                if pdf.is_encrypted:
                    raise HTTPException(status_code=400, detail="ملفات PDF المحمية بكلمة مرور غير مدعومة.")
                if len(pdf) > MAX_PDF_PAGES:
                    raise HTTPException(status_code=413, detail=f"الحد الأقصى هو {MAX_PDF_PAGES} صفحة لكل ملف PDF.")
                for page in pdf:
                    page_texts.append(pdf_page_text(page))
        else:
            try:
                image = Image.open(io.BytesIO(contents))
                image.verify()
                image = Image.open(io.BytesIO(contents))
            except (UnidentifiedImageError, OSError) as exc:
                raise HTTPException(status_code=400, detail="تعذر قراءة الصورة؛ تحقق من أن الملف صورة سليمة.") from exc
            page_texts.append(ocr_image(image))

        raw_text = "\n\n--- صفحة جديدة ---\n\n".join(t for t in page_texts if t.strip()).strip()
        if not raw_text:
            raise HTTPException(status_code=422, detail="لم يُستخرج نص. جرّب صورة أوضح أو ملفًا بدقة أعلى.")
        parsed = parse_document(raw_text)
        return {"filename": file.filename, "success": True, "parsed_data": parsed,
                "raw_text": raw_text, "page_count": len(page_texts), "ocr_version": APP_VERSION}
    except HTTPException:
        raise
    except pytesseract.TesseractNotFoundError as exc:
        raise HTTPException(status_code=503, detail="Tesseract غير مثبت أو غير متاح على الخادم.") from exc
    except Exception as exc:
        # Do not return stack traces or internal paths to clients.
        raise HTTPException(status_code=500, detail=f"حدث خطأ أثناء معالجة المستند ({type(exc).__name__}).") from exc
