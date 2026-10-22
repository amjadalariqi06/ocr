import io
import os
import re
import logging
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pytesseract
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

APP_VERSION = "3.0.0"
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(15 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "30"))
OCR_DPI = int(os.getenv("OCR_DPI", "250"))
OCR_LANG = os.getenv("OCR_LANG", "ara+eng")
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("umrah_track_ocr")

app = FastAPI(
    title="Umrah Track Pro OCR API",
    version=APP_VERSION,
    description="OCR and field extraction for travel documents. Extracted data is not document authentication.",
)

origins_setting = os.getenv("ALLOWED_ORIGINS", "*").strip()
allowed_origins = ["*"] if origins_setting == "*" else [
    origin.strip().rstrip("/") for origin in origins_setting.split(",") if origin.strip()
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Accept"],
)


@app.get("/")
def home():
    return {
        "status": "ok",
        "service": "Umrah Track Pro OCR API",
        "version": APP_VERSION,
        "message": "OCR API is running. Extraction results require human review.",
    }


@app.get("/health")
def health():
    return {"status": "healthy", "version": APP_VERSION}


def normalize_digits(text: str) -> str:
    """Convert Arabic/Persian digits to ASCII digits for reliable number matching."""
    table = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
    return text.translate(table)


def normalize_text(text: str) -> str:
    text = normalize_digits(text or "")
    text = text.replace("\u200e", "").replace("\u200f", "")
    text = text.replace("\u202a", "").replace("\u202b", "").replace("\u202c", "")
    text = text.replace("：", ":").replace("‐", "-").replace("‑", "-").replace("–", "-")
    return text.replace("\x00", "").strip()


def clean_text(text: str) -> str:
    return re.sub(r"[ \t]+", " ", normalize_text(text)).strip()


def first_match(patterns, text: str, flags=re.IGNORECASE | re.MULTILINE) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            value = re.sub(r"\s+", " ", match.group(1)).strip(" \t:;|")
            if value:
                return value
    return None


def parse_universal_document(text: str):
    """Heuristic field extraction. It does not validate a document's authenticity."""
    text = normalize_text(text)
    data = {
        "full_name": "غير متاح",
        "passport_number": "غير متاح",
        "reference_number": "غير متاح",
        "date": "غير متاح",
        "document_type": "مستند غير معروف",
    }

    # Classify visas before tickets because some documents can contain generic English labels.
    is_visa = any(term in text.lower() for term in (
        "visa no", "visa number", "umrah operator", "تأشيرة", "وزارة الحج",
        "المملكة العربية السعودية", "السعودية"
    ))
    is_yemenia = any(term in text.lower() for term in (
        "yemenia", "yemen airways", "passenger name", "ticket number", "booking reference"
    ))
    is_local_ticket = any(term in text for term in (
        "البركة", "سماء النخبة", "رقم التذكرة", "رقم الجواز"
    ))

    if is_visa:
        data["document_type"] = "تأشيرة / مستند متعلق بالتأشيرة"
        name = first_match([
            r"(?:Full\s+Name|Name|الاسم الكامل|اسم المعتمر|الاسم)\s*[:\-]?\s*([^\r\n]+)"
        ], text)
        passport = first_match([
            r"(?:Passport\s*(?:No\.?|Number)|رقم جواز السفر|رقم الجواز)\s*[:\-]?\s*([A-Z0-9]{5,15})"
        ], text)
        reference = first_match([
            r"(?:Visa\s*(?:No\.?|Number)|رقم التأشيرة)\s*[:\-]?\s*([0-9]{5,20})"
        ], text)
        date = first_match([
            r"(?:Issue Date|Date of Issue|تاريخ الإصدار|تاريخ بداية الصلاحية)\s*[:\-]?\s*(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})",
            r"\b(\d{2}/\d{2}/\d{4})\b",
        ], text)
    elif is_yemenia:
        data["document_type"] = "تذكرة طيران دولية (تصنيف آلي)"
        name = first_match([
            r"PASSENGER\s+NAME\s*[:\-]?\s*([A-Z][A-Z /'-]{2,})",
            r"NAME\s*OF\s*PASSENGER\s*[:\-]?\s*([^\r\n]+)"
        ], text)
        passport = first_match([
            r"(?:PASSPORT\s*(?:NO\.?|NUMBER)|رقم الجواز)\s*[:\-]?\s*([A-Z0-9]{5,15})"
        ], text)
        reference = first_match([
            r"TICKET\s*NUMBER\s*[:\-]?\s*([0-9 ]{8,20})",
            r"\b(635[0-9 ]{7,15})\b"
        ], text)
        date = first_match([
            r"\b(\d{1,2}\.[A-Za-z]{3}\.\d{4})\b",
            r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b"
        ], text)
    elif is_local_ticket:
        data["document_type"] = "تذكرة محلية / برية (تصنيف آلي)"
        name = first_match([
            r"(?:الاسم الكامل|اسم المسافر|الاسم)\s*[:\-]?\s*([^\r\n]+)"
        ], text)
        passport = first_match([
            r"(?:رقم جواز السفر|رقم الجواز)\s*[:\-]?\s*([A-Z0-9]{5,15})"
        ], text)
        reference = first_match([
            r"(?:رقم التذكرة|رقم الحجز)\s*[:\-]?\s*([A-Z0-9-]{4,20})"
        ], text)
        date = first_match([
            r"(?:التاريخ|تاريخ السفر|موعد الرحلة)\s*[:\-]?\s*(\d{1,4}[/-]\d{1,2}[/-]\d{1,4})",
            r"\b(\d{4}/\d{2}/\d{2})\b"
        ], text)
    else:
        name = first_match([
            r"(?:Full\s+Name|Passenger\s+Name|Name|الاسم الكامل|اسم المسافر|الاسم)\s*[:\-]?\s*([^\r\n]+)"
        ], text)
        passport = first_match([
            r"(?:Passport\s*(?:No\.?|Number)|رقم جواز السفر|رقم الجواز)\s*[:\-]?\s*([A-Z0-9]{5,15})"
        ], text)
        reference = first_match([
            r"(?:Reference|Booking|Ticket|رقم التذكرة|رقم الحجز)\s*(?:No\.?|Number)?\s*[:\-]?\s*([A-Z0-9-]{5,20})"
        ], text)
        date = first_match([
            r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b",
            r"\b(\d{4}/\d{2}/\d{2})\b"
        ], text)

    if name:
        name = re.sub(r"\s{2,}", " ", name).strip(" :-|")
        if len(name) <= 100 and not re.fullmatch(r"[\d\s./-]+", name):
            data["full_name"] = name
    if passport:
        data["passport_number"] = re.sub(r"\s+", "", passport).upper()
    if reference:
        data["reference_number"] = re.sub(r"\s+", "", reference)
    if date:
        data["date"] = date

    # Conservative fallback only when labelled fields were not found.
    if data["passport_number"] == "غير متاح":
        mrz_pass = re.search(r"\b([A-Z]{1,2}\d{6,8})\b", text)
        if mrz_pass:
            data["passport_number"] = mrz_pass.group(1)
    if data["reference_number"] == "غير متاح":
        ref_match = re.search(r"\b(6\d{10,13})\b", text)
        if ref_match:
            data["reference_number"] = ref_match.group(1)

    return data


def ocr_image(image: Image.Image) -> str:
    image = ImageOps.exif_transpose(image).convert("RGB")
    # Enlarge small images to help OCR without using aggressive thresholding.
    if image.width < 1400:
        scale = min(2.0, 1400 / max(image.width, 1))
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    try:
        return pytesseract.image_to_string(image, lang=OCR_LANG, config="--oem 3 --psm 6")
    except pytesseract.TesseractError as exc:
        # Some environments may not have Arabic trained data installed.
        logger.warning("Configured OCR language unavailable; retrying English OCR.")
        try:
            return pytesseract.image_to_string(image, lang="eng", config="--oem 3 --psm 6")
        except pytesseract.TesseractError:
            raise HTTPException(
                status_code=503,
                detail="محرك OCR أو ملفات اللغة غير متاحة على الخادم. تحقق من تثبيت Tesseract وبيانات ara/eng."
            ) from exc


async def read_limited_upload(file: UploadFile) -> bytes:
    chunks = []
    total = 0
    while True:
        chunk = await file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > MAX_FILE_BYTES:
            raise HTTPException(status_code=413, detail=f"حجم الملف يتجاوز الحد المسموح ({MAX_FILE_BYTES // (1024 * 1024)} MB).")
        chunks.append(chunk)
    if not total:
        raise HTTPException(status_code=400, detail="الملف فارغ.")
    return b"".join(chunks)


@app.post("/extract-document/")
async def extract_document(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name
    suffix = Path(filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(status_code=415, detail="نوع الملف غير مدعوم. ارفع PDF أو PNG أو JPG/JPEG.")

    extracted_parts = []
    try:
        file_bytes = await read_limited_upload(file)
        if suffix == ".pdf":
            if not file_bytes.startswith(b"%PDF-"):
                raise HTTPException(status_code=400, detail="الملف لا يبدو مستند PDF صالحًا.")
            try:
                with fitz.open(stream=file_bytes, filetype="pdf") as doc:
                    if doc.is_encrypted:
                        raise HTTPException(status_code=400, detail="ملفات PDF المحمية بكلمة مرور غير مدعومة.")
                    if len(doc) == 0:
                        raise HTTPException(status_code=400, detail="ملف PDF لا يحتوي على صفحات.")
                    if len(doc) > MAX_PDF_PAGES:
                        raise HTTPException(status_code=413, detail=f"عدد صفحات PDF يتجاوز الحد ({MAX_PDF_PAGES} صفحة).")
                    for page in doc:
                        page_text = page.get_text("text") or ""
                        if len(page_text.strip()) > 30:
                            extracted_parts.append(page_text)
                        else:
                            pix = page.get_pixmap(dpi=OCR_DPI, alpha=False)
                            image = Image.open(io.BytesIO(pix.tobytes("png")))
                            extracted_parts.append(ocr_image(image))
            except HTTPException:
                raise
            except Exception as exc:
                logger.info("PDF parsing failed: %s", type(exc).__name__)
                raise HTTPException(status_code=400, detail="تعذر قراءة PDF. تأكد أن الملف سليم وغير تالف.") from exc
        else:
            try:
                with Image.open(io.BytesIO(file_bytes)) as image:
                    image.verify()
                with Image.open(io.BytesIO(file_bytes)) as image:
                    extracted_parts.append(ocr_image(image))
            except UnidentifiedImageError as exc:
                raise HTTPException(status_code=400, detail="الصورة غير صالحة أو تنسيقها غير مدعوم.") from exc
            except HTTPException:
                raise
            except Exception as exc:
                logger.info("Image parsing failed: %s", type(exc).__name__)
                raise HTTPException(status_code=400, detail="تعذر قراءة الصورة. جرّب حفظها بصيغة PNG أو JPG.") from exc

        extracted_text = "\n".join(part for part in extracted_parts if part).strip()
        if not extracted_text:
            return {
                "filename": filename,
                "success": True,
                "parsed_data": parse_universal_document(""),
                "raw_text": "",
                "extracted_text": "",
                "warning": "لم يتم استخراج نص واضح. جرّب صورة أوضح أو راجع المستند يدويًا."
            }

        parsed_result = parse_universal_document(extracted_text)
        return {
            "filename": filename,
            "success": True,
            "parsed_data": parsed_result,
            "raw_text": extracted_text,
            "extracted_text": extracted_text,
            "warning": "النتيجة استخراج آلي وليست تحققًا من صحة الوثيقة. راجع البيانات يدويًا."
        }
    finally:
        await file.close()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    # Avoid returning internal exception details, file contents, or PII to the client.
    logger.exception("Unhandled API error on %s", request.url.path)
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={
        "success": False,
        "error": "حدث خطأ داخلي أثناء معالجة المستند. حاول مرة أخرى أو راجع سجلات الخادم."
    })
