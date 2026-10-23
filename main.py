import io
import os
import re
import logging
from pathlib import Path
from typing import Optional

import fitz  # PyMuPDF
import pytesseract
from fastapi import FastAPI, UploadFile, File, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

APP_VERSION = "3.2.0"
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(15 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "10"))
OCR_DPI = int(os.getenv("OCR_DPI", "220"))
OCR_LANG = os.getenv("OCR_LANG", "ara+eng")
REVERSE_ARABIC_DEFAULT = os.getenv("REVERSE_ARABIC_DEFAULT", "true").lower() in ("1", "true", "yes")
ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg"}

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
logger = logging.getLogger("umrah_track_ocr")

app = FastAPI(
    title="VANT // Umrah Track Pro OCR API",
    version=APP_VERSION,
    description="Smart OCR and field extraction for Umrah visas, passports, and travel tickets.",
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
        "service": "VANT // Umrah Track Pro OCR API",
        "version": APP_VERSION,
        "message": "OCR API is running with server-side Arabic name, company, and nationality normalization.",
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


def has_arabic(text: str) -> bool:
    return bool(re.search(r"[\u0600-\u06FF]", text or ""))


def first_match(patterns, text: str, flags=re.IGNORECASE | re.MULTILINE) -> Optional[str]:
    for pattern in patterns:
        match = re.search(pattern, text, flags)
        if match:
            value = re.sub(r"\s+", " ", match.group(1)).strip(" \t:;|")
            if value:
                return value
    return None


# ---------------------------------------------------------------------------
# Smart Arabic Cleaning & Reversal Helpers
# ---------------------------------------------------------------------------

NAME_NOISE_PATTERNS = [
    r"^(?:الاسم\s+الكامل|الإسم\s+الكامل|اسم\s+المعتمر|اسم\s+المسافر|اسم\s+الحاج|الاسم\s+رباعي|الاسم|الإسم|السم|إسم|اسم|Full\s*Name|Passenger\s*Name|Name\s*of\s*Passenger|Name)\s*[:\-–—_|/\\]*\s*",
    r"\s*[:\-–—_|/\\]*\s*(?:الاسم\s+الكامل|الإسم\s+الكامل|اسم\s+المعتمر|اسم\s+المسافر|اسم\s+الحاج|الاسم\s+رباعي|الاسم|الإسم|السم|إسم|اسم|Full\s*Name|Passenger\s*Name|Name\s*of\s*Passenger|Name)$",
]

NAME_STOP_TOKENS = {"السم", "الاسم", "الإسم", "اسم", "إسم", "name", "fullname"}


def strip_name_noise(raw: str) -> str:
    s = re.sub(r"\s+", " ", (raw or "")).strip(" :;|,-_/")
    for _ in range(3):
        for pat in NAME_NOISE_PATTERNS:
            s = re.sub(pat, "", s, flags=re.IGNORECASE).strip(" :;|,-_/")
    tokens = [
        t for t in s.split(" ")
        if t and re.sub(r"[:\-–—_|/\\.,]", "", t).lower() not in NAME_STOP_TOKENS
    ]
    return " ".join(tokens).strip()


def clean_and_fix_arabic_name(raw_name: str, reverse_arabic: bool = True) -> str:
    if not raw_name or raw_name == "غير متاح":
        return "غير متاح"

    cleaned = strip_name_noise(raw_name)
    if not cleaned:
        return "غير متاح"

    # Reverse word order only when the extracted name is Arabic (fixes PDF/OCR RTL word-order inversion)
    if reverse_arabic and has_arabic(cleaned):
        words = [w for w in cleaned.split(" ") if w]
        if len(words) > 1:
            cleaned = " ".join(reversed(words))
        cleaned = strip_name_noise(cleaned)

    return cleaned if cleaned else "غير متاح"


COMPANY_NOISE_PATTERNS = [
    r"^(?:شركة\s+العمرة|مؤسسة\s+العمرة|اسم\s+الشركة|الوكيل\s+الخارجي|الوكيل\s+المحلي|الوكيل|الجهة|شركة|مؤسسة|Umrah\s*Operator|Operator|Agent|Sponsor|Company|Carrier)\s*[:\-–—_|/\\]*\s*",
    r"\s*[:\-–—_|/\\]*\s*(?:شركة\s+العمرة|مؤسسة\s+العمرة|اسم\s+الشركة|الوكيل\s+الخارجي|الوكيل\s+المحلي|الوكيل|الجهة|Umrah\s*Operator|Operator|Agent|Sponsor|Company|Carrier)$",
    r"^(?:الاسم|الإسم|السم|إسم|اسم)\s*[:\-–—_|/\\]*\s*",
    r"\s*[:\-–—_|/\\]*\s*(?:الاسم|الإسم|السم|إسم|اسم)$",
]


def strip_company_noise(raw: str) -> str:
    s = re.sub(r"\s+", " ", (raw or "")).strip(" :;|,-_/")
    for _ in range(3):
        for pat in COMPANY_NOISE_PATTERNS:
            s = re.sub(pat, "", s, flags=re.IGNORECASE).strip(" :;|,-_/")
    return s.strip()


def clean_and_fix_company_name(raw_company: str, reverse_arabic: bool = True) -> str:
    if not raw_company or raw_company == "غير متاح":
        return "غير متاح"

    cleaned = strip_company_noise(raw_company)
    if not cleaned:
        return "غير متاح"

    if reverse_arabic and has_arabic(cleaned):
        words = [w for w in cleaned.split(" ") if w]
        if len(words) > 1:
            cleaned = " ".join(reversed(words))
        cleaned = strip_company_noise(cleaned)

    return cleaned if cleaned else "غير متاح"


# ---------------------------------------------------------------------------
# Nationality Extraction Engine
# ---------------------------------------------------------------------------

NATIONALITY_PATTERNS = [
    (r"(?:يمني|اليمن|الجمهورية\s+اليمنية|YEMENI|YEMEN|REPUBLIC\s+OF\s+YEMEN|\bYEM\b)", "اليمن"),
    (r"(?:مصري|مصر|جمهورية\s+مصر|EGYPTIAN|EGYPT|\bEGY\b)", "مصر"),
    (r"(?:سوداني|السودان|SUDANESE|SUDAN|\bSDN\b)", "السودان"),
    (r"(?:أردني|اردني|الأردن|الاردن|JORDANIAN|JORDAN|\bJOR\b)", "الأردن"),
    (r"(?:سوري|سوريا|الجمهورية\s+العربية\s+السورية|SYRIAN|SYRIA|\bSYR\b)", "سوريا"),
    (r"(?:عراقي|العراق|IRAQI|IRAQ|\bIRQ\b)", "العراق"),
    (r"(?:فلسطيني|فلسطين|PALESTINIAN|PALESTINE|\bPSE\b|\bPLT\b)", "فلسطين"),
    (r"(?:لبناني|لبنان|LEBANESE|LEBANON|\bLBN\b)", "لبنان"),
    (r"(?:مغربي|المغرب|MOROCCAN|MOROCCO|\bMAR\b)", "المغرب"),
    (r"(?:جزائري|الجزائر|ALGERIAN|ALGERIA|\bDZA\b|\bALG\b)", "الجزائر"),
    (r"(?:تونسي|تونس|TUNISIAN|TUNISIA|\bTUN\b)", "تونس"),
    (r"(?:ليبي|ليبيا|LIBYAN|LIBYA|\bLBY\b)", "ليبيا"),
    (r"(?:عماني|سلطنة\s+عمان|OMANI|OMAN|\bOMN\b)", "عمان"),
    (r"(?:كويتي|الكويت|KUWAITI|KUWAIT|\bKWT\b)", "الكويت"),
    (r"(?:إماراتي|اماراتي|الإمارات|الامارات|EMIRATI|UAE|\bARE\b)", "الإمارات"),
    (r"(?:قطري|قطر|QATARI|QATAR|\bQAT\b)", "قطر"),
    (r"(?:بحريني|البحرين|BAHRAINI|BAHRAIN|\bBHR\b)", "البحرين"),
    (r"(?:باكستاني|باكستان|PAKISTANI|PAKISTAN|\bPAK\b)", "باكستان"),
    (r"(?:هندي|الهند|INDIAN|INDIA|\bIND\b)", "الهند"),
    (r"(?:بنغلاديشي|بنغلاديش|بنجلاديش|BANGLADESHI|BANGLADESH|\bBGD\b)", "بنغلاديش"),
    (r"(?:إندونيسي|اندونيسي|إندونيسيا|اندونيسيا|INDONESIAN|INDONESIA|\bIDN\b)", "إندونيسيا"),
    (r"(?:ماليزي|ماليزيا|MALAYSIAN|MALAYSIA|\bMYS\b)", "ماليزيا"),
    (r"(?:تركي|تركيا|TURKISH|TURKEY|TÜRKIYE|\bTUR\b)", "تركيا"),
    (r"(?:صومالي|الصومال|SOMALI|SOMALIA|\bSOM\b)", "الصومال"),
    (r"(?:جيبوتي|DJIBOUTI|\bDJI\b)", "جيبوتي"),
    (r"(?:موريتاني|موريتانيا|MAURITANIAN|MAURITANIA|\bMRT\b)", "موريتانيا"),
]


def normalize_country_label(raw_val: str) -> Optional[str]:
    if not raw_val:
        return None
    cleaned = re.sub(
        r"(?:الجنسية|جنسية|Nationality|Nat\.?|Citizenship|Country)\s*[:\-–—_|/\\]*",
        "",
        raw_val,
        flags=re.IGNORECASE,
    ).strip(" :;|,-_/")
    if not cleaned:
        return None

    for pattern, canonical in NATIONALITY_PATTERNS:
        if re.search(pattern, cleaned, re.IGNORECASE):
            return canonical

    if has_arabic(cleaned):
        words = [w for w in cleaned.split(" ") if w]
        if 1 < len(words) <= 4:
            reversed_try = " ".join(reversed(words))
            for pattern, canonical in NATIONALITY_PATTERNS:
                if re.search(pattern, reversed_try, re.IGNORECASE):
                    return canonical

    return cleaned[:40]


def extract_nationality(text: str) -> str:
    if not text:
        return "غير متاح"

    # 1. Explicit labelled field
    labelled = first_match([
        r"(?:Nationality|Nat\.?|Citizenship|الجنسية|جنسية\s+المعتمر|جنسية)\s*[:\-]?\s*([^\r\n|:]{2,35})"
    ], text)
    if labelled:
        normalized = normalize_country_label(labelled)
        if normalized:
            return normalized

    # 2. MRZ line detection (e.g. P<YEM... or 3-letter country code in MRZ)
    mrz_match = re.search(r"P<([A-Z]{3})", text)
    if mrz_match:
        normalized = normalize_country_label(mrz_match.group(1))
        if normalized:
            return normalized

    # 3. Direct keyword scan across the document
    for pattern, canonical in NATIONALITY_PATTERNS:
        if re.search(pattern, text, re.IGNORECASE):
            return canonical

    return "غير متاح"


def extract_company_or_agent(text: str, is_visa: bool, is_yemenia: bool, is_local_ticket: bool, reverse_arabic: bool = True) -> str:
    raw_company = first_match([
        r"(?:شركة\s+العمرة|مؤسسة\s+العمرة|الوكيل\s+الخارجي|الوكيل\s+المحلي|اسم\s+الشركة|اسم\s+الوكيل|Umrah\s+Operator|Operator\s+Name|Agent\s+Name|Sponsor|Company)\s*[:\-]?\s*([^\r\n|:]{3,80})"
    ], text)

    if raw_company:
        fixed = clean_and_fix_company_name(raw_company, reverse_arabic=reverse_arabic)
        if fixed != "غير متاح":
            return fixed

    # Secondary search for lines containing "شركة ... للعمرة" or "وكالة ... للسفريات"
    line_match = first_match([
        r"((?:شركة|مؤسسة|وكالة)\s+[^\r\n|:]{3,65})"
    ], text)
    if line_match:
        fixed = clean_and_fix_company_name(line_match, reverse_arabic=reverse_arabic)
        if fixed != "غير متاح":
            return fixed

    if is_yemenia:
        return "الخطوط الجوية اليمنية (Yemenia)"
    if is_local_ticket:
        if "البركة" in text:
            return "شركة البركة للنقل"
        if "سماء النخبة" in text:
            return "سماء النخبة للسفريات"

    return "غير متاح"


# ---------------------------------------------------------------------------
# Universal Document Parser
# ---------------------------------------------------------------------------

def parse_universal_document(text: str, reverse_arabic: bool = True):
    """Heuristic field extraction with Arabic word-order correction and nationality detection."""
    text = normalize_text(text)
    data = {
        "full_name": "غير متاح",
        "passport_number": "غير متاح",
        "reference_number": "غير متاح",
        "date": "غير متاح",
        "nationality": "غير متاح",
        "extra1": "غير متاح",
        "document_type": "مستند غير معروف",
    }

    if not text:
        return data

    is_visa = any(term in text.lower() for term in (
        "visa no", "visa number", "umrah operator", "تأشيرة", "وزارة الحج",
        "المملكة العربية السعودية", "السعودية", "مقام", "نسك", "nusuk"
    ))
    is_yemenia = any(term in text.lower() for term in (
        "yemenia", "yemen airways", "passenger name", "ticket number", "booking reference"
    ))
    is_local_ticket = any(term in text for term in (
        "البركة", "سماء النخبة", "رقم التذكرة", "رقم الجواز"
    ))

    if is_visa:
        data["document_type"] = "تأشيرة عمرة"
        name = first_match([
            r"(?:Full\s+Name|Name|الاسم\s+الكامل|الإسم\s+الكامل|اسم\s+المعتمر|الاسم|السم)\s*[:\-]?\s*([^\r\n]+)"
        ], text)
        passport = first_match([
            r"(?:Passport\s*(?:No\.?|Number)|رقم\s+جواز\s+السفر|رقم\s+الجواز)\s*[:\-]?\s*([A-Z0-9]{5,15})"
        ], text)
        reference = first_match([
            r"(?:Visa\s*(?:No\.?|Number)|رقم\s+التأشيرة|رقم\s+الفيزا)\s*[:\-]?\s*([0-9]{5,20})"
        ], text)
        date = first_match([
            r"(?:Issue\s+Date|Date\s+of\s+Issue|تاريخ\s+الإصدار|تاريخ\s+بداية\s+الصلاحية)\s*[:\-]?\s*(\d{1,4}[./-]\d{1,2}[./-]\d{1,4})",
            r"\b(\d{2}/\d{2}/\d{4})\b",
            r"\b(\d{4}[/-]\d{2}[/-]\d{2})\b",
        ], text)
    elif is_yemenia:
        data["document_type"] = "تذكرة طيران دولية"
        name = first_match([
            r"PASSENGER\s+NAME\s*[:\-]?\s*([A-Z][A-Z /'-]{2,})",
            r"NAME\s*OF\s*PASSENGER\s*[:\-]?\s*([^\r\n]+)"
        ], text)
        passport = first_match([
            r"(?:PASSPORT\s*(?:NO\.?|NUMBER)|رقم\s+الجواز)\s*[:\-]?\s*([A-Z0-9]{5,15})"
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
        data["document_type"] = "تذكرة نقل / سفر"
        name = first_match([
            r"(?:الاسم\s+الكامل|اسم\s+المسافر|الاسم|السم)\s*[:\-]?\s*([^\r\n]+)"
        ], text)
        passport = first_match([
            r"(?:رقم\s+جواز\s+السفر|رقم\s+الجواز)\s*[:\-]?\s*([A-Z0-9]{5,15})"
        ], text)
        reference = first_match([
            r"(?:رقم\s+التذكرة|رقم\s+الحجز)\s*[:\-]?\s*([A-Z0-9-]{4,20})"
        ], text)
        date = first_match([
            r"(?:التاريخ|تاريخ\s+السفر|موعد\s+الرحلة)\s*[:\-]?\s*(\d{1,4}[/-]\d{1,2}[/-]\d{1,4})",
            r"\b(\d{4}/\d{2}/\d{2})\b"
        ], text)
    else:
        name = first_match([
            r"(?:Full\s+Name|Passenger\s+Name|Name|الاسم\s+الكامل|اسم\s+المسافر|الاسم|السم)\s*[:\-]?\s*([^\r\n]+)"
        ], text)
        passport = first_match([
            r"(?:Passport\s*(?:No\.?|Number)|رقم\s+جواز\s+السفر|رقم\s+الجواز)\s*[:\-]?\s*([A-Z0-9]{5,15})"
        ], text)
        reference = first_match([
            r"(?:Reference|Booking|Ticket|رقم\s+التذكرة|رقم\s+الحجز)\s*(?:No\.?|Number)?\s*[:\-]?\s*([A-Z0-9-]{5,20})"
        ], text)
        date = first_match([
            r"\b(\d{1,2}[./-]\d{1,2}[./-]\d{2,4})\b",
            r"\b(\d{4}/\d{2}/\d{2})\b"
        ], text)

    # Fallback when Arabic PDF layout places "الاسم" or "السم" at the end of the line
    if not name:
        rtl_line_name = first_match([
            r"([^\r\n:0-9]{4,70})\s+(?:الاسم\s+الكامل|اسم\s+المعتمر|الاسم|السم)\b"
        ], text)
        if rtl_line_name:
            name = rtl_line_name

    if name:
        name = re.sub(r"\s{2,}", " ", name).strip(" :-|")
        if len(name) <= 100 and not re.fullmatch(r"[\d\s./-]+", name):
            data["full_name"] = clean_and_fix_arabic_name(name, reverse_arabic=reverse_arabic)

    if passport:
        data["passport_number"] = re.sub(r"\s+", "", passport).upper()
    if reference:
        data["reference_number"] = re.sub(r"\s+", "", reference)
    if date:
        data["date"] = date

    # Conservative fallback for passport & visa/ticket reference numbers
    if data["passport_number"] == "غير متاح":
        mrz_pass = re.search(r"\b([A-Z]{1,2}\d{6,8})\b", text)
        if mrz_pass:
            data["passport_number"] = mrz_pass.group(1)
    if data["reference_number"] == "غير متاح":
        ref_match = re.search(r"\b(6\d{10,13})\b", text)
        if ref_match:
            data["reference_number"] = ref_match.group(1)

    # Extract Nationality & Umrah Company / Agent
    data["nationality"] = extract_nationality(text)
    data["extra1"] = extract_company_or_agent(
        text,
        is_visa=is_visa,
        is_yemenia=is_yemenia,
        is_local_ticket=is_local_ticket,
        reverse_arabic=reverse_arabic,
    )

    return data


def ocr_image(image: Image.Image) -> str:
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image.width < 1400:
        scale = min(2.0, 1400 / max(image.width, 1))
        image = image.resize((int(image.width * scale), int(image.height * scale)))
    try:
        return pytesseract.image_to_string(image, lang=OCR_LANG, config="--oem 3 --psm 6")
    except pytesseract.TesseractError as exc:
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
async def extract_document(
    file: UploadFile = File(...),
    reverse_arabic: bool = Query(default=REVERSE_ARABIC_DEFAULT, description="عكس ترتيب الكلمات العربية المعكوسة تلقائياً")
):
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
                            pix = None
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
                "server_normalized": True,
                "parsed_data": parse_universal_document("", reverse_arabic=reverse_arabic),
                "raw_text": "",
                "extracted_text": "",
                "warning": "لم يتم استخراج نص واضح. جرّب صورة أوضح أو راجع المستند يدويًا."
            }

        parsed_result = parse_universal_document(extracted_text, reverse_arabic=reverse_arabic)
        return {
            "filename": filename,
            "success": True,
            "server_normalized": True,
            "parsed_data": parsed_result,
            "raw_text": extracted_text,
            "extracted_text": extracted_text,
            "warning": "النتيجة استخراج آلي وليست تحققًا من صحة الوثيقة. راجع البيانات يدويًا."
        }
    finally:
        await file.close()


@app.exception_handler(Exception)
async def unhandled_exception_handler(request, exc):
    logger.exception("Unhandled API error on %s", request.url.path)
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={
        "success": False,
        "error": "حدث خطأ داخلي أثناء معالجة المستند. حاول مرة أخرى أو راجع سجلات الخادم."
    })
