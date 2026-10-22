import io
import os
import re
import logging
from pathlib import Path
from typing import Optional, List, Dict

import fitz  # PyMuPDF
import pytesseract
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, ImageFilter, UnidentifiedImageError

APP_VERSION = "4.0.0"
MAX_FILE_BYTES = int(os.getenv("MAX_FILE_BYTES", str(15 * 1024 * 1024)))
MAX_PDF_PAGES = int(os.getenv("MAX_PDF_PAGES", "30"))
OCR_DPI = int(os.getenv("OCR_DPI", "300"))
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
    """تحويل الأرقام العربية والفارسية إلى أرقام إنجليزية قياسية."""
    table = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")
    return text.translate(table)


def normalize_text(text: str) -> str:
    text = normalize_digits(text or "")
    text = re.sub(r"[\u200e\u200f\u202a-\u202e\x00]", "", text)
    text = text.replace("：", ":").replace("‐", "-").replace("‑", "-").replace("–", "-").replace("—", "-")
    lines = [re.sub(r"[ \t]+", " ", line).strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


# الكلمات التي تمثل عناوين حقول ولا يجوز اعتبارها اسماً أو رقم جواز
INVALID_NAME_TOKENS = {
    "name", "full name", "passenger name", "passenger", "first name", "last name",
    "passport", "passport no", "passport number", "visa", "visa no", "visa number",
    "nationality", "sex", "gender", "birth", "date", "issue", "expiry", "operator",
    "الاسم", "الاسم الكامل", "اسم المعتمر", "اسم المسافر", "المسافر", "المعتمر",
    "رقم الجواز", "رقم جواز السفر", "جواز السفر", "رقم التأشيرة", "التأشيرة",
    "الجنسية", "الجنس", "تاريخ الميلاد", "تاريخ الإصدار", "تاريخ السفر", "الوكيل",
}

INVALID_PASSPORT_WORDS = {
    "PASSPORT", "NUMBER", "NORMAL", "ORDINARY", "DIPLOMATIC", "OFFICIAL",
    "YEMEN", "SAUDI", "ARABIA", "REPUBLIC", "ISSUED", "EXPIRY", "HOLDER",
    "SINGLE", "MULTIPLE", "UMRAH", "TICKET", "BOOKING", "STATUS",
}

# الفواصل التي تدل على بداية حقل جديد في نفس السطر
FIELD_STOP_PATTERN = re.compile(
    r"\s+(?=(?:Passport\s*(?:No|Number)?|Visa\s*(?:No|Number)?|Ticket\s*(?:No|Number)?|"
    r"Booking|Reference|Nationality|Date\s+of|Issue\s+Date|Birth\s+Date|Gender|Sex|"
    r"رقم\s*الجواز|رقم\s*جواز|رقم\s*التأشيرة|رقم\s*التذكرة|رقم\s*الحجز|"
    r"تاريخ\s*الإصدار|تاريخ\s*الميلاد|تاريخ\s*السفر|الجنسية|الجنس|المهنة|الوكيل)\b)",
    re.IGNORECASE,
)


def clean_candidate_name(raw_val: str) -> Optional[str]:
    if not raw_val:
        return None
    # قطع أي حقل آخر ملتصق في نفس السطر
    val = FIELD_STOP_PATTERN.split(raw_val)[0]
    # إزالة العناوين المتبقية مثل "/ الاسم" أو ": Name"
    val = re.sub(
        r"^(?:Full\s+Name|Name\s+of\s+Passenger|Passenger\s+Name|Name|"
        r"الاسم\s*الكامل|اسم\s*المعتمر|اسم\s*المسافر|الاسم)\b[\s:/\-|]*",
        "",
        val,
        flags=re.IGNORECASE,
    )
    val = re.sub(r"[\d<>:;|\[\]()_]+", " ", val)
    val = re.sub(r"\s+", " ", val).strip(" ./-")

    if not val or len(val) < 3 or len(val) > 85:
        return None
    if val.lower() in INVALID_NAME_TOKENS:
        return None
    # يجب أن يحتوي الاسم على حروف عربية أو إنجليزية كافية
    letters = re.findall(r"[A-Za-z\u0621-\u064A]", val)
    if len(letters) < 3:
        return None
    return val


def is_valid_passport(val: str) -> bool:
    if not val:
        return False
    clean = re.sub(r"[\s\-]", "", val).upper()
    if clean in INVALID_PASSPORT_WORDS:
        return False
    # رقم الجواز يتكون من 6 إلى 10 خانات (حرف أو حرفين على الأكثر والباقي أرقام، أو أرقام بالكامل)
    if not re.fullmatch(r"[A-Z]{0,2}\d{6,9}", clean):
        return False
    return True


def extract_mrz_data(text: str) -> Dict[str, str]:
    """استخراج الاسم ورقم الجواز من الشريط المشفر MRZ إن وجد في التأشيرة أو الجواز."""
    result = {}
    lines = [re.sub(r"\s+", "", line) for line in text.splitlines() if "<<<" in line]
    if not lines:
        return result

    for i, line in enumerate(lines):
        # البحث عن سطر الاسم في MRZ مثل: P<YEMALARIQI<<AMGED<AHMED<<<<<
        name_mrz = re.match(r"^[PVIACG][<A-Z][A-Z]{3}([A-Z]+)<<([A-Z<]+)$", line)
        if name_mrz:
            surname = name_mrz.group(1).replace("<", " ").strip()
            given = name_mrz.group(2).replace("<", " ").strip()
            full = f"{given} {surname}".strip()
            if len(full) >= 4:
                result["full_name"] = re.sub(r"\s+", " ", full)
            # السطر التالي في MRZ يحتوي على رقم الجواز في أول 9 خانات
            if i + 1 < len(lines):
                next_line = lines[i + 1]
                pass_match = re.match(r"^([A-Z0-9<]{9})\d[A-Z]{3}\d{6}", next_line)
                if pass_match:
                    cand = pass_match.group(1).replace("<", "")
                    if is_valid_passport(cand):
                        result["passport_number"] = cand
    return result


def extract_labeled_value(lines: List[str], label_regex: str, validator=None) -> Optional[str]:
    """البحث عن قيمة حقل في نفس السطر بعد العنوان، أو في السطر التالي مباشرة إذا كان العنوان منفصلاً."""
    pattern = re.compile(label_regex, re.IGNORECASE)
    for idx, line in enumerate(lines):
        match = pattern.search(line)
        if not match:
            continue
        # 1. فحص بقية السطر الحالي بعد العنوان
        remainder = line[match.end():].strip(" :\t-/|")
        if remainder:
            cand = validator(remainder) if validator else remainder
            if cand:
                return cand
        # 2. إذا كان السطر يحتوي على العنوان فقط (مثل Name / الاسم)، نفحص السطر التالي
        if idx + 1 < len(lines):
            next_line = lines[idx + 1].strip(" :\t-/|")
            cand = validator(next_line) if validator else next_line
            if cand:
                return cand
    return None


def extract_passport_from_line(raw_val: str) -> Optional[str]:
    tokens = re.findall(r"\b[A-Za-z0-9]{6,10}\b", raw_val)
    for token in tokens:
        if is_valid_passport(token):
            return token.upper()
    return None


def extract_date_smart(lines: List[str]) -> str:
    """استخراج تاريخ الإصدار أو السفر مع تجاهل تاريخ الميلاد وتاريخ انتهاء الجواز."""
    birth_or_expiry = re.compile(
        r"(?:Birth|DOB|Expiry|Expiration|الميلاد|ميلاد|الانتهاء|انتهاء)", re.IGNORECASE
    )
    preferred_date_label = re.compile(
        r"(?:Issue\s*Date|Date\s*of\s*Issue|Valid\s*From|Flight\s*Date|Departure|"
        r"تاريخ\s*الإصدار|تاريخ\s*بداية\s*الصلاحية|صالح\s*من|تاريخ\s*السفر|تاريخ\s*الرحلة|موعد\s*الرحلة|التاريخ)",
        re.IGNORECASE,
    )
    date_Formats = re.compile(
        r"\b(\d{1,2}\.[A-Za-z]{3}\.\d{4}|\d{1,2}\s+[A-Za-z]{3}\s+\d{4}|"
        r"\d{2}[./-]\d{2}[./-]\d{4}|\d{4}[./-]\d{2}[./-]\d{2})\b"
    )

    # المرحلة 1: البحث في الأسطر التي تحمل عنوان تاريخ إصدار أو سفر صريح
    for idx, line in enumerate(lines):
        if preferred_date_label.search(line) and not birth_or_expiry.search(line):
            m = date_Formats.search(line)
            if m:
                return m.group(1)
            if idx + 1 < len(lines) and not birth_or_expiry.search(lines[idx + 1]):
                m_next = date_Formats.search(lines[idx + 1])
                if m_next:
                    return m_next.group(1)

    # المرحلة 2: البحث عن تاريخ لا يقع في سطر ميلاد/انتهاء ويكون حديثاً (بعد 2020)
    for line in lines:
        if birth_or_expiry.search(line):
            continue
        for m in date_Formats.finditer(line):
            d_str = m.group(1)
            year_match = re.search(r"\b(19\d{2}|20\d{2})\b", d_str)
            if year_match:
                year = int(year_match.group(1))
                if 2020 <= year <= 2035:
                    return d_str
            else:
                return d_str

    return "غير متاح"


def parse_universal_document(text: str):
    """استخراج ذكي ودقيق لحقول التأشيرات وتذاكر الطيران والتذاكر البرية."""
    text = normalize_text(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    data = {
        "full_name": "غير متاح",
        "passport_number": "غير متاح",
        "reference_number": "غير متاح",
        "date": "غير متاح",
        "extra1": "غير متاح",
        "document_type": "مستند غير معروف",
    }

    if not lines:
        return data

    lower_text = text.lower()
    is_visa = any(term in lower_text for term in (
        "visa no", "visa number", "umrah operator", "تأشيرة", "وزارة الحج",
        "المملكة العربية السعودية", "السعودية", "مستند تأشيرة", "nusuk"
    ))
    is_yemenia = any(term in lower_text for term in (
        "yemenia", "yemen airways", "passenger name", "ticket number", "booking reference", "635"
    )) and not is_visa
    is_local_ticket = any(term in text for term in (
        "البركة", "سماء النخبة", "رقم التذكرة", "رقم الجواز", "تذكرة سفر", "النقل الجماعي"
    )) and not is_visa and not is_yemenia

    # 1. فحص بيانات MRZ إن وجدت
    mrz_info = extract_mrz_data(text)

    if is_visa:
        data["document_type"] = "تأشيرة / مستند متعلق بالتأشيرة"

        name = extract_labeled_value(
            lines,
            r"(?:Full\s+Name|Name|الاسم\s*الكامل|اسم\s*المعتمر|الاسم)\b",
            validator=clean_candidate_name,
        )
        passport = extract_labeled_value(
            lines,
            r"(?:Passport\s*(?:No\.?|Number)|رقم\s*جواز\s*السفر|رقم\s*الجواز)\b",
            validator=extract_passport_from_line,
        )
        reference = extract_labeled_value(
            lines,
            r"(?:Visa\s*(?:No\.?|Number)|رقم\s*التأشيرة|رقم\s*المستند)\b",
            validator=lambda s: (re.search(r"\b(\d{8,15})\b", s) or [None, None])[1],
        )
        operator = extract_labeled_value(
            lines,
            r"(?:Umrah\s*Operator|شركة\s*العمرة|مكتب\s*العمرة|اسم\s*الشركة)\b",
            validator=lambda s: s[:80].strip() if len(s.strip()) > 3 else None,
        )
        if not operator:
            op_match = re.search(r"(شركة\s+[^\r\n:]{4,60})", text)
            if op_match:
                operator = op_match.group(1).strip()
        if operator:
            data["extra1"] = operator

    elif is_yemenia:
        data["document_type"] = "تذكرة طيران دولية (تصنيف آلي)"
        data["extra1"] = "الخطوط الجوية اليمنية (Yemenia)"

        name = extract_labeled_value(
            lines,
            r"(?:PASSENGER\s+NAME|NAME\s+OF\s+PASSENGER|NAME)\b",
            validator=clean_candidate_name,
        )
        passport = extract_labeled_value(
            lines,
            r"(?:PASSPORT\s*(?:NO\.?|NUMBER)|DOC\s*NO|رقم\s*الجواز)\b",
            validator=extract_passport_from_line,
        )
        reference = extract_labeled_value(
            lines,
            r"(?:TICKET\s*(?:NO\.?|NUMBER)|E-TICKET)\b",
            validator=lambda s: re.sub(r"[\s\-]", "", m.group(1))
            if (m := re.search(r"\b(\d[\d\s\-]{8,16}\d)\b", s))
            else None,
        )
        if not reference:
            yem_tkt = re.search(r"\b(635[\s\-]?\d{10})\b", text)
            if yem_tkt:
                reference = re.sub(r"[\s\-]", "", yem_tkt.group(1))

    elif is_local_ticket:
        data["document_type"] = "تذكرة محلية / برية (تصنيف آلي)"

        name = extract_labeled_value(
            lines,
            r"(?:الاسم\s*الكامل|اسم\s*المسافر|اسم\s*المعتمر|الاسم|المسافر)\b",
            validator=clean_candidate_name,
        )
        passport = extract_labeled_value(
            lines,
            r"(?:رقم\s*جواز\s*السفر|رقم\s*الجواز|جواز\s*السفر)\b",
            validator=extract_passport_from_line,
        )
        reference = extract_labeled_value(
            lines,
            r"(?:رقم\s*التذكرة|رقم\s*الحجز|رقم\s*السند)\b",
            validator=lambda s: (re.search(r"\b([A-Za-z0-9\-]{4,20})\b", s) or [None, None])[1],
        )
        agent = extract_labeled_value(
            lines,
            r"(?:الوكيل|مكتب\s*الإصدار|الفرع|شركة\s*النقل)\b",
            validator=lambda s: s[:60].strip() if len(s.strip()) > 2 else None,
        )
        if agent:
            data["extra1"] = agent

    else:
        name = extract_labeled_value(
            lines,
            r"(?:Full\s+Name|Passenger\s+Name|Name|الاسم\s*الكامل|اسم\s*المسافر|اسم\s*المعتمر|الاسم)\b",
            validator=clean_candidate_name,
        )
        passport = extract_labeled_value(
            lines,
            r"(?:Passport\s*(?:No\.?|Number)|رقم\s*جواز\s*السفر|رقم\s*الجواز)\b",
            validator=extract_passport_from_line,
        )
        reference = extract_labeled_value(
            lines,
            r"(?:Visa\s*(?:No\.?|Number)|Ticket\s*(?:No\.?|Number)|Booking|Reference|رقم\s*التأشيرة|رقم\s*التذكرة|رقم\s*الحجز)\b",
            validator=lambda s: (re.search(r"\b([A-Za-z0-9\-]{5,20})\b", s) or [None, None])[1],
        )

    # تعبئة النتائج مع الاستفادة من MRZ كمرجع داعم
    if name:
        data["full_name"] = name
    elif "full_name" in mrz_info:
        data["full_name"] = mrz_info["full_name"]

    if passport:
        data["passport_number"] = passport
    elif "passport_number" in mrz_info:
        data["passport_number"] = mrz_info["passport_number"]
    else:
        # بحث احتياطي محكم عن رقم الجواز بشرط ألا يتطابق مع رقم التأشيرة
        for m in re.finditer(r"\b([A-Z]{1,2}\d{6,8}|\d{8,9})\b", text):
            cand = m.group(1)
            if is_valid_passport(cand) and cand != reference:
                data["passport_number"] = cand
                break

    if reference:
        data["reference_number"] = re.sub(r"\s+", "", reference)
    else:
        ref_fallback = re.search(r"\b(6\d{9,12})\b", text)
        if ref_fallback:
            data["reference_number"] = ref_fallback.group(1)

    data["date"] = extract_date_smart(lines)

    return data


def extract_visual_lines_from_pdf_page(page: fitz.Page) -> str:
    """إعادة بناء الأسطر بصرياً حسب إحداثيات (Y, X) لمنع تداخل أعمدة الجداول في PDF."""
    words = page.get_text("words")
    if not words:
        return ""

    # ترتيب الكلمات حسب الإحداثي العمودي y0 ثم الأفقي x0
    words_sorted = sorted(words, key=lambda w: (round(w[1] / 4.0) * 4, w[0]))
    lines: List[List[tuple]] = []
    current_line: List[tuple] = []
    current_y: Optional[float] = None

    for w in words_sorted:
        y0 = w[1]
        if current_y is None or abs(y0 - current_y) <= 5.0:
            current_line.append(w)
            if current_y is None:
                current_y = y0
        else:
            lines.append(sorted(current_line, key=lambda item: item[0]))
            current_line = [w]
            current_y = y0

    if current_line:
        lines.append(sorted(current_line, key=lambda item: item[0]))

    reconstructed = [" ".join(w[4] for w in line) for line in lines]
    sorted_text = page.get_text("text", sort=True) or ""
    return "\n".join(reconstructed) + "\n" + sorted_text


def preprocess_for_ocr(image: Image.Image) -> Image.Image:
    """تحسين أبعاد وتباين الصورة لرفع دقة قراءة Tesseract للعربية والإنجليزية."""
    image = ImageOps.exif_transpose(image).convert("RGB")
    if image.width < 1800:
        scale = min(2.5, 1800 / max(image.width, 1))
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS,
        )
    gray = ImageOps.grayscale(image)
    contrasted = ImageOps.autocontrast(gray, cutoff=1)
    sharpened = contrasted.filter(ImageFilter.SHARPEN)
    return sharpened


def ocr_image(image: Image.Image) -> str:
    processed = preprocess_for_ocr(image)
    # استخدام psm 4 للحفاظ على بنية الجداول والأعمدة، ثم دمج psm 6 عند الحاجة
    try:
        text_psm4 = pytesseract.image_to_string(processed, lang=OCR_LANG, config="--oem 3 --psm 4")
        if len(text_psm4.strip()) < 40:
            text_psm6 = pytesseract.image_to_string(processed, lang=OCR_LANG, config="--oem 3 --psm 6")
            return f"{text_psm4}\n{text_psm6}".strip()
        return text_psm4
    except pytesseract.TesseractError as exc:
        logger.warning("Configured OCR language unavailable; retrying English OCR.")
        try:
            return pytesseract.image_to_string(processed, lang="eng", config="--oem 3 --psm 4")
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
            raise HTTPException(
                status_code=413,
                detail=f"حجم الملف يتجاوز الحد المسموح ({MAX_FILE_BYTES // (1024 * 1024)} MB).",
            )
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
                        visual_text = extract_visual_lines_from_pdf_page(page)
                        preliminary_check = parse_universal_document(visual_text)

                        # إذا كان النص الرقمي قصيراً أو لم يستخرج الحقول الأساسية (كأن تكون البيانات داخل صورة مدمجة)، نشغل OCR أيضاً
                        needs_ocr = (
                            len(visual_text.strip()) < 80
                            or (
                                preliminary_check["full_name"] == "غير متاح"
                                and preliminary_check["passport_number"] == "غير متاح"
                                and preliminary_check["reference_number"] == "غير متاح"
                            )
                        )
                        if visual_text.strip():
                            extracted_parts.append(visual_text)
                        if needs_ocr:
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
    logger.exception("Unhandled API error on %s", request.url.path)
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=500, content={
        "success": False,
        "error": "حدث خطأ داخلي أثناء معالجة المستند. حاول مرة أخرى أو راجع سجلات الخادم."
    })
