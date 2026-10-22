from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re

app = FastAPI(title="Umrah Track Pro OCR API", version="2.5")

# تفعيل الـ CORS لتجنب أي مشاكل اتصال مع واجهة الويب
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def home():
    return {"status": "Running", "message": "Umrah Track Pro OCR & Parser is active!"}

def clean_text(text: str) -> str:
    """تنظيف النص وإزالة الفراغات الزائدة لتحسين دقة البحث"""
    return re.sub(r'\s+', ' ', text).strip()

def parse_universal_document(text: str):
    """
    محرك تحليل ذكي فائق الدقة يستخرج البيانات بغض النظر عن اختلاف تنسيق المستند
    """
    cleaned_text = clean_text(text)
    
    data = {
        "full_name": "غير متاح",
        "passport_number": "غير متاح",
        "reference_number": "غير متاح", # رقم التذكرة أو التأشيرة أو مرجع الحجز
        "date": "غير متاح",
        "document_type": "مستند غير معروف"
    }
    
    # 1. التعرف على تذاكر الطيران الدولية (مثل اليمنية)[cite: 5]
    if "Yemenia" in text or "PASSENGER NAME" in text or "TICKET NUMBER" in text:
        data["document_type"] = "تذكرة طيران دولية (Yemenia)"
        
        # استخراج الاسم بدقة
        name_match = re.search(r'PASSENGER NAME[:\s]+([A-Z\/\s]+?)(?:\s{2,}|BOOKING|TICKET|ISSUED|$)', text)
        if name_match:
            data["full_name"] = name_match.group(1).strip()
            
        # استخراج رقم التذكرة
        ticket_match = re.search(r'TICKET NUMBER[:\s]+([0-9\s]+)', text)
        if ticket_match:
            data["reference_number"] = ticket_match.group(1).replace(" ", "").strip()
            
        # استخراج التاريخ
        date_match = re.search(r'(\d{2}\.[A-Za-z]{3}\.\d{4})', text)
        if date_match:
            data["date"] = date_match.group(1)

    # 2. التعرف على التذاكر المحلية / البرية (مثل سماء النخبة / البركة)[cite: 6]
    elif "البركة" in text or "سماء النخبة" in text or "رقم التذكرة" in text or "رقم الجواز" in text:
        data["document_type"] = "تذكرة محلية / برية"
        
        # استخراج الاسم
        name_match = re.search(r'الاسم\s*[:\-]?\s*([^\n\r]+)', text)
        if name_match:
            data["full_name"] = name_match.group(1).strip()
            
        # استخراج رقم الجواز
        pass_match = re.search(r'رقم الجواز\s*[:\-]?\s*([A-Za-z0-9]+)', text)
        if pass_match:
            data["passport_number"] = pass_match.group(1).strip()
            
        # استخراج رقم التذكرة
        ticket_match = re.search(r'رقم التذكرة\s*[:\-]?\s*([0-9]+)', text)
        if ticket_match:
            data["reference_number"] = ticket_match.group(1).strip()
            
        # استخراج التاريخ
        date_match = re.search(r'(\d{4}/\d{2}/\d{2})', text)
        if date_match:
            data["date"] = date_match.group(1)

    # 3. التعرف على تأشيرات العمرة الرسمية (منصة التأشيرات السعودية)[cite: 7]
    elif "تأشيرة السعودية" in text or "Visa No" in text or "Umrah Operator" in text or "وزارة الحج" in text:
        data["document_type"] = "تأشيرة عمرة رسمية"
        
        # استخراج الاسم (إنجليزي أو عربي)
        name_match = re.search(r'(?:Name|الاسم)\s*[:\-]?\s*([^\n\r/]+)', text)
        if name_match:
            data["full_name"] = name_match.group(1).strip()
            
        # استخراج رقم الجواز من التأشيرة
        pass_match = re.search(r'(?:Passport No\.|رقم جواز السفر)\s*[:\-]?\s*([A-Za-z0-9]+)', text)
        if pass_match:
            data["passport_number"] = pass_match.group(1).strip()
            
        # استخراج رقم التأشيرة
        visa_match = re.search(r'(?:Visa No\.|رقم التأشيرة)\s*[:\-]?\s*([0-9]+)', text)
        if visa_match:
            data["reference_number"] = visa_match.group(1).strip()
            
        # استخراج التاريخ
        date_match = re.search(r'(\d{2}/\d{2}/\d{4})', text)
        if date_match:
            data["date"] = date_match.group(1)

    # --- أنظمة بحث احتياطية (Fallback Extractors) لضمان عدم ضياع أي حقل ---
    if data["passport_number"] == "غير متاح":
        # البحث عن شريط الـ MRZ أو رقم جواز يبدأ بحرفين وأرقام
        mrz_pass = re.search(r'\b([A-Z]{1,2}\d{6,8})\b', text)
        if mrz_pass:
            data["passport_number"] = mrz_pass.group(1)

    if data["reference_number"] == "غير متاح":
        # البحث عن أي رقم مرجعي أو تذكرة طويل
        ref_match = re.search(r'\b(6\d{10}|[0-9]{8,12})\b', text)
        if ref_match:
            data["reference_number"] = ref_match.group(1)

    return data

@app.post("/extract-document/")
async def extract_document(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        extracted_text = ""
        
        # قراءة ملفات الـ PDF (رقمية أو مسحوبة كصور Scan)
        if file.filename.endswith(".pdf"):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if len(text.strip()) > 30:
                    extracted_text += f"\n{text}\n"
                else:
                    # تحويل الصفحة لصورة واستخدام Tesseract لزيادة الدقة
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_text = pytesseract.image_to_string(img, config='--oem 3 --psm 6')
                    extracted_text += f"\n{ocr_text}\n"
                    
        # قراءة الصور العادية (PNG, JPG)
        elif file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(io.BytesIO(file_bytes))
            ocr_text = pytesseract.image_to_string(image, config='--oem 3 --psm 6')
            extracted_text += f"\n{ocr_text}\n"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")

        # تحليل وتدقيق البيانات المستخرجة
        parsed_result = parse_universal_document(extracted_text)

        return {
            "filename": file.filename,
            "success": True,
            "parsed_data": parsed_result,
            "raw_text": extracted_text
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
