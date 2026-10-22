from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io
import re

app = FastAPI(title="Umrah Track Universal OCR API", version="2.0")

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
    return {"status": "Running", "message": "Universal Umrah Track OCR is active!"}

def parse_universal_document(text: str):
    """
    دالة ذكية للتعامل مع أي نموذج مستند (تذكرة طيران، تذكرة محلية، أو تأشيرة رسمية)
    """
    data = {
        "full_name": "غير متاح",
        "passport_number": "غير متاح",
        "reference_number": "غير متاح", # رقم التذكرة أو رقم التأشيرة
        "date": "غير متاح",
        "document_type": "مستند غير معروف"
    }
    
    # 1. تحديد ما إذا كان المستند عبارة عن تذكرة طيران دولية (مثل اليمنية)
    if "Yemenia" in text or "PASSENGER NAME" in text or "TICKET NUMBER" in text:
        data["document_type"] = "تذكرة طيران دولية (Yemenia)"
        
        # استخراج الاسم
        name_match = re.search(r'PASSENGER NAME[:\s]+([^\n\r]+)', text)
        if name_match:
            data["full_name"] = name_match.group(1).strip()
            
        # استخراج رقم التذكرة
        ticket_match = re.search(r'TICKET NUMBER[:\s]+([0-9\s]+)', text)
        if ticket_match:
            data["reference_number"] = ticket_match.group(1).strip().replace(" ", "")
            
        # استخراج التاريخ
        date_match = re.search(r'(\d{2}\.[A-Za-z]{3}\.\d{4})', text)
        if date_match:
            data["date"] = date_match.group(1)

    # 2. تحديد ما إذا كان المستند تذكرة محلية أو برية (مثل سماء النخبة / البركة)
    elif "البركة" in text or "سماء النخبة" in text or "رقم التذكرة" in text:
        data["document_type"] = "تذكرة محلية / برية"
        
        name_match = re.search(r'الاسم\s*[:\-]?\s*([^\n\r]+)', text)
        if name_match:
            data["full_name"] = name_match.group(1).strip()
            
        pass_match = re.search(r'رقم الجواز\s*[:\-]?\s*([A-Za-z0-9]+)', text)
        if pass_match:
            data["passport_number"] = pass_match.group(1).strip()
            
        ticket_match = re.search(r'رقم التذكرة\s*[:\-]?\s*([0-9]+)', text)
        if ticket_match:
            data["reference_number"] = ticket_match.group(1).strip()
            
        date_match = re.search(r'(\d{4}/\d{2}/\d{2})', text)
        if date_match:
            data["date"] = date_match.group(1)

    # 3. تحديد ما إذا كان المستند تأشيرة عمرة رسمية (منصة التأشيرات السعودية)
    elif "تأشيرة السعودية" in text or "Visa No" in text or "Umrah Operator" in text:
        data["document_type"] = "تأشيرة عمرة رسمية"
        
        name_match = re.search(r'(?:Name|الاسم)\s*[:\-]?\s*([^\n\r]+)', text)
        if name_match:
            clean_name = name_match.group(1).split('/')[0].strip()
            data["full_name"] = clean_name
            
        pass_match = re.search(r'(?:Passport No\.|رقم جواز السفر)\s*[:\-]?\s*([A-Za-z0-9]+)', text)
        if pass_match:
            data["passport_number"] = pass_match.group(1).strip()
            
        visa_match = re.search(r'(?:Visa No\.|رقم التأشيرة)\s*[:\-]?\s*([0-9]+)', text)
        if visa_match:
            data["reference_number"] = visa_match.group(1).strip()
            
        date_match = re.search(r'(\d{2}/\d{2}/\d{4})', text)
        if date_match:
            data["date"] = date_match.group(1)

    # 4. البحث العام الاحتياطي (Fallback) في حال وجود بيانات بارزة لم يلتقطها التصنيف المحدد
    if data["passport_number"] == "غير متاح":
        mrz_match = re.search(r'\b([A-Z0-9]{8,9})<\b', text)
        if mrz_match:
            data["passport_number"] = mrz_match.group(1)

    return data

@app.post("/extract-document/")
async def extract_document(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        extracted_text = ""
        
        # قراءة ملفات الـ PDF (رقمية أو مسحوبة كصور)
        if file.filename.endswith(".pdf"):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if len(text.strip()) > 30:
                    extracted_text += f"\n{text}\n"
                else:
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_text = pytesseract.image_to_string(img, config='--oem 3 --psm 6')
                    extracted_text += f"\n{ocr_text}\n"
                    
        # قراءة الصور العادية
        elif file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(io.BytesIO(file_bytes))
            ocr_text = pytesseract.image_to_string(image, config='--oem 3 --psm 6')
            extracted_text += f"\n{ocr_text}\n"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")

        # تحليل النص المستخرج بناءً على نوع المستند
        parsed_result = parse_universal_document(extracted_text)

        return {
            "filename": file.filename,
            "success": True,
            "parsed_data": parsed_result,
            "raw_text": extracted_text
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
