from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware  # <-- أضف هذه المكتبة
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

app = FastAPI(title="Umrah Track OCR API", version="1.0")

# تفعيل الـ CORS للسماح بالاتصال من أي صفحة ويب
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # السماح لكل النطاقات
    allow_credentials=True,
    allow_methods=["*"],  # السماح بكل الطرق (GET, POST, etc.)
    allow_headers=["*"],  # السماح بكل الهيدر
)

@app.get("/")
def home():
    return {"status": "Running", "message": "Umrah Track OCR Service is active!"}

@app.post("/extract-document/")
async def extract_document(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        extracted_text = ""
        
        if file.filename.endswith(".pdf"):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if len(text.strip()) > 30:
                    extracted_text += f"\n[PDF Text - Page {page_num + 1}]\n" + text
                else:
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_text = pytesseract.image_to_string(img, config='--oem 3 --psm 6')
                    extracted_text += f"\n[PDF OCR - Page {page_num + 1}]\n" + ocr_text
                    
        elif file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(io.BytesIO(file_bytes))
            ocr_text = pytesseract.image_to_string(image, config='--oem 3 --psm 6')
            extracted_text += f"\n[Image OCR]\n" + ocr_text
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format.")

        return {
            "filename": file.filename,
            "success": True,
            "extracted_text": extracted_text
        }

    except Exception as e:
        return {"success": False, "error": str(e)}
