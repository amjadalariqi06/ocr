from fastapi import FastAPI, UploadFile, File, HTTPException
import fitz  # PyMuPDF
import pytesseract
from PIL import Image
import io

app = FastAPI(title="Umrah Track OCR API", version="1.0")

@app.get("/")
def home():
    return {"status": "Running", "message": "Umrah Track OCR Service is active!"}

@app.post("/extract-document/")
async def extract_document(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        extracted_text = ""
        
        # التحقق إذا كان الملف PDF
        if file.filename.endswith(".pdf"):
            doc = fitz.open(stream=file_bytes, filetype="pdf")
            for page_num, page in enumerate(doc):
                text = page.get_text("text")
                if len(text.strip()) > 30:
                    extracted_text += f"\n[PDF Text - Page {page_num + 1}]\n" + text
                else:
                    # تحويل صفحة الـ PDF إلى صورة لاستخدام OCR
                    pix = page.get_pixmap(dpi=300)
                    img = Image.open(io.BytesIO(pix.tobytes("png")))
                    ocr_text = pytesseract.image_to_string(img, config='--oem 3 --psm 6')
                    extracted_text += f"\n[PDF OCR - Page {page_num + 1}]\n" + ocr_text
                    
        # التحقق إذا كان الملف صورة (JPG / PNG)
        elif file.filename.lower().endswith((".png", ".jpg", ".jpeg")):
            image = Image.open(io.BytesIO(file_bytes))
            ocr_text = pytesseract.image_to_string(image, config='--oem 3 --psm 6')
            extracted_text += f"\n[Image OCR]\n" + ocr_text
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format. Use PDF or Images.")

        return {
            "filename": file.filename,
            "success": True,
            "extracted_text": extracted_text
        }

    except Exception as e:
        return {"success": False, "error": str(e)}