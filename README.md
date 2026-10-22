# OCR Document Extraction API (v3)

## What changed
- Uses image enhancement and multiple Tesseract page-segmentation modes.
- Prefers Arabic + English OCR when both language packs are installed.
- Reads selectable PDF text and OCRs scanned pages.
- Extracts values only when a field label supports them; no arbitrary long-number fallback.
- Normalizes Arabic/Persian digits and common punctuation.
- Adds upload size/page limits and clearer HTTP errors.
- Frontend displays server-parsed fields instead of running a second, conflicting regex parser.
- Includes `/health` endpoint for checking Tesseract language availability.

## Run locally
```bash
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000
```
Tesseract and its Arabic/English language data must also be installed. On Render, `apt.packages` installs these packages where supported by the build environment.

## Important limitations
OCR cannot guarantee perfect extraction on every scan, font, layout, or low-quality image. This version intentionally returns `غير متاح` when a field cannot be linked reliably to its label rather than guessing. Always compare extracted values with the original document. Classification is not verification of document authenticity.

CORS is permissive for ease of deployment; restrict `allow_origins` to your actual frontend domain before using real personal documents. The app does not store uploaded files, and does not implement a central passenger registry or automatic notification scheduler.
