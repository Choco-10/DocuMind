import pdfplumber
from PIL import Image
import pytesseract

def load_pdf(file_path: str) -> str:
    text_chunks = []

    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                text_chunks.append(page_text)
            else:
                im = page.to_image(resolution=300)
                pil_image = im.original
                ocr_text = pytesseract.image_to_string(pil_image)
                text_chunks.append(ocr_text)

    return "\n".join(text_chunks)
