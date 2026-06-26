from PIL import Image
import pytesseract

def load_image(file_path: str) -> str:
    with Image.open(file_path) as img:

        if img.mode in ("RGBA", "P"):
            img = img.convert("RGB")
        ocr_text = pytesseract.image_to_string(img)

    return ocr_text.strip()