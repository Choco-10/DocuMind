from app.loaders.pdf import load_pdf
from app.loaders.image import load_image

ALLOWED_LOADERS = {
    ".pdf": load_pdf,
    ".png": load_image,
    ".jpg": load_image,
    ".jpeg": load_image,
    ".bmp": load_image,
    ".tiff": load_image,
    ".tif": load_image,
    ".webp": load_image,
}


def load_document(file_path: str) -> str:
    import os

    ext = os.path.splitext(file_path)[1].lower()
    loader = ALLOWED_LOADERS.get(ext)

    if loader is None:
        raise ValueError(f"Unsupported file extension: {ext}")

    return loader(file_path)