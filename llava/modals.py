import os

__all__ = ["Modal", "Image", "Video"]


class Modal:
    pass


class File(Modal):
    EXTENSIONS = None

    def __init__(self, path: str) -> None:
        self.path = path
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
        if self.EXTENSIONS is not None and not any(path.endswith(ext) for ext in self.EXTENSIONS):
            raise ValueError(f"Unsupported file extension: {os.path.splitext(path)[1]}")


class Image(File):
    EXTENSIONS = [".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp", ".mp4", ".mov", ".avi", ".mkv", ".webm"]


class Video(File):
    EXTENSIONS = [".mp4"]
