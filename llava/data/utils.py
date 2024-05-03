import io
import pathlib
from io import BytesIO

from iopath.common.file_io import g_pathmgr
from pytorchvideo.data.encoded_video import EncodedVideo, select_video_class


class VILAEncodedVideo(EncodedVideo):
    @classmethod
    def from_bytesio(cls, file_path: str, decode_audio: bool = True, decoder: str = "pyav"):
        if isinstance(file_path, io.BytesIO):
            video_file = file_path
            file_path = "tmp.mp4"
        elif isinstance(file_path, str):
            # We read the file with PathManager so that we can read from remote uris.
            with g_pathmgr.open(file_path, "rb") as fh:
                video_file = io.BytesIO(fh.read())
        else:
            print(f"unsupported type {type(file_path)}")
        video_cls = select_video_class(decoder)
        return video_cls(video_file, pathlib.Path(file_path).name, decode_audio)
