import ffmpeg
import subprocess
from typing import Tuple
import numpy as np


class VideoFrameEncoder:
    """
    This function expects the input frame size and path
    size is (W,H)

    """

    def __init__(self, size: Tuple[int, int], path: str):
        args = (
            ffmpeg
            .input('pipe:', format='rawvideo', pix_fmt='rgb24', s=f"{size[0]}x{size[1]}")
            .output(path, pix_fmt="yuv420p", vcodec="hevc_videotoolbox")
            .overwrite_output()
            .compile()
        )
        self.subprocess = subprocess.Popen(args, stdin=subprocess.PIPE)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.close()

    """
    Expected frame shape is (W,H,3) dtype=uint8
    where 3 is RGB
    """

    def write_frame(self, frame: np.ndarray):
        self.subprocess.stdin.write(
            frame
            .astype(np.uint8)
            .tobytes()
        )

    def close(self):
        self.subprocess.stdin.close()
