import os
import shutil
import tempfile
from typing import Any, Callable

import ffmpeg
from PIL import Image as PillowImage


def process_video(
    path: str,
    dest: str,
    factory: Any,
    fast: bool = False,
    console: Any = None,
) -> bool:
    """
    Process a video file frame-by-frame and apply the factory's palette.

    Args:
        path (str): Input video path.
        dest (str): Output video path.
        factory (Any): The GruvboxFactory instance.
        fast (bool): Whether to use fast quantization.
        console (Any): Rich console for printing (optional).

    Returns:
        bool: True if success, False otherwise.
    """
    try:
        probe = ffmpeg.probe(path)
        video_info = next(s for s in probe["streams"] if s["codec_type"] == "video")
        width = int(video_info["width"])
        height = int(video_info["height"])
        avg_frame_rate = video_info["avg_frame_rate"]
        if "/" in avg_frame_rate:
            n, d = avg_frame_rate.split("/")
            fps = float(n) / float(d)
        else:
            fps = float(avg_frame_rate)

        # Use a temporary file for the video without audio first
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
            tmp_path = tmp.name

        # ffmpeg input: suppress logs with quiet=True
        process = (
            ffmpeg.input(path)
            .output("pipe:", format="rawvideo", pix_fmt="rgb24")
            .run_async(pipe_stdout=True, quiet=True)
        )

        # ffmpeg output
        out_process = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                s=f"{width}x{height}",
                framerate=fps,
            )
            .output(tmp_path, pix_fmt="yuv420p", vcodec="libx264")
            .overwrite_output()
            .run_async(pipe_stdin=True, quiet=True)
        )

        while True:
            in_bytes = process.stdout.read(width * height * 3)
            if not in_bytes:
                break

            # Convert bytes to PIL Image
            frame = PillowImage.frombytes("RGB", (width, height), in_bytes)
            
            # Convert using factory
            if fast:
                new_frame = factory.quantize_image(frame)
            else:
                new_frame = factory.convert_image(frame, parallel_threading=True)
            
            # Write back to pipe
            out_process.stdin.write(new_frame.tobytes())

        process.wait()
        out_process.stdin.close()
        out_process.wait()

        # Try to add audio back if it exists
        has_audio = any(s["codec_type"] == "audio" for s in probe["streams"])
        if has_audio:
            final_video = ffmpeg.output(
                ffmpeg.input(tmp_path),
                ffmpeg.input(path).audio,
                dest,
                vcodec="copy",
                acodec="copy",
            ).overwrite_output()
            final_video.run(quiet=True)
            os.remove(tmp_path)
        else:
            if os.path.exists(dest):
                os.remove(dest)
            shutil.move(tmp_path, dest)

        return True

    except Exception as e:
        if console:
            console.print(f"[red]Failed to convert video: {e}[/]")
        return False
