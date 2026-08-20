import argparse
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from signal import SIGINT, Signals, signal
from types import FrameType
from typing import Any, Literal

import ImageGoNord
from factory.video import process_video
from pick import pick
from PIL import ImageSequence
from PIL import Image as PillowImage
from PIL.Image import Image as PilImage
from PIL.ImageFile import ImageFile
from rich import console, panel

Palette = Literal["pink", "white", "mix"]
Image = str
Images = list[Image]


@dataclass
class Arguments:
    """
    Attributes:
        palette (Palette | None): The selected palette value.
        input (list[Image]): List of input path strings.
        fast (bool): Whether to use fast quantization for video/gif.
    """

    palette: Palette | None
    input: list[Image] = field(default_factory=list[Image])
    fast: bool = False


class Parser(argparse.ArgumentParser):
    """
    A subclass of argparse.ArgumentParser that stores parsed arguments in an Arguments instance.
    """

    def __init__(self) -> None:
        """
        Initializes the Parser instance.

        Parameters: None.
        Raises: None.
        """
        super().__init__(
            description="A simple cli to manufacture Gruvbox themed wallpapers.",
            prefix_chars="-",
            argument_default=None,
            conflict_handler="error",
        )

        self.add_argument(
            "-p",
            "--palette",
            choices=["white", "pink", "mix"],
            nargs="?",
            default="pink",
            const="pink",
            help="choose your palette, panther 'pink' (default), snoopy 'white' or smooth 'mix'",
        )

        self.add_argument(
            "-i", "--input", nargs="+", type=str, help="path(s) to the image(s), gif(s) or video(s)."
        )

        self.add_argument(
            "-f",
            "--fast",
            action="store_true",
            help="use fast quantization for video/gif (recommended for videos, especially 4K).",
        )

        self._parsed_args: argparse.Namespace  # type: ignore
        self.arguments: Arguments  # type: ignore

    def parse(self) -> None:
        """
        Parses command-line arguments and stores them in the 'arguments' attribute.

        Sets:
            self._parsed_args (argparse.Namespace): Raw parsed args.
            self.arguments (Arguments): Parsed arguments converted to an Arguments instance.
        Returns: None.
        Raises: None.
        """
        self._parsed_args = self.parse_args()
        self.arguments = Arguments(
            palette=self._parsed_args.palette,
            input=self._parsed_args.input,
            fast=self._parsed_args.fast,
        )


def is_palette(value: str | None) -> bool:
    """
    Args:
        value (str | None): A palette value.
    Returns:
        bool: True if value is one of "pink", "white", or "mix"; otherwise False.
    """
    return value in {"pink", "white", "mix"}


def signal_handler(signum: int, _frame: FrameType | None = None) -> None:
    """
    Args:
        signum (int): The signal number.
        _frame (FrameType | None): The current stack frame (optional).
    Raises:
        SystemExit: Exits with code 2 if signum equals Signals.SIGINT.
    Returns: None.
    """
    if signum == Signals.SIGINT:
        sys.exit(2)


class Console(console.Console):
    """
    Subclass of rich.console.Console.
    """

    def __init__(self) -> None:
        """
        Initializes the Console instance.

        Parameters: None.
        Returns: None.
        """
        super().__init__()

    def print_title(self) -> None:
        """
        Prints the title panel.

        Returns:
            None.
        """
        self.print(
            panel.Panel(
                "🏭 [bold green] Gruvbox Factory [/] 🏭",
                expand=False,
                border_style="yellow",
            )
        )


def select_palette() -> Palette:
    """
    Returns:
        Palette: The selected palette from a TUI picker. If the selection is not valid,
                 returns "white".
    """
    prompt: str = (
        "🎨 [bold yellow]Palette (panther 'pink', snoopy 'white' or smooth 'mix'):[/] "
    )
    options: list[str] = ["pink", "white", "mix"]

    value: Any = pick(options, title=prompt, clear_screen=False)
    _, selection = value

    return selection if is_palette(selection) else "white"


@dataclass
class Factory(ImageGoNord.GoNord):
    """
    Subclass of ImageGoNord.GoNord for manufacturing images.

    Attributes:
        image_paths (list[str]): List of image file paths.
    """

    def __init__(self) -> None:
        """
        Initializes the Factory instance.

        Sets:
            self.image_paths to an empty list.
        Calls:
            self.reset_palette() to initialize the palette.
        Returns: None.
        """
        super().__init__()
        self.image_paths: list[str] = []
        self.reset_palette()


class GruvboxFactory:
    """
    Main class to process Gruvbox themed wallpapers.
    """

    def __init__(self) -> None:
        """
        Initializes GruvboxFactory.

        Registers:
            SIGINT signal with signal_handler.
        Creates:
            self.console (Console), self.parser (Parser), and self.factory (Factory).
        Returns: None.
        """
        _ = signal(SIGINT, signal_handler)
        self.console: Console = Console()
        self.parser: Parser = Parser()
        self.factory: Factory = Factory()
        self.fast: bool = False

    def get_palette(self) -> Palette:
        """
        Returns:
            Palette: The valid palette value from command-line arguments or via TUI.
        Raises:
            Exception: If no valid palette is determined (should be unreachable).
        """
        palette: Palette | None = self.parser.arguments.palette
        palette = palette if palette is not None else select_palette()
        if is_palette(palette):
            return palette
        raise Exception("This is unreachable.")

    def add_palette(self, palette: str) -> None:
        """
        Args:
            palette (str): The palette name.
        Reads:
            File "gruvbox-{palette}.txt" in the directory of this script.
        Raises:
            SystemExit: If the palette file is not found.
        Returns:
            None.
        """
        cwd: Path = Path(__file__).parent.absolute()
        path: Path = cwd / f"gruvbox-{palette}.txt"
        colors: list[str] = []

        try:
            with open(path) as fd:
                colors = fd.read().splitlines()
        except FileNotFoundError:
            self.console.print(f"❌ [red]Palette file {path} not found.[/]")
            sys.exit(1)

        for color in colors:
            self.factory.add_color_to_palette(color)

        return None

    def select_paths(self) -> list[str]:
        """
        Prompts the user to input image paths and validates each.

        Returns:
            list[str]: List of valid image file paths.
        """
        prompt: str = "🖼️ [bold yellow]Image paths (separated by spaces):[/] "
        user_input = self.console.input(prompt)
        paths: list[str] = []
        for raw_path in user_input.split():
            path = Path(os.path.expanduser(raw_path))
            if "*" in raw_path:
                paths.extend(
                    [str(p) for p in path.parent.glob(path.name) if p.is_file()]
                )
            elif path.exists() and path.is_file():
                paths.append(str(path))
            else:
                self.console.print(f"❌ [red]Skipping {raw_path} (not a valid file)[/]")
        return paths

    def write_image_color(self, path: str) -> None:
        """
        Args:
            path (str): The file path to an image.
        Performs:
            Opens the image, converts it using the factory, and saves the result.
        Returns:
            None.
        """
        parent = os.path.dirname(path)
        base = os.path.basename(path)
        dest = os.path.join(parent, f"gruvbox_{base}")

        self.console.print(f"🔨 [yellow]manufacturing '{base}' -> {dest}[/]")

        if path.lower().endswith((".mp4", ".mov", ".avi", ".mkv")):
            process_video(path, dest, self.factory, fast=self.fast, console=self.console)
        elif path.lower().endswith(".gif"):
            image = PillowImage.open(path)
            if not getattr(image, "is_animated", False):
                if self.fast:
                    self.factory.quantize_image(image, save_path=dest)
                else:
                    self.factory.convert_image(image, save_path=dest, parallel_threading=True)
            else:
                frames = []
                durations = []
                for frame in ImageSequence.Iterator(image):
                    durations.append(frame.info.get("duration", 100))
                    if self.fast:
                        new_frame = self.factory.quantize_image(frame.convert("RGB"))
                    else:
                        new_frame = self.factory.convert_image(frame.convert("RGB"), parallel_threading=True)
                    frames.append(new_frame)
                frames[0].save(dest, save_all=True, append_images=frames[1:], duration=durations, loop=image.info.get("loop", 0), optimize=False)
        else:
            image: PilImage | ImageFile = self.factory.open_image(path)
            if self.fast:
                self.factory.quantize_image(image, save_path=dest)
            else:
                self.factory.convert_image(image, save_path=dest, parallel_threading=True)

        self.console.print(f"✅ [bold green]Done![/] [green](saved to '{dest}')[/]")

    def process_images(self, images: Images) -> bool:
        """
        Args:
            images (list[str]): List of image file paths.
        Returns:
            bool: True if all images are processed successfully; False otherwise.
        """
        failed = 0
        passed = 0
        for path in images:
            if os.path.exists(path) and os.path.isfile(path):
                self.write_image_color(path)
                passed += 1
            else:
                self.console.print(f"❌ [red]Skipping {path} (file not found) [/]")
                failed += 1

        succeeded: bool = failed == 0
        if succeeded:
            self.console.print("🎉 [bold green]All images processed successfully![/]")
        elif failed and passed:
            self.console.print("🎉 [bold orange]Some images processed successfully![/]")
        else:
            self.console.print("🎉 [bold red]Couldn't process any images[/]")
            return False
        return True


def main() -> None:
    """
    Main function of the script.

    Performs:
        - Argument parsing via GruvboxFactory.parser.
        - Palette determination and palette file processing.
        - Image processing via GruvboxFactory.
    Raises:
        SystemExit: Exits with code from parser.print_help() if insufficient args,
                    or code 1 if image processing fails.
    Returns:
        None.
    """
    factory = GruvboxFactory()

    factory.parser.parse()
    inputs = factory.parser.arguments.input
    factory.fast = factory.parser.arguments.fast

    if len(sys.argv) < 2:
        sys.exit(factory.parser.print_help())

    palette = factory.get_palette()
    factory.add_palette(palette)

    if not inputs:
        inputs = factory.select_paths()

    if factory.process_images(inputs) is not True:
        sys.exit(1)

    sys.exit(0)


if __name__ == "__main__":
    main()
