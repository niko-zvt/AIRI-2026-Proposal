"""
md_to_pdf.py

CLI:
    python md_to_pdf.py [<markdown-path>] [--css <css-path>] [--out <pdf-path>] [-v]
"""

from __future__ import annotations

import argparse
import logging
import sys
from html import escape
from pathlib import Path
from typing import Final

import markdown
from weasyprint import CSS, HTML

logger: logging.Logger = logging.getLogger("md_to_pdf")

_DEFAULT_REPORT_RELATIVE: Final[Path] = Path("..") / "Research Proposal" / "Zhivotenko-RP.md"
_DEFAULT_CSS_FILENAME: Final[str] = "style.css"

_MD_EXTENSIONS: Final[list[str]] = [
    "extra",        # tables, fenced_code, footnotes, attr_list, def_list, abbr, md_in_html
    "sane_lists",   # stricter list parsing
    "smarty",       # smart quotes / dashes
    "toc",          # heading id anchors
]


class MarkdownToPdfError(Exception):
    """Domain-specific error raised on any conversion failure."""


def _read_text(path: Path) -> str:
    """
    Read a UTF-8 text file and return its contents.

    Parameters
    ----------
    path : Path
        Path to the file to read.

    Returns
    -------
    str
        The file contents decoded as UTF-8.

    Raises
    ------
    MarkdownToPdfError
        If the file does not exist or cannot be read.
    """
    if not isinstance(path, Path):
        raise MarkdownToPdfError(f"Expected Path, got {type(path).__name__}")
    if not path.is_file():
        raise MarkdownToPdfError(f"File not found: {path}")
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise MarkdownToPdfError(f"Cannot read file {path}: {exc}") from exc


def _render_html(md_text: str, title: str) -> str:
    """
    Convert Markdown text to a complete HTML5 document.

    The body is wrapped in a `<main class="document">` element so the stylesheet
    can target the report container without restyling arbitrary other content.

    Parameters
    ----------
    md_text : str
        The Markdown source to convert.
    title : str
        Document title used in the HTML `<title>` element.

    Returns
    -------
    str
        A self-contained HTML5 document ready to be passed to WeasyPrint.

    Raises
    ------
    MarkdownToPdfError
        If the Markdown library fails to render the document.
    """
    if not isinstance(md_text, str):
        raise MarkdownToPdfError("Markdown content must be a string")
    if not isinstance(title, str) or not title.strip():
        raise MarkdownToPdfError("Document title must be a non-empty string")

    try:
        body_html: str = markdown.markdown(
            md_text,
            extensions=_MD_EXTENSIONS,
            output_format="html5",
        )
    except Exception as exc:
        raise MarkdownToPdfError(f"Markdown rendering failed: {exc}") from exc

    safe_title: str = escape(title, quote=True)
    lines: list[str] = [
        "<!DOCTYPE html>",
        '<html lang="ru">',
        "<head>",
        '<meta charset="utf-8" />',
        f"<title>{safe_title}</title>",
        "</head>",
        "<body>",
        '<main class="document">',
        body_html,
        "</main>",
        "</body>",
        "</html>",
    ]
    return "\n".join(lines)


def convert_markdown_to_pdf(md_path: Path, css_path: Path, out_path: Path) -> Path:
    """
    Convert a single Markdown file into a styled PDF document.

    Parameters
    ----------
    md_path : Path
        Path to the source Markdown file. Must exist and be readable.
    css_path : Path
        Path to the CSS stylesheet applied during PDF rendering.
    out_path : Path
        Destination path for the generated PDF. Parent directories are
        created if missing.

    Returns
    -------
    Path
        Absolute path of the written PDF file.

    Raises
    ------
    MarkdownToPdfError
        If any of the inputs are invalid or rendering/writing fails.
    """
    if not isinstance(md_path, Path) or not isinstance(css_path, Path) or not isinstance(out_path, Path):
        raise MarkdownToPdfError("All path arguments must be pathlib.Path instances")

    md_path = md_path.expanduser().resolve()
    css_path = css_path.expanduser().resolve()
    out_path = out_path.expanduser().resolve()

    logger.info("Reading Markdown source: %s", md_path)
    md_text: str = _read_text(md_path)

    if not css_path.is_file():
        raise MarkdownToPdfError(f"CSS file not found: {css_path}")
    logger.info("Using stylesheet: %s", css_path)

    html_document: str = _render_html(md_text, title=md_path.stem)

    # WeasyPrint resolves relative URLs (e.g. <img src="./Images/foo.png">)
    # against the supplied base_url. Trailing slash makes Path semantics explicit.
    base_url: str = md_path.parent.as_uri() + "/"
    logger.debug("HTML base URL for asset resolution: %s", base_url)

    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        HTML(string=html_document, base_url=base_url).write_pdf(
            target=str(out_path),
            stylesheets=[CSS(filename=str(css_path))],
        )
    except MarkdownToPdfError:
        raise
    except Exception as exc:
        raise MarkdownToPdfError(f"Failed to render PDF: {exc}") from exc

    logger.info("PDF written: %s", out_path)
    return out_path


def _build_arg_parser() -> argparse.ArgumentParser:
    """
    Build the command-line argument parser used by `main`.

    Returns
    -------
    argparse.ArgumentParser
        Configured argument parser instance.
    """
    parser = argparse.ArgumentParser(
        description="Convert a Markdown report into a styled PDF using WeasyPrint.",
    )
    parser.add_argument(
        "markdown",
        nargs="?",
        type=Path,
        default=None,
        help=(
            "Path to the Markdown source. Defaults to "
            "'../Research Proposal/Zhivotenko-RP.md' relative to this script."
        ),
    )
    parser.add_argument(
        "--css",
        type=Path,
        default=None,
        help="Path to the CSS stylesheet. Defaults to './style.css' next to this script.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Output PDF path. Defaults to the input file with the '.pdf' suffix.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable DEBUG-level logging.",
    )
    return parser


def _configure_logging(verbose: bool) -> None:
    """
    Configure the root logger for CLI usage.

    Parameters
    ----------
    verbose : bool
        If True, emit DEBUG messages; otherwise INFO and above.
    """
    if not isinstance(verbose, bool):
        raise MarkdownToPdfError("verbose flag must be a bool")

    logging.basicConfig(
        level=logging.DEBUG if verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
    )


def main(argv: list[str] | None = None) -> int:
    """
    Command-line entry point.

    Parameters
    ----------
    argv : list[str] | None
        Optional argument list (mainly for tests). Defaults to `sys.argv[1:]`.

    Returns
    -------
    int
        Process exit code: 0 on success, 1 on conversion failure.
    """
    parser: argparse.ArgumentParser = _build_arg_parser()
    args = parser.parse_args(argv)

    _configure_logging(args.verbose)

    script_dir: Path = Path(__file__).resolve().parent

    md_path: Path = (
        args.markdown if args.markdown is not None else script_dir / _DEFAULT_REPORT_RELATIVE
    )
    css_path: Path = (
        args.css if args.css is not None else script_dir / _DEFAULT_CSS_FILENAME
    )
    out_path: Path = (
        args.out if args.out is not None else md_path.with_suffix(".pdf")
    )

    try:
        convert_markdown_to_pdf(md_path, css_path, out_path)
    except MarkdownToPdfError:
        logger.exception("Conversion failed")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
