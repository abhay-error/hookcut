"""Caption fonts: open-licensed (OFL / Apache) Google Fonts, downloaded on first use.

Only single-weight font files are listed. libass picks the default instance of
variable fonts, which is often the thinnest weight, so those are avoided.
``family`` must match the family name stored inside the font file, because
that is how the ASS subtitle file refers to it.
"""

from __future__ import annotations

import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from .config import home

#: Same files from two hosts; the CDN is tried first, GitHub is the fallback.
MIRRORS = (
    "https://cdn.jsdelivr.net/gh/google/fonts@main/",
    "https://raw.githubusercontent.com/google/fonts/main/",
)
TIMEOUT_S = 20


@dataclass(frozen=True)
class Font:
    id: str
    label: str
    family: str
    path: str  # in the google/fonts repo
    vibe: str
    #: ExtraBold-only files are registered as bold so libass picks them.
    bold: bool = False

    @property
    def filename(self) -> str:
        return self.path.rsplit("/", 1)[1]


FONTS: dict[str, Font] = {f.id: f for f in [
    Font("poppins", "Poppins", "Poppins", "ofl/poppins/Poppins-ExtraBold.ttf",
         "Clean & modern", True),
    Font("anton", "Anton", "Anton", "ofl/anton/Anton-Regular.ttf", "Tall & punchy"),
    Font("bebas", "Bebas Neue", "Bebas Neue", "ofl/bebasneue/BebasNeue-Regular.ttf", "Cinematic"),
    Font("kanit", "Kanit", "Kanit", "ofl/kanit/Kanit-ExtraBold.ttf", "Bold geometric", True),
    Font("barlow", "Barlow Condensed", "Barlow Condensed",
         "ofl/barlowcondensed/BarlowCondensed-ExtraBold.ttf", "Sporty", True),
    Font("archivo", "Archivo Black", "Archivo Black", "ofl/archivoblack/ArchivoBlack-Regular.ttf",
         "Heavy"),
    Font("rubikmono", "Rubik Mono", "Rubik Mono One",
         "ofl/rubikmonoone/RubikMonoOne-Regular.ttf", "Wide & loud"),
    Font("luckiest", "Luckiest Guy", "Luckiest Guy",
         "apache/luckiestguy/LuckiestGuy-Regular.ttf", "Playful"),
    Font("bangers", "Bangers", "Bangers", "ofl/bangers/Bangers-Regular.ttf", "Comic"),
    Font("titan", "Titan One", "Titan One", "ofl/titanone/TitanOne-Regular.ttf", "Rounded"),
    Font("lilita", "Lilita One", "Lilita One", "ofl/lilitaone/LilitaOne-Regular.ttf", "Friendly"),
    Font("marker", "Permanent Marker", "Permanent Marker",
         "apache/permanentmarker/PermanentMarker-Regular.ttf", "Handwritten"),
    Font("dmserif", "DM Serif", "DM Serif Display",
         "ofl/dmserifdisplay/DMSerifDisplay-Regular.ttf", "Elegant serif"),
]}

DEFAULT_FONT = "poppins"


class FontError(RuntimeError):
    pass


def get(font_id: str) -> Font:
    font = FONTS.get(font_id)
    if font is None:
        raise FontError(f"Unknown font '{font_id}'. Choose from: {', '.join(FONTS)}")
    return font


def fonts_dir() -> Path:
    return home() / "fonts"


def ensure(font_id: str) -> Path:
    """Return the local font file, downloading it the first time."""
    font = get(font_id)
    path = fonts_dir() / font.filename
    if path.exists() and path.stat().st_size > 1000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".part")
    errors = []
    for base in MIRRORS:
        try:
            with urllib.request.urlopen(base + font.path, timeout=TIMEOUT_S) as response, \
                    open(tmp, "wb") as out:
                shutil.copyfileobj(response, out)
            if tmp.stat().st_size > 1000:
                tmp.replace(path)
                return path
            errors.append(f"{base}: empty file")
        except OSError as exc:
            errors.append(f"{base}: {exc}")
        tmp.unlink(missing_ok=True)
    raise FontError(f"Could not download the {font.label} font ({'; '.join(errors)}).")
