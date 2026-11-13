"""Text from files the user attaches to a message: PDF, Word, Excel, PowerPoint and plain text or code.

Everything runs on this computer (pypdf for PDF; Word, Excel and PowerPoint files are zip archives of XML, read with
the standard library). Nothing is uploaded anywhere."""
from __future__ import annotations

import io
import re
import zipfile
from xml.etree import ElementTree as ET

TEXT_EXT = {"txt", "md", "markdown", "csv", "tsv", "json", "html", "htm", "xml", "yaml", "yml", "py", "js", "ts", "tsx",
            "jsx", "go", "rs", "java", "c", "cpp", "h", "hpp", "rb", "php", "sh", "sql", "css", "log", "ini", "toml", "swift",
            "kt", "cs", "lua", "r", "m", "ipynb", "env", "cfg", "conf"}
IMAGE_EXT = {"png", "jpg", "jpeg", "gif", "webp", "heic", "bmp", "tiff"}
W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
S = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
A = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


class Unsupported(ValueError):
    """The file type cannot be read (the message says why, for the user)."""


def extract_text(name: str, data: bytes) -> str:
    ext = name.rsplit(".", 1)[-1].lower() if "." in name else ""
    if ext == "pdf":
        return _pdf(data)
    if ext == "docx":
        return _docx(data)
    if ext == "xlsx":
        return _xlsx(data)
    if ext == "pptx":
        return _pptx(data)
    if ext in IMAGE_EXT:
        raise Unsupported("Pictures can't be read yet: your local model only reads text")
    if ext in ("doc", "xls", "ppt"):
        raise Unsupported(f"Old .{ext} files can't be read: save it as .{ext}x or PDF first")
    if b"\x00" in data[:2000]:
        raise Unsupported("This file is not text")
    return data.decode("utf-8", errors="replace")


def _pdf(data: bytes) -> str:
    from pypdf import PdfReader
    try:
        reader = PdfReader(io.BytesIO(data))
        if reader.is_encrypted:
            reader.decrypt("")
        pages = []
        for i, page in enumerate(reader.pages, 1):
            t = (page.extract_text() or "").strip()
            if t:
                pages.append(f"[page {i}]\n{t}")
    except Exception as e:  # damaged or password-protected
        raise Unsupported(f"This PDF could not be read ({type(e).__name__})") from e
    if not pages:
        raise Unsupported("This PDF has no text layer (it is probably scanned pictures)")
    return "\n\n".join(pages)


def _zip(data: bytes) -> zipfile.ZipFile:
    try:
        return zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as e:
        raise Unsupported("This file is damaged or not really that format") from e


def _docx(data: bytes) -> str:
    z = _zip(data)
    root = ET.fromstring(z.read("word/document.xml"))
    out = []
    for p in root.iter(f"{W}p"):
        style = p.find(f"{W}pPr/{W}pStyle")
        text = "".join(t.text or "" for t in p.iter(f"{W}t"))
        if not text.strip():
            continue
        head = style is not None and re.match(r"(?i)heading(\d)", style.get(f"{W}val", ""))
        out.append(("#" * int(head.group(1)) + " " if head else "") + text)
    for tbl in root.iter(f"{W}tbl"):  # tables again, as rows, so their layout survives
        rows = ["| " + " | ".join("".join(t.text or "" for t in c.iter(f"{W}t")) for c in r.iter(f"{W}tc")) + " |"
                for r in tbl.iter(f"{W}tr")]
        if rows:
            out.append("\n".join(rows))
    return "\n\n".join(out)


def _col(ref: str) -> int:
    n = 0
    for ch in re.match(r"[A-Z]+", ref).group(0):
        n = n * 26 + ord(ch) - 64
    return n - 1


def _xlsx(data: bytes, max_rows: int = 2000) -> str:
    z = _zip(data)
    shared = []
    if "xl/sharedStrings.xml" in z.namelist():
        for si in ET.fromstring(z.read("xl/sharedStrings.xml")).iter(f"{S}si"):
            shared.append("".join(t.text or "" for t in si.iter(f"{S}t")))
    names = []
    if "xl/workbook.xml" in z.namelist():
        names = [s.get("name") for s in ET.fromstring(z.read("xl/workbook.xml")).iter(f"{S}sheet")]
    sheets = sorted((n for n in z.namelist() if re.match(r"xl/worksheets/sheet\d+\.xml$", n)),
                    key=lambda n: int(re.search(r"(\d+)", n.rsplit("/", 1)[1]).group(1)))
    out = []
    for i, path in enumerate(sheets):
        rows = []
        for r in ET.fromstring(z.read(path)).iter(f"{S}row"):
            cells = {}
            for c in r.iter(f"{S}c"):
                v = c.find(f"{S}v")
                if c.get("t") == "inlineStr":
                    val = "".join(t.text or "" for t in c.iter(f"{S}t"))
                elif v is None:
                    continue
                elif c.get("t") == "s":
                    val = shared[int(v.text)] if v.text and int(v.text) < len(shared) else ""
                else:
                    val = v.text or ""
                cells[_col(c.get("r", "A1"))] = val.replace(",", " ").replace("\n", " ")
            if cells:
                rows.append(",".join(cells.get(j, "") for j in range(max(cells) + 1)))
            if len(rows) >= max_rows:
                rows.append(f"(only the first {max_rows} rows)")
                break
        title = names[i] if i < len(names) and names[i] else f"Sheet {i + 1}"
        out.append(f"## {title}\n" + "\n".join(rows))
    return "\n\n".join(out)


def _pptx(data: bytes) -> str:
    z = _zip(data)
    slides = sorted((n for n in z.namelist() if re.match(r"ppt/slides/slide\d+\.xml$", n)),
                    key=lambda n: int(re.search(r"(\d+)", n.rsplit("/", 1)[1]).group(1)))
    out = []
    for i, path in enumerate(slides, 1):
        paras = []
        for p in ET.fromstring(z.read(path)).iter(f"{A}p"):
            t = "".join(x.text or "" for x in p.iter(f"{A}t")).strip()
            if t:
                paras.append(t)
        if paras:
            out.append(f"[slide {i}]\n" + "\n".join(paras))
    return "\n\n".join(out)
