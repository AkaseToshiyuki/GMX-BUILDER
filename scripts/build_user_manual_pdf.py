#!/usr/bin/env python3
"""Build the published GMXBUILDER user manual as an A4 PDF.

The renderer intentionally depends only on ReportLab. It supports the Markdown
constructs used by the paired English and Chinese manuals and embeds both Latin
and CJK-capable fonts.
"""

from __future__ import annotations

import argparse
import html
import re
import textwrap
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    ListFlowable,
    ListItem,
    NextPageTemplate,
    PageBreak,
    PageTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    XPreformatted,
)
from reportlab.platypus.tableofcontents import TableOfContents


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = ROOT / "docs" / "GMXBUILDER_USER_MANUAL_V1.0.1.md"
DEFAULT_OUTPUT = ROOT / "docs" / "GMXBUILDER_USER_MANUAL_V1.0.1.pdf"
ZH_SOURCE = ROOT / "docs" / "GMXBUILDER_USER_MANUAL_V1.0.1.zh-CN.md"
ZH_OUTPUT = ROOT / "docs" / "GMXBUILDER_USER_MANUAL_V1.0.1.zh-CN.pdf"
FONT_PATH = Path("/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf")
LATIN_FONT_PATH = Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf")
TITLE = "GMXBUILDER User Manual"
DOC_VERSION = "V1.0.1"
AUTHOR = "Haochen Yang"
DATE = "2026-08-11"
SOFTWARE = "GMXBUILDER v0.8.5 or later"
LANGUAGE = "en"


def register_fonts() -> None:
    if not FONT_PATH.is_file():
        raise FileNotFoundError(f"CJK font not found: {FONT_PATH}")
    if not LATIN_FONT_PATH.is_file():
        raise FileNotFoundError(f"Latin font not found: {LATIN_FONT_PATH}")
    pdfmetrics.registerFont(TTFont("GMXCJK", str(FONT_PATH)))
    pdfmetrics.registerFont(TTFont("GMXLatin", str(LATIN_FONT_PATH)))


def apply_font_fallback(markup: str) -> str:
    """Use the embedded Latin font for ASCII and the CJK font otherwise."""
    parts = re.split(r"(<[^>]+>)", markup)
    converted = []
    for part in parts:
        if part.startswith("<") and part.endswith(">"):
            converted.append(part)
        else:
            for run in re.findall(r"[\x00-\x7f]+|[^\x00-\x7f]+", part):
                font = "GMXLatin" if all(ord(char) < 128 for char in run) else "GMXCJK"
                converted.append(f'<font name="{font}">{run}</font>')
    return "".join(converted)


def inline_markup(text: str) -> str:
    """Convert the small inline-Markdown subset used by the manual."""
    placeholders: list[str] = []

    def stash_code(match: re.Match[str]) -> str:
        placeholders.append(
            '<font name="GMXCJK" color="#0f4c81">'
            + html.escape(match.group(1))
            + "</font>"
        )
        return f"\x00{len(placeholders) - 1}\x00"

    text = re.sub(r"`([^`]+)`", stash_code, text)
    text = html.escape(text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r'<link href="\2" color="#1d4ed8">\1</link>', text)
    for index, replacement in enumerate(placeholders):
        text = text.replace(f"\x00{index}\x00", replacement)
    return apply_font_fallback(text)


def code_markup(text: str) -> str:
    wrapped: list[str] = []
    for line in text.splitlines():
        wrapped.extend(
            textwrap.wrap(
                line,
                width=105,
                subsequent_indent="    ",
                replace_whitespace=False,
                drop_whitespace=False,
                break_long_words=False,
                break_on_hyphens=False,
            )
            or [""]
        )
    return apply_font_fallback(html.escape("\n".join(wrapped)))


class ManualDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str, **kwargs):
        super().__init__(filename, **kwargs)
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="body",
        )
        self.addPageTemplates(
            [
                PageTemplate(id="cover", frames=[frame], onPage=self._cover_page),
                PageTemplate(id="body", frames=[frame], onPage=self._body_page),
            ]
        )
        self._bookmark_count = 0

    def beforeDocument(self) -> None:
        # multiBuild renders more than once while resolving the TOC. Stable
        # bookmark keys are required for the index to converge.
        self._bookmark_count = 0

    def _set_metadata(self, canvas) -> None:
        canvas.setTitle(TITLE)
        canvas.setAuthor(AUTHOR)
        canvas.setSubject(
            f"{TITLE}，文档版本 {DOC_VERSION}"
            if LANGUAGE == "zh-CN"
            else f"{TITLE}, document version {DOC_VERSION}"
        )
        canvas.setCreator("GMXBUILDER ReportLab documentation builder")
        canvas.setKeywords("GMXBUILDER, GROMACS, membrane, API, CLI, Web")

    def _cover_page(self, canvas, doc) -> None:
        self._set_metadata(canvas)

    def _body_page(self, canvas, doc) -> None:
        self._set_metadata(canvas)
        canvas.saveState()
        footer_style = ParagraphStyle(
            "Footer",
            fontName="GMXCJK",
            fontSize=7.5,
            leading=9,
            textColor=colors.HexColor("#64748b"),
        )
        left_footer = Paragraph(
            inline_markup(f"{TITLE} · {DOC_VERSION}"), footer_style
        )
        _, footer_height = left_footer.wrap(90 * mm, 10 * mm)
        left_footer.drawOn(canvas, doc.leftMargin, 10.5 * mm)
        page_label = f"第 {doc.page - 1} 页" if LANGUAGE == "zh-CN" else f"Page {doc.page - 1}"
        right_footer = Paragraph(
            inline_markup(page_label),
            ParagraphStyle("FooterRight", parent=footer_style, alignment=2),
        )
        right_footer.wrapOn(canvas, 35 * mm, footer_height)
        right_footer.drawOn(
            canvas, A4[0] - doc.rightMargin - 35 * mm, 10.5 * mm
        )
        canvas.setStrokeColor(colors.HexColor("#cbd5e1"))
        canvas.line(
            doc.leftMargin,
            16 * mm,
            A4[0] - doc.rightMargin,
            16 * mm,
        )
        canvas.restoreState()

    def afterFlowable(self, flowable) -> None:
        if isinstance(flowable, Paragraph):
            level = getattr(flowable, "_heading_level", None)
            if level is None:
                return
            self._bookmark_count += 1
            key = f"heading-{self._bookmark_count}"
            text = flowable.getPlainText()
            heading_top = min(
                A4[1] - self.topMargin,
                self.frame._y + flowable.height + flowable.getSpaceAfter(),
            )
            self.canv.bookmarkHorizontalAbsolute(key, heading_top)
            self.canv.addOutlineEntry(text, key, level=level - 1, closed=False)
            self.notify(
                "TOCEntry",
                (level - 1, apply_font_fallback(html.escape(text)), self.page - 1, key),
            )


def make_styles():
    sample = getSampleStyleSheet()
    body = ParagraphStyle(
        "BodyCJK",
        parent=sample["BodyText"],
        fontName="GMXCJK",
        fontSize=9,
        leading=14,
        textColor=colors.HexColor("#1f2937"),
        spaceAfter=5,
        wordWrap="CJK",
    )
    styles = {
        "body": body,
        "quote": ParagraphStyle(
            "Quote",
            parent=body,
            leftIndent=8 * mm,
            rightIndent=4 * mm,
            borderColor=colors.HexColor("#94a3b8"),
            borderWidth=1,
            borderPadding=5,
            backColor=colors.HexColor("#f8fafc"),
            textColor=colors.HexColor("#475569"),
        ),
        "h1": ParagraphStyle(
            "Heading1CJK",
            parent=body,
            fontSize=18,
            leading=24,
            textColor=colors.HexColor("#0f3d64"),
            spaceBefore=14,
            spaceAfter=9,
            keepWithNext=True,
        ),
        "h2": ParagraphStyle(
            "Heading2CJK",
            parent=body,
            fontSize=14,
            leading=19,
            textColor=colors.HexColor("#0f4c81"),
            spaceBefore=11,
            spaceAfter=6,
            keepWithNext=True,
        ),
        "h3": ParagraphStyle(
            "Heading3CJK",
            parent=body,
            fontSize=11,
            leading=16,
            textColor=colors.HexColor("#155e75"),
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True,
        ),
        "code": ParagraphStyle(
            "CodeCJK",
            parent=body,
            fontName="GMXCJK",
            fontSize=6.5,
            leading=9,
            leftIndent=4 * mm,
            rightIndent=3 * mm,
            borderColor=colors.HexColor("#cbd5e1"),
            borderWidth=0.5,
            borderPadding=5,
            backColor=colors.HexColor("#f8fafc"),
            textColor=colors.HexColor("#111827"),
            spaceBefore=3,
            spaceAfter=7,
        ),
        "small": ParagraphStyle(
            "SmallCJK",
            parent=body,
            fontSize=7.5,
            leading=10,
        ),
        "cover_title": ParagraphStyle(
            "CoverTitle",
            parent=body,
            fontSize=28,
            leading=38,
            alignment=TA_CENTER,
            textColor=colors.HexColor("#0f3d64"),
            spaceAfter=12,
        ),
    }
    return styles


def make_table(rows: list[list[str]], styles, page_width: float) -> Table:
    parsed = [
        [Paragraph(inline_markup(cell.strip()), styles["small"]) for cell in row]
        for row in rows
    ]
    columns = max(len(row) for row in parsed)
    for row in parsed:
        row.extend([Paragraph("", styles["small"])] * (columns - len(row)))
    header = [cell.strip() for cell in rows[0]] if rows else []
    if header in (
        ["文档版本", "日期", "变更内容", "编写人"],
        ["Document version", "Date", "Change", "Author"],
    ):
        # Keep the release notes readable: the change description needs more
        # room than the three short metadata columns.
        ratios = (0.16, 0.18, 0.46, 0.20)
        widths = [page_width * ratio for ratio in ratios]
    else:
        widths = [page_width / columns] * columns
    table = Table(parsed, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#dbeafe")),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.HexColor("#0f3d64")),
                ("FONTNAME", (0, 0), (-1, -1), "GMXCJK"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#94a3b8")),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [
                    colors.white,
                    colors.HexColor("#f8fafc"),
                ]),
                ("LEFTPADDING", (0, 0), (-1, -1), 4),
                ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                ("TOPPADDING", (0, 0), (-1, -1), 4),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
            ]
        )
    )
    return table


def parse_markdown(source: str, styles, page_width: float):
    lines = source.splitlines()
    story = []
    paragraph: list[str] = []
    list_items: list[str] = []
    list_kind = "bullet"
    code_lines: list[str] = []
    in_code = False
    table_rows: list[list[str]] = []

    def flush_paragraph():
        if paragraph:
            story.append(
                Paragraph(inline_markup(" ".join(part.strip() for part in paragraph)), styles["body"])
            )
            paragraph.clear()

    def flush_list():
        nonlocal list_items
        if list_items:
            items = [
                ListItem(
                    Paragraph(inline_markup(item), styles["body"]),
                    leftIndent=4 * mm,
                )
                for item in list_items
            ]
            story.append(
                ListFlowable(
                    items,
                    bulletType="1" if list_kind == "number" else "bullet",
                    start="1",
                    leftIndent=7 * mm,
                    bulletFontName="GMXCJK",
                    bulletFontSize=8,
                    spaceAfter=5,
                )
            )
            list_items = []

    def flush_table():
        if table_rows:
            if len(table_rows) >= 2 and all(
                re.fullmatch(r":?-{3,}:?", cell.strip())
                for cell in table_rows[1]
            ):
                table_rows.pop(1)
            story.append(make_table(table_rows, styles, page_width))
            story.append(Spacer(1, 5))
            table_rows.clear()

    for line in lines:
        if line.startswith("```"):
            flush_paragraph()
            flush_list()
            flush_table()
            if in_code:
                story.append(
                    XPreformatted(
                        code_markup("\n".join(code_lines)),
                        styles["code"],
                    )
                )
                code_lines = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_lines.append(line.expandtabs(4))
            continue

        heading = re.match(r"^(#{1,3})\s+(.+)$", line)
        if heading:
            flush_paragraph()
            flush_list()
            flush_table()
            markdown_level = len(heading.group(1))
            # The Markdown H1 is rendered as the dedicated PDF cover. Promote
            # the remaining H2/H3 sections so bookmarks start at level zero.
            level = max(1, markdown_level - 1)
            if level == 1 and story and not isinstance(story[-1], PageBreak):
                story.append(PageBreak())
            heading_flowable = Paragraph(
                inline_markup(heading.group(2)),
                styles[f"h{level}"],
            )
            heading_flowable._heading_level = level
            story.append(heading_flowable)
            continue

        if line.startswith("|") and line.endswith("|"):
            flush_paragraph()
            flush_list()
            table_rows.append([cell for cell in line.strip("|").split("|")])
            continue
        flush_table()

        bullet = re.match(r"^\s*[-*]\s+(.+)$", line)
        number = re.match(r"^\s*\d+\.\s+(.+)$", line)
        if bullet or number:
            flush_paragraph()
            new_kind = "number" if number else "bullet"
            if list_items and new_kind != list_kind:
                flush_list()
            list_kind = new_kind
            list_items.append((number or bullet).group(1))
            continue
        flush_list()

        if line.startswith("> "):
            flush_paragraph()
            story.append(Paragraph(inline_markup(line[2:]), styles["quote"]))
            continue
        if line.strip() == "---":
            flush_paragraph()
            story.append(Spacer(1, 4))
            continue
        if not line.strip():
            flush_paragraph()
            continue
        paragraph.append(line)

    flush_paragraph()
    flush_list()
    flush_table()
    if in_code and code_lines:
        story.append(XPreformatted(code_markup("\n".join(code_lines)), styles["code"]))
    return story


def build_pdf(source: Path, output: Path) -> None:
    global TITLE, DOC_VERSION, AUTHOR, DATE, SOFTWARE, LANGUAGE
    register_fonts()
    styles = make_styles()
    text = source.read_text(encoding="utf-8")

    LANGUAGE = "zh-CN" if "## 变更日志" in text else "en"
    labels = (
        {
            "version": "文档版本", "software": "适用软件", "author": "编写人",
            "date": "发布日期", "status": "文档状态",
        }
        if LANGUAGE == "zh-CN"
        else {
            "version": "Document version", "software": "Software", "author": "Author",
            "date": "Release date", "status": "Status",
        }
    )

    def metadata_value(label: str) -> str:
        match = re.search(
            rf"(?m)^\|\s*{re.escape(label)}\s*\|\s*([^|]+?)\s*\|$", text
        )
        if not match:
            raise ValueError(f"Manual metadata is missing: {label}")
        return match.group(1).strip()

    TITLE = "GMXBUILDER 用户手册" if LANGUAGE == "zh-CN" else "GMXBUILDER User Manual"
    DOC_VERSION = metadata_value(labels["version"])
    SOFTWARE = metadata_value(labels["software"])
    AUTHOR = metadata_value(labels["author"])
    DATE = metadata_value(labels["date"])
    status_value = metadata_value(labels["status"])

    # The PDF cover provides the same metadata more cleanly than rendering the
    # opening Markdown title and metadata table twice.
    marker = "## 变更日志" if LANGUAGE == "zh-CN" else "## Change log"
    if marker not in text:
        raise ValueError(f"Manual is missing required section: {marker}")
    content = marker + text.split(marker, 1)[1]
    # Appendix B is a maintainer-only source note. Keep it in Markdown but do
    # not publish it in the reader-facing PDF.
    content = content.split("## 附录 B：文档维护要求", 1)[0].rstrip()

    doc = ManualDocTemplate(
        str(output),
        pagesize=A4,
        leftMargin=18 * mm,
        rightMargin=18 * mm,
        topMargin=18 * mm,
        bottomMargin=22 * mm,
    )

    story = [
        Spacer(1, 42 * mm),
        Paragraph(inline_markup(TITLE), styles["cover_title"]),
        Spacer(1, 30 * mm),
        Table(
            [
                [Paragraph(inline_markup(labels["version"]), styles["body"]), Paragraph(inline_markup(DOC_VERSION), styles["body"])],
                [Paragraph(inline_markup(labels["software"]), styles["body"]), Paragraph(inline_markup(SOFTWARE), styles["body"])],
                [Paragraph(inline_markup(labels["author"]), styles["body"]), Paragraph(inline_markup(AUTHOR), styles["body"])],
                [Paragraph(inline_markup(labels["date"]), styles["body"]), Paragraph(inline_markup(DATE), styles["body"])],
                [Paragraph(inline_markup(labels["status"]), styles["body"]), Paragraph(inline_markup(status_value), styles["body"])],
            ],
            colWidths=[38 * mm, 80 * mm],
            hAlign="CENTER",
            style=TableStyle(
                [
                    ("FONTNAME", (0, 0), (-1, -1), "GMXCJK"),
                    ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#dbeafe")),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#94a3b8")),
                    ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
                    ("LEFTPADDING", (0, 0), (-1, -1), 7),
                    ("TOPPADDING", (0, 0), (-1, -1), 7),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 7),
                ]
            ),
        ),
        Spacer(1, 28 * mm),
        Paragraph(
            inline_markup(
                "本手册描述可验证的起始体系构建流程；任何生产模拟均需独立完成"
                "能量最小化、充分平衡、重复采样和科学复核。"
                if LANGUAGE == "zh-CN"
                else "This manual describes validated preparation of starting systems. "
                "Production simulation still requires minimization, sufficient "
                "equilibration, repeat sampling, and scientific review."
            ),
            ParagraphStyle(
                "CoverNotice",
                parent=styles["body"],
                alignment=TA_CENTER,
                textColor=colors.HexColor("#64748b"),
            ),
        ),
        NextPageTemplate("body"),
        PageBreak(),
        Paragraph(inline_markup("目录" if LANGUAGE == "zh-CN" else "Contents"), styles["h1"]),
        Paragraph(
            inline_markup(
                "单击目录条目或右侧页码可跳转到对应章节。"
                if LANGUAGE == "zh-CN"
                else "Select a section title or page number to follow the document link."
            ),
            styles["small"],
        ),
    ]

    toc = TableOfContents()
    toc.levelStyles = [
        ParagraphStyle(
            "TOC1",
            fontName="GMXLatin",
            fontSize=9,
            leading=14,
            leftIndent=0,
            firstLineIndent=0,
            textColor=colors.HexColor("#0f3d64"),
        ),
        ParagraphStyle(
            "TOC2",
            fontName="GMXLatin",
            fontSize=8,
            leading=12,
            leftIndent=7 * mm,
            firstLineIndent=0,
            textColor=colors.HexColor("#334155"),
        ),
        ParagraphStyle(
            "TOC3",
            fontName="GMXLatin",
            fontSize=7,
            leading=10,
            leftIndent=14 * mm,
            firstLineIndent=0,
            textColor=colors.HexColor("#475569"),
        ),
    ]
    story.extend([toc, PageBreak()])
    story.extend(parse_markdown(content, styles, doc.width))

    output.parent.mkdir(parents=True, exist_ok=True)
    doc.multiBuild(story)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--all", action="store_true", help="build both English and Chinese manuals"
    )
    args = parser.parse_args()
    targets = (
        [(DEFAULT_SOURCE, DEFAULT_OUTPUT), (ZH_SOURCE, ZH_OUTPUT)]
        if args.all
        else [(args.source, args.output)]
    )
    for source, output in targets:
        build_pdf(source.resolve(), output.resolve())
        print(output.resolve())


if __name__ == "__main__":
    main()
