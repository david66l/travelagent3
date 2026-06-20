"""OutputFormatAgent — Markdown polish, PDF/Excel export, map links.

Produces multi-modal itinerary artifacts from a planner-generated itinerary.
Failures in any formatter are isolated: the agent always returns at least the
original Markdown text.
"""

from __future__ import annotations

import logging
import os
import platform
import re
import uuid
from datetime import datetime
from typing import Any

from core.settings import settings

logger = logging.getLogger(__name__)

# Output directories relative to project root.
_PROJECT_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)
PDF_DIR = os.path.join(_PROJECT_ROOT, "outputs", "pdfs")
EXCEL_DIR = os.path.join(_PROJECT_ROOT, "outputs", "excel")


def _ensure_dirs() -> None:
    os.makedirs(PDF_DIR, exist_ok=True)
    os.makedirs(EXCEL_DIR, exist_ok=True)


def _safe_filename(prefix: str, ext: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}.{ext}"


def _strip_markdown_for_excel(text: str) -> str:
    """Remove markdown syntax for plain-cell export."""
    text = re.sub(r"#+\s*", "", text)
    text = re.sub(r"\*\*|__", "", text)
    text = re.sub(r"`", "", text)
    text = re.sub(r"\[([^\]]+)\]\([^\)]+\)", r"\1", text)
    return text.strip()


class OutputFormatAgent:
    """Format itinerary output into Markdown, PDF, Excel and map links."""

    def __init__(self, base_url: str | None = None) -> None:
        if base_url:
            self.base_url = base_url
        elif settings.public_app_url:
            self.base_url = settings.public_app_url.rstrip("/")
        else:
            # 0.0.0.0 is fine for binding but not for client-side download URLs.
            host = "localhost" if settings.app_host == "0.0.0.0" else settings.app_host
            self.base_url = f"http://{host}:{settings.app_port}"
        _ensure_dirs()

    async def format(
        self,
        proposal_text: str,
        itinerary: list[dict[str, Any]] | None = None,
        city: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Return formatted outputs and download URLs.

        Always returns at least `markdown`. PDF/Excel/map_url are best-effort.
        """
        polished = await self.polish_markdown(proposal_text)
        file_id = session_id or uuid.uuid4().hex[:12]

        pdf_path, pdf_url = await self.generate_pdf(polished, file_id)
        excel_path, excel_url = await self.generate_excel(
            polished, itinerary or [], file_id
        )
        map_url = self.generate_map_url(itinerary, city)

        return {
            "markdown": polished,
            "output_markdown": polished,
            "output_pdf_url": pdf_url,
            "output_excel_url": excel_url,
            "output_map_url": map_url,
            "files": {
                "pdf_path": pdf_path,
                "excel_path": excel_path,
            },
        }

    async def polish_markdown(self, proposal_text: str) -> str:
        """Light LLM polish of the itinerary Markdown without changing facts."""
        if not proposal_text:
            return ""
        try:
            from core.llm_client import llm

            messages = [
                {
                    "role": "system",
                    "content": (
                        "你是一位行程文档排版助手。请对以下 Markdown 行程进行排版和"
                        "润色（修正标题层级、统一列表符号、优化过渡语句），但**不要"
                        "修改任何景点名称、时间、价格、路线等事实信息**。只返回 Markdown 文本。"
                    ),
                },
                {"role": "user", "content": proposal_text[:6000]},
            ]
            polished = await llm.chat(messages, temperature=0.3, max_tokens=2048, task_type="output_format")
            return polished.strip() or proposal_text
        except Exception as exc:
            logger.warning("Markdown polish failed, using original: %s", exc)
            return proposal_text

    @staticmethod
    def _ensure_weasyprint_lib_path() -> None:
        """On macOS, WeasyPrint's CFFI loader needs Homebrew libs in dyld path."""
        if os.environ.get("DYLD_LIBRARY_PATH"):
            return
        if platform.system() != "Darwin":
            return
        for path in ("/opt/homebrew/lib", "/usr/local/lib"):
            if os.path.isdir(path):
                os.environ["DYLD_LIBRARY_PATH"] = path
                break

    async def generate_pdf(
        self, markdown_text: str, file_id: str
    ) -> tuple[str | None, str | None]:
        """Convert Markdown to PDF via WeasyPrint if available."""
        if not markdown_text:
            return None, None
        self._ensure_weasyprint_lib_path()
        try:
            import markdown as md_lib
            from weasyprint import HTML
        except Exception as exc:
            logger.warning("PDF dependencies unavailable: %s", exc)
            return None, None

        try:
            html_body = md_lib.markdown(markdown_text)
            html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
body {{ font-family: "Noto Sans CJK SC", "PingFang SC", "Microsoft YaHei", sans-serif; margin: 2cm; }}
h1, h2, h3 {{ color: #2c3e50; }}
p {{ line-height: 1.6; }}
ul {{ margin-left: 1em; }}
</style>
</head>
<body>
{html_body}
</body>
</html>"""
            filename = _safe_filename(file_id, "pdf")
            path = os.path.join(PDF_DIR, filename)
            HTML(string=html).write_pdf(path)
            url = f"{self.base_url}/api/v1/download/pdfs/{filename}"
            return path, url
        except Exception as exc:
            logger.warning("PDF generation failed: %s", exc)
            return None, None

    async def generate_excel(
        self,
        markdown_text: str,
        itinerary: list[dict[str, Any]],
        file_id: str,
    ) -> tuple[str | None, str | None]:
        """Export day-by-day itinerary to Excel."""
        try:
            from openpyxl import Workbook
            from openpyxl.styles import Font
        except Exception as exc:
            logger.warning("openpyxl unavailable: %s", exc)
            return None, None

        try:
            wb = Workbook()
            ws = wb.active
            if ws is None:
                ws = wb.create_sheet("行程")
            ws.title = "行程"

            ws.append([" TravelAgent 行程单"])
            ws["A1"].font = Font(bold=True, size=14)
            ws.append(["生成时间", datetime.now().strftime("%Y-%m-%d %H:%M")])
            ws.append([])
            ws.append(["天数", "时间", "类型", "名称", "时长(分钟)", "备注"])
            header_row = ws[4]
            for cell in header_row:
                cell.font = Font(bold=True)

            for day_idx, day in enumerate(itinerary, start=1):
                activities = day.get("activities") or day.get("items") or []
                for act in activities:
                    name = act.get("name") or act.get("poi_name") or ""
                    start_time = act.get("start_time") or act.get("time") or ""
                    category = act.get("category") or act.get("type") or ""
                    duration = act.get("duration_minutes") or act.get("duration") or ""
                    note = act.get("note") or act.get("reason") or ""
                    ws.append([f"第{day_idx}天", str(start_time), category, name, duration, note])

            # Summary sheet with plain-text Markdown.
            summary = wb.create_sheet("行程摘要")
            summary.append(["行程摘要"])
            for line in _strip_markdown_for_excel(markdown_text).splitlines():
                summary.append([line])

            filename = _safe_filename(file_id, "xlsx")
            path = os.path.join(EXCEL_DIR, filename)
            wb.save(path)
            url = f"{self.base_url}/api/v1/download/excel/{filename}"
            return path, url
        except Exception as exc:
            logger.warning("Excel generation failed: %s", exc)
            return None, None

    def generate_map_url(
        self, itinerary: list[dict[str, Any]] | None, city: str | None
    ) -> str | None:
        """Build an AMap static map URL with day markers."""
        if not settings.amap_key:
            return None
        if not itinerary:
            return None

        markers: list[str] = []
        seen: set[str] = set()
        for day_idx, day in enumerate(itinerary[:3], start=1):
            activities = day.get("activities") or day.get("items") or []
            for act in activities[:6]:
                name = act.get("name") or act.get("poi_name") or ""
                if not name or name in seen:
                    continue
                seen.add(name)
                # Use a deterministic pseudo-location for the label.
                lat, lng = self._pseudo_coords(name)
                markers.append(f"{lng},{lat},{day_idx}")
        if not markers:
            return None

        markers_param = "|".join(markers)
        return (
            "https://restapi.amap.com/v3/staticmap"
            f"?key={settings.amap_key}"
            f"&markers=mid,0xFF0000:{markers_param}"
            f"&size=800x600"
            f"&zoom=12"
        )

    @staticmethod
    def _pseudo_coords(name: str) -> tuple[float, float]:
        h = hash(name) % 10000
        return 30.0 + (h % 100) / 100.0, 110.0 + (h // 100) / 100.0


# Singleton agent for the application.
output_format_agent = OutputFormatAgent()

__all__ = ["OutputFormatAgent", "output_format_agent", "PDF_DIR", "EXCEL_DIR"]
