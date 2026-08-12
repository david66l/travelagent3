"use client";

import { useState } from "react";
import jsPDF from "jspdf";
import writeXlsxFile from "write-excel-file/browser";
import { saveAs } from "file-saver";
import { useChatStore } from "@/stores/chatStore";
import { cn } from "@/lib/utils";

const formats = [
  { id: "pdf", label: "PDF 文档" },
  { id: "excel", label: "Excel 工作簿" },
  { id: "json", label: "JSON" },
  { id: "markdown", label: "Markdown" },
];

export function ExportCenter() {
  const store = useChatStore();
  const [selectedFormat, setSelectedFormat] = useState("pdf");
  const [exporting, setExporting] = useState(false);

  const itinerary = store.itinerary || [];
  const tripTitle = store.itinerary?.length
    ? `${store.itinerary.length} 天行程`
    : "行程导出";

  const handleExport = async () => {
    setExporting(true);
    try {
      switch (selectedFormat) {
        case "pdf":
          await exportPDF(itinerary);
          break;
        case "excel":
          exportExcel(itinerary);
          break;
        case "json":
          exportJSON(itinerary);
          break;
        case "markdown":
          exportMarkdown(itinerary);
          break;
      }
    } finally {
      setExporting(false);
    }
  };

  const hasData = itinerary.length > 0;

  return (
    <div className="glass-card flex h-full flex-col gap-3.5 rounded-4xl p-4">
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-semibold text-ink">{tripTitle}</h2>
        <button
          onClick={() => store.setActiveView("itinerary")}
          className="rounded-xl bg-canvas-soft px-3 py-2 text-xs font-medium text-body transition-colors hover:bg-primary-pale"
        >
          关闭
        </button>
      </div>

      {/* Format selector */}
      <div className="flex flex-col gap-2 rounded-3xl bg-canvas-soft p-3.5">
        <h3 className="text-base font-semibold text-ink">格式选择</h3>
        <div className="flex flex-col gap-2">
          {formats.map((f) => (
            <button key={f.id} onClick={() => setSelectedFormat(f.id)}
              className={cn(
                "w-full rounded-xl px-3 py-2.5 text-left text-sm transition-colors",
                selectedFormat === f.id
                  ? "bg-ink text-canvas"
                  : "bg-canvas text-ink hover:bg-primary-pale"
              )}
            >{f.label}</button>
          ))}
        </div>
      </div>

      {/* Export button */}
      <button onClick={handleExport} disabled={!hasData || exporting}
        className={cn(
          "rounded-xl px-4 py-3 text-sm font-semibold transition-all",
          hasData
            ? "btn-primary-dark"
            : "bg-hairline text-mute cursor-not-allowed"
        )}
      >
        {exporting ? "导出中..." : `导出 ${selectedFormat.toUpperCase()}`}
      </button>
    </div>
  );
}

/* ── 导出函数 ── */

async function exportPDF(itinerary: any[]) {
  const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  // jsPDF's built-in fonts do not contain Chinese glyphs. Render each page
  // through the browser canvas first, which uses the user's installed CJK font,
  // then place the rendered page into the PDF.
  const pageWidth = 1240;
  const pageHeight = 1754;
  const margin = 96;
  const pages: HTMLCanvasElement[] = [];
  let canvas!: HTMLCanvasElement;
  let ctx!: CanvasRenderingContext2D;
  let y = 0;

  const newPage = () => {
    canvas = document.createElement("canvas");
    canvas.width = pageWidth;
    canvas.height = pageHeight;
    ctx = canvas.getContext("2d")!;
    ctx.fillStyle = "#ffffff";
    ctx.fillRect(0, 0, pageWidth, pageHeight);
    ctx.textBaseline = "top";
    pages.push(canvas);
    y = margin;
  };

  const writeWrapped = (text: string, font: string, color: string, indent = 0) => {
    ctx.font = font;
    ctx.fillStyle = color;
    const maxWidth = pageWidth - margin * 2 - indent;
    const chars = Array.from(text);
    const lines: string[] = [];
    let line = "";
    for (const char of chars) {
      if (ctx.measureText(line + char).width > maxWidth && line) {
        lines.push(line);
        line = char;
      } else {
        line += char;
      }
    }
    if (line) lines.push(line);
    const lineHeight = font.includes("44px") ? 58 : font.includes("34px") ? 48 : 36;
    if (y + lines.length * lineHeight > pageHeight - margin) newPage();
    for (const value of lines) {
      ctx.fillText(value, margin + indent, y);
      y += lineHeight;
    }
  };

  newPage();
  writeWrapped("TravelAgent 行程单", "600 44px sans-serif", "#18212b");
  y += 24;

  for (const day of itinerary) {
    if (y > pageHeight - margin - 120) newPage();
    writeWrapped(
      `第 ${day.day_number} 天${day.date ? ` · ${day.date}` : ""} · ${day.theme || "行程"}`,
      "600 34px sans-serif",
      "#1f2937",
    );
    y += 12;
    for (const act of day.activities || []) {
      const time = act.start_time ? `${act.start_time}-${act.end_time || ""}` : "";
      const cost = act.ticket_price ? ` · ¥${act.ticket_price}` : "";
      writeWrapped(`${time}  ${act.poi_name}${cost}`, "500 25px sans-serif", "#111827", 20);
      if (act.recommendation_reason) {
        writeWrapped(String(act.recommendation_reason), "23px sans-serif", "#6b7280", 42);
      }
      y += 12;
    }
    y += 24;
  }

  pages.forEach((page, index) => {
    if (index > 0) doc.addPage();
    doc.addImage(page.toDataURL("image/jpeg", 0.92), "JPEG", 0, 0, 210, 297);
  });

  doc.save(`trip-${Date.now()}.pdf`);
}

async function exportExcel(itinerary: any[]) {
  const data: (string | number)[][] = [["天数", "时间", "地点", "费用", "推荐理由"]];
  for (const day of itinerary) {
    for (const act of day.activities || []) {
      data.push([
        `Day ${day.day_number}`,
        act.start_time ? `${act.start_time}-${act.end_time}` : "",
        act.poi_name,
        act.ticket_price || act.meal_cost || 0,
        act.recommendation_reason || "",
      ]);
    }
  }
  const blob = await writeXlsxFile(data, { sheet: "行程" }).toBlob();
  saveAs(blob, `trip-${Date.now()}.xlsx`);
}

function exportJSON(itinerary: any[]) {
  const blob = new Blob([JSON.stringify(itinerary, null, 2)], { type: "application/json" });
  saveAs(blob, `trip-${Date.now()}.json`);
}

function exportMarkdown(itinerary: any[]) {
  let md = "# TravelAgent 行程\n\n";
  for (const day of itinerary) {
    md += `## Day ${day.day_number} — ${day.theme || "行程"}\n\n`;
    for (const act of day.activities || []) {
      const time = act.start_time ? `(${act.start_time}-${act.end_time}) ` : "";
      const cost = act.ticket_price ? ` — ¥${act.ticket_price}` : "";
      md += `- ${time}**${act.poi_name}**${cost}\n`;
      if (act.recommendation_reason) md += `  _${act.recommendation_reason}_\n`;
    }
    md += "\n";
  }
  const blob = new Blob([md], { type: "text/markdown" });
  saveAs(blob, `trip-${Date.now()}.md`);
}
