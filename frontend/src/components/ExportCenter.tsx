"use client";

import { useState } from "react";
import jsPDF from "jspdf";
import * as XLSX from "xlsx";
import { saveAs } from "file-saver";
import { useChatStore } from "@/stores/chatStore";

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
          exportPDF(itinerary);
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
    <div
      className="flex h-full flex-col gap-3.5 rounded-4xl p-4"
      style={{
        background: "rgba(255,255,255,0.64)",
        backdropFilter: "blur(30px)",
        border: "1px solid rgba(255,255,255,0.7)",
      }}
    >
      <div className="flex items-center justify-between">
        <h2 className="text-2xl font-semibold text-[#111111]">{tripTitle}</h2>
        <button
          onClick={() => store.setActiveView("itinerary")}
          className="rounded-xl bg-[#111111] px-3 py-2 text-xs font-medium text-white"
        >
          关闭
        </button>
      </div>

      {/* Format selector */}
      <div className="flex flex-col gap-2 rounded-[18px] p-3.5" style={{ background: "rgba(255,255,255,0.55)" }}>
        <h3 className="text-base font-semibold text-[#111111]">格式选择</h3>
        <div className="flex flex-col gap-2">
          {formats.map((f) => (
            <button key={f.id} onClick={() => setSelectedFormat(f.id)}
              className="w-full rounded-xl px-3 py-2.5 text-left text-sm transition-colors"
              style={selectedFormat === f.id
                ? { background: "#111111", color: "#FFFFFF" }
                : { background: "rgba(255,255,255,0.65)", color: "#111111" }}
            >{f.label}</button>
          ))}
        </div>
      </div>

      {/* Export button */}
      <button onClick={handleExport} disabled={!hasData || exporting}
        className="rounded-xl px-4 py-3 text-sm font-semibold transition-all"
        style={hasData
          ? { background: "#111111", color: "#FFFFFF" }
          : { background: "#E5E5E5", color: "#999", cursor: "not-allowed" }}
      >
        {exporting ? "导出中..." : `导出 ${selectedFormat.toUpperCase()}`}
      </button>
    </div>
  );
}

/* ── 导出函数 ── */

function exportPDF(itinerary: any[]) {
  const doc = new jsPDF({ orientation: "portrait", unit: "mm", format: "a4" });
  let y = 20;

  doc.setFontSize(18);
  doc.text("TravelAgent 行程单", 20, y);
  y += 12;

  for (const day of itinerary) {
    if (y > 260) { doc.addPage(); y = 20; }
    doc.setFontSize(14);
    doc.text(`Day ${day.day_number} — ${day.theme || "行程"}`, 20, y);
    y += 8;

    for (const act of day.activities || []) {
      if (y > 270) { doc.addPage(); y = 20; }
      const time = act.start_time ? `${act.start_time}-${act.end_time}` : "";
      const cost = act.ticket_price ? ` ¥${act.ticket_price}` : "";
      doc.setFontSize(10);
      doc.text(`${time}  ${act.poi_name}${cost}`, 25, y);
      y += 6;
      if (act.recommendation_reason) {
        doc.setFontSize(8);
        doc.setTextColor(100);
        doc.text(`  ${act.recommendation_reason}`, 25, y);
        doc.setTextColor(0);
        y += 5;
      }
    }
    y += 6;
  }

  doc.save(`trip-${Date.now()}.pdf`);
}

function exportExcel(itinerary: any[]) {
  const wsData: any[][] = [["天数", "时间", "地点", "费用", "推荐理由"]];
  for (const day of itinerary) {
    for (const act of day.activities || []) {
      wsData.push([
        `Day ${day.day_number}`,
        act.start_time ? `${act.start_time}-${act.end_time}` : "",
        act.poi_name,
        act.ticket_price || act.meal_cost || 0,
        act.recommendation_reason || "",
      ]);
    }
  }
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(wsData), "行程");
  const buf = XLSX.write(wb, { bookType: "xlsx", type: "array" });
  saveAs(new Blob([buf]), `trip-${Date.now()}.xlsx`);
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
