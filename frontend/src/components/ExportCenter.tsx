"use client";

import { useState } from "react";
import { useChatStore } from "@/stores/chatStore";

const formats = [
  { id: "pdf", label: "PDF 文档" },
  { id: "excel", label: "Excel 工作簿" },
  { id: "json", label: "JSON" },
  { id: "markdown", label: "Markdown" },
];

const scopes = [
  { id: "full", label: "全部", desc: "导出完整行程" },
  { id: "selected", label: "选定", desc: "仅选中天数" },
  { id: "summary", label: "摘要", desc: "仅关键信息" },
];

const permissions = [
  { role: "财务团队", level: "可编辑" },
  { role: "运营经理", level: "可评论" },
  { role: "外部审计", level: "仅查看" },
];

export function ExportCenter() {
  const store = useChatStore();
  const [selectedFormat, setSelectedFormat] = useState("pdf");
  const [selectedScope, setSelectedScope] = useState("full");

  const tripTitle = store.itinerary
    ? `行程导出（共 ${store.itinerary.length} 天）`
    : "行程导出";

  return (
    <div
      className="flex h-full flex-col gap-3.5 rounded-4xl p-4"
      style={{
        background: "rgba(255,255,255,0.64)",
        backdropFilter: "blur(30px)",
        WebkitBackdropFilter: "blur(30px)",
        border: "1px solid rgba(255,255,255,0.7)",
        boxShadow: "0 14px 32px rgba(0,0,0,0.12)",
      }}
    >
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex flex-col gap-1">
          <h2 className="text-2xl font-semibold text-[#111111]">{tripTitle}</h2>
          <p className="font-mono text-xs text-[#111111]/60">
            仅针对当前行程配置格式、范围、权限与发送方式
          </p>
        </div>
        <button
          onClick={() => store.setActiveView("itinerary")}
          className="rounded-xl bg-[#111111] px-3 py-2 text-xs font-medium text-white transition-colors hover:bg-[#333333]"
        >
          关闭
        </button>
      </div>

      {/* Format Chooser */}
      <div
        className="flex flex-col gap-2.5 rounded-[18px] p-3.5"
        style={{
          background: "rgba(255,255,255,0.55)",
          border: "1px solid rgba(255,255,255,0.7)",
        }}
      >
        <h3 className="text-base font-semibold text-[#111111]">格式选择</h3>
        <div className="flex flex-col gap-2">
          {formats.map((f) => (
            <button
              key={f.id}
              onClick={() => setSelectedFormat(f.id)}
              className="w-full rounded-xl px-3 py-2.5 text-left text-sm transition-colors"
              style={
                selectedFormat === f.id
                  ? { background: "#111111", color: "#FFFFFF" }
                  : {
                      background: "rgba(255,255,255,0.65)",
                      color: "#111111",
                      border: "1px solid rgba(255,255,255,0.8)",
                    }
              }
            >
              {f.label}
            </button>
          ))}
        </div>
      </div>

      {/* Scope Options */}
      <div
        className="flex flex-col gap-2.5 rounded-[18px] p-3.5"
        style={{
          background: "rgba(255,255,255,0.55)",
          border: "1px solid rgba(255,255,255,0.7)",
        }}
      >
        <h3 className="text-base font-semibold text-[#111111]">范围选项</h3>
        <div className="flex flex-col gap-2">
          {scopes.map((s) => (
            <button
              key={s.id}
              onClick={() => setSelectedScope(s.id)}
              className="flex flex-col gap-1 rounded-xl p-3 text-left transition-colors"
              style={
                selectedScope === s.id
                  ? { background: "rgba(255,255,255,0.9)", border: "1px solid rgba(255,255,255,0.9)" }
                  : {
                      background: "rgba(255,255,255,0.65)",
                      border: "1px solid rgba(255,255,255,0.8)",
                    }
              }
            >
              <span className="text-sm text-[#111111]">{s.label}</span>
              <span className="text-xs text-[#111111]/50">{s.desc}</span>
            </button>
          ))}
        </div>
      </div>

      {/* Share Permissions */}
      <div
        className="flex flex-col gap-2 rounded-[18px] p-3.5"
        style={{
          background: "rgba(255,255,255,0.55)",
          border: "1px solid rgba(255,255,255,0.7)",
        }}
      >
        <div className="flex items-center justify-between">
          <h3 className="text-base font-semibold text-[#111111]">共享权限</h3>
          <span className="font-mono text-xs text-[#111111]/60">策略已生效</span>
        </div>
        <div className="flex flex-col gap-2">
          {permissions.map((p) => (
            <div
              key={p.role}
              className="flex items-center justify-between rounded-xl px-3 py-2.5"
              style={{ background: "rgba(255,255,255,0.65)" }}
            >
              <span className="text-xs text-[#111111]/70">{p.role}</span>
              <span className="text-xs text-[#111111]">{p.level}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
