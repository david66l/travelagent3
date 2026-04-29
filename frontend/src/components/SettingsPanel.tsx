"use client";

import { useState } from "react";
import { useChatStore } from "@/stores/chatStore";

interface PreferenceConfig {
  id: string;
  label: string;
  value: string;
  options: string[];
}

const defaultPreferences: PreferenceConfig[] = [
  { id: "budget", label: "预算优先", value: "高", options: ["高", "中", "低"] },
  { id: "walk", label: "步行容忍", value: "15 分钟", options: ["5 分钟", "10 分钟", "15 分钟", "20 分钟"] },
  { id: "accommodation", label: "住宿偏好", value: "安静旅馆", options: ["安静旅馆", "市中心酒店", "民宿", "豪华酒店"] },
  { id: "pace", label: "行程节奏", value: "适中", options: ["轻松", "适中", "紧凑"] },
  { id: "food", label: "饮食偏好", value: "本地特色", options: ["本地特色", "国际 cuisine", "素食", "无偏好"] },
];

export function SettingsPanel() {
  const store = useChatStore();
  const [prefs, setPrefs] = useState<PreferenceConfig[]>(defaultPreferences);
  const [saved, setSaved] = useState(false);

  const updatePref = (id: string, value: string) => {
    setPrefs((prev) =>
      prev.map((p) => (p.id === id ? { ...p, value } : p))
    );
    setSaved(false);
  };

  const handleSave = () => {
    setSaved(true);
    setTimeout(() => setSaved(false), 2000);
  };

  return (
    <div
      className="flex h-full flex-col gap-2.5 rounded-3xl p-3.5"
      style={{
        background: "rgba(255,255,255,0.66)",
        backdropFilter: "blur(24px)",
        WebkitBackdropFilter: "blur(24px)",
        border: "1px solid rgba(255,255,255,0.79)",
      }}
    >
      {/* Header */}
      <h2 className="text-lg font-semibold text-[#111111]">
        行程偏好设置
      </h2>

      {/* Preferences List */}
      <div className="flex flex-1 flex-col gap-2 overflow-y-auto scrollbar-thin">
        {prefs.map((pref) => (
          <div
            key={pref.id}
            className="flex flex-col gap-1.5 rounded-xl p-3"
            style={{ background: "rgba(255,255,255,0.8)" }}
          >
            <span className="text-sm text-[#111111]">{pref.label}</span>
            <div className="flex flex-wrap gap-1.5">
              {pref.options.map((opt) => (
                <button
                  key={opt}
                  onClick={() => updatePref(pref.id, opt)}
                  className="rounded-lg px-2.5 py-1 text-xs transition-colors"
                  style={
                    pref.value === opt
                      ? { background: "#111111", color: "#FFFFFF" }
                      : {
                          background: "rgba(255,255,255,0.6)",
                          color: "#333333",
                          border: "1px solid rgba(255,255,255,0.8)",
                        }
                  }
                >
                  {opt}
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>

      {/* Footer Buttons */}
      <div className="flex gap-2">
        <button
          onClick={() => store.setActiveView("itinerary")}
          className="flex-1 rounded-xl py-2.5 text-center text-sm text-[#333333] transition-colors"
          style={{ background: "rgba(255,255,255,0.8)" }}
        >
          关闭
        </button>
        <button
          onClick={handleSave}
          className="flex-1 rounded-xl bg-[#111111] py-2.5 text-center text-sm font-medium text-white transition-colors hover:bg-[#333333]"
        >
          {saved ? "已保存" : "保存策略"}
        </button>
      </div>
    </div>
  );
}
