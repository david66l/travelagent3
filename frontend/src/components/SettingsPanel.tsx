"use client";

import { useState } from "react";
import { useChatStore } from "@/stores/chatStore";
import { cn } from "@/lib/utils";

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
    <div className="glass-card flex h-full flex-col gap-2.5 rounded-3xl p-3.5">
      {/* Header */}
      <h2 className="text-lg font-semibold text-ink">
        行程偏好设置
      </h2>

      {/* Preferences List */}
      <div className="flex flex-1 flex-col gap-2 overflow-y-auto scrollbar-thin">
        {prefs.map((pref) => (
          <div
            key={pref.id}
            className="flex flex-col gap-1.5 rounded-xl bg-canvas-soft p-3"
          >
            <span className="text-sm text-ink">{pref.label}</span>
            <div className="flex flex-wrap gap-1.5">
              {pref.options.map((opt) => (
                <button
                  key={opt}
                  onClick={() => updatePref(pref.id, opt)}
                  className={cn(
                    "rounded-lg px-2.5 py-1 text-xs transition-colors",
                    pref.value === opt
                      ? "bg-ink text-canvas"
                      : "border border-hairline bg-canvas text-body hover:bg-primary-pale"
                  )}
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
          className="flex-1 rounded-xl bg-canvas-soft py-2.5 text-center text-sm text-body transition-colors hover:bg-primary-pale"
        >
          关闭
        </button>
        <button
          onClick={handleSave}
          className="btn-primary-dark flex-1 py-2.5 text-center text-sm"
        >
          {saved ? "已保存" : "保存策略"}
        </button>
      </div>
    </div>
  );
}
