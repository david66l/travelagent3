"use client";

import { useState, useEffect } from "react";
import { useChatStore } from "@/stores/chatStore";
import { cn } from "@/lib/utils";
import {
  getMe,
  getMyProfile,
  updateMyProfile,
  logoutUser,
  type MeResponse,
  type ProfileResponse,
} from "@/lib/api";

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
  const [me, setMe] = useState<MeResponse | null>(null);
  const [profile, setProfile] = useState<ProfileResponse | null>(null);

  // Load the account + the profile the agent has remembered for this user.
  useEffect(() => {
    let alive = true;
    (async () => {
      try {
        const [m, p] = await Promise.all([getMe(), getMyProfile()]);
        if (!alive) return;
        setMe(m);
        setProfile(p);
        // Hydrate the preference selectors from the saved profile.
        if (p?.preferences) {
          setPrefs((prev) =>
            prev.map((pref) => {
              const v = (p.preferences as Record<string, unknown>)[pref.id];
              return typeof v === "string" && pref.options.includes(v)
                ? { ...pref, value: v }
                : pref;
            })
          );
        }
      } catch {
        /* not signed in yet / offline — keep defaults */
      }
    })();
    return () => {
      alive = false;
    };
  }, []);

  const updatePref = (id: string, value: string) => {
    setPrefs((prev) =>
      prev.map((p) => (p.id === id ? { ...p, value } : p))
    );
    setSaved(false);
  };

  const handleSave = async () => {
    const preferences = Object.fromEntries(prefs.map((p) => [p.id, p.value]));
    try {
      await updateMyProfile({ preferences });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch {
      /* swallow — UI stays on the unsaved state */
    }
  };

  const handleLogout = async () => {
    await logoutUser();
    store.clear();
    store.setActiveView("chat");
  };

  const frequent = (profile?.frequent_destinations ?? []) as unknown[];

  return (
    <div className="glass-card flex h-full flex-col gap-2.5 rounded-3xl p-3.5">
      {/* Header */}
      <h2 className="text-lg font-semibold text-ink">账户与偏好</h2>

      {/* Account + what the agent remembers */}
      <div className="flex flex-col gap-2 rounded-xl bg-canvas-soft p-3">
        <div className="flex items-center justify-between">
          <span className="text-sm text-ink">账户</span>
          <span className="rounded-lg bg-hairline px-1.5 py-0.5 font-mono text-[10px] text-mute">
            {me?.role === "guest" ? "访客" : me?.role || "—"}
          </span>
        </div>
        {me?.email && <span className="font-mono text-xs text-mute">{me.email}</span>}
        {profile?.updated_at && (
          <span className="font-mono text-[11px] text-mute/70">
            画像更新于 {String(profile.updated_at).slice(0, 10)}
          </span>
        )}
        {frequent.length > 0 && (
          <div className="flex flex-col gap-1">
            <span className="text-xs text-mute">AI 记住的常去目的地</span>
            <div className="flex flex-wrap gap-1">
              {frequent.slice(0, 8).map((d, i) => (
                <span
                  key={i}
                  className="rounded-lg border border-hairline px-2 py-0.5 font-mono text-[11px] text-body"
                >
                  {typeof d === "string" ? d : JSON.stringify(d)}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>

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
      <div className="flex flex-col gap-2">
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
            {saved ? "已保存" : "保存偏好"}
          </button>
        </div>
        <button
          onClick={handleLogout}
          className="rounded-xl border border-hairline py-2 text-center text-xs text-mute transition-colors hover:text-ink"
        >
          退出登录
        </button>
      </div>
    </div>
  );
}
