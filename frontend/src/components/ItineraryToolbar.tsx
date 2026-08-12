"use client";

import { useState } from "react";
import { Loader2 } from "lucide-react";
import type { ItineraryChange } from "@/lib/api";

interface ItineraryToolbarProps {
  onModify: (change: ItineraryChange) => void | Promise<void>;
  onConfirm: () => void | Promise<void>;
  onRegenerate: () => void | Promise<void>;
  isLoading?: boolean;
  waitingForConfirmation?: boolean;
}

const budgetPresets = [1000, 2000, 5000];
const paceOptions = ["轻松", "适中", "紧凑"];

export function ItineraryToolbar({
  onModify,
  onConfirm,
  onRegenerate,
  isLoading = false,
  waitingForConfirmation = false,
}: ItineraryToolbarProps) {
  const [budgetInput, setBudgetInput] = useState("");
  const [showBudgetInput, setShowBudgetInput] = useState(false);

  const handleBudgetSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const amount = Number(budgetInput);
    if (!amount || isLoading) return;
    await onModify({ action: "set_budget", value: amount });
    setBudgetInput("");
    setShowBudgetInput(false);
  };

  const buttonBase =
    "rounded-xl border border-hairline bg-canvas-soft px-2.5 py-1.5 text-xs font-medium text-ink/70 transition-colors hover:bg-primary-pale hover:text-ink disabled:cursor-not-allowed disabled:opacity-50";

  return (
    <div className="glass-card p-2">
      <div className="flex flex-wrap items-center gap-2">
        {/* Confirmation */}
        <button
          onClick={() => onConfirm()}
          disabled={isLoading || !waitingForConfirmation}
          className={buttonBase}
          title="确认满意"
        >
          {waitingForConfirmation && isLoading ? (
            <Loader2 className="h-3.5 w-3.5 animate-spin" />
          ) : (
            "确认满意"
          )}
        </button>

        {/* Day adjustments */}
        <div className="flex items-center gap-1 rounded-xl border border-hairline bg-canvas-soft/60 p-0.5">
          <button
            onClick={() => onModify({ action: "change_days", delta: 1 })}
            disabled={isLoading}
            className={buttonBase}
          >
            增加一天
          </button>
          <button
            onClick={() => onModify({ action: "change_days", delta: -1 })}
            disabled={isLoading}
            className={buttonBase}
          >
            减少一天
          </button>
        </div>

        {/* Budget */}
        <div className="flex items-center gap-1">
          {showBudgetInput ? (
            <form
              onSubmit={handleBudgetSubmit}
              className="flex items-center gap-1"
            >
              <input
                type="number"
                value={budgetInput}
                onChange={(e) => setBudgetInput(e.target.value)}
                placeholder="金额"
                disabled={isLoading}
                autoFocus
                className="input-wise w-20 py-1.5 px-2 text-xs"
              />
              <button
                type="submit"
                disabled={isLoading || !budgetInput}
                className={buttonBase}
              >
                确认
              </button>
              <button
                type="button"
                onClick={() => {
                  setShowBudgetInput(false);
                  setBudgetInput("");
                }}
                disabled={isLoading}
                className={buttonBase}
              >
                取消
              </button>
            </form>
          ) : (
            <>
              {budgetPresets.map((amount) => (
                <button
                  key={amount}
                  onClick={() => onModify({ action: "set_budget", value: amount })}
                  disabled={isLoading}
                  className={buttonBase}
                >
                  ¥{amount}
                </button>
              ))}
              <button
                onClick={() => setShowBudgetInput(true)}
                disabled={isLoading}
                className={buttonBase}
              >
                调整预算
              </button>
            </>
          )}
        </div>

        {/* Pace */}
        <div className="flex items-center gap-1 rounded-xl border border-hairline bg-canvas-soft/60 p-0.5">
          <span className="px-1.5 text-[11px] text-mute">节奏</span>
          {paceOptions.map((pace) => (
            <button
              key={pace}
              onClick={() => onModify({ action: "set_pace", value: pace })}
              disabled={isLoading}
              className={buttonBase}
            >
              {pace}
            </button>
          ))}
        </div>

        {/* Cancel */}
        <button
          onClick={() => onRegenerate()}
          disabled={isLoading}
          className={buttonBase}
        >
          重新生成
        </button>
      </div>
    </div>
  );
}
