"use client";

import { useState } from "react";
import { Loader2 } from "lucide-react";

interface ItineraryToolbarProps {
  onModify: (message: string) => void | Promise<void>;
  isLoading?: boolean;
  waitingForConfirmation?: boolean;
}

const budgetPresets = [1000, 2000, 5000];
const paceOptions = ["轻松", "适中", "紧凑"];

export function ItineraryToolbar({
  onModify,
  isLoading = false,
  waitingForConfirmation = false,
}: ItineraryToolbarProps) {
  const [budgetInput, setBudgetInput] = useState("");
  const [showBudgetInput, setShowBudgetInput] = useState(false);

  const handleAction = async (message: string) => {
    if (isLoading) return;
    await onModify(message);
  };

  const handleBudgetSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const amount = Number(budgetInput);
    if (!amount || isLoading) return;
    await onModify(`预算调整到 ${amount}`);
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
          onClick={() => handleAction("确认行程")}
          disabled={isLoading}
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
            onClick={() => handleAction("增加一天")}
            disabled={isLoading}
            className={buttonBase}
          >
            增加一天
          </button>
          <button
            onClick={() => handleAction("减少一天")}
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
                  onClick={() => handleAction(`预算调整到 ${amount}`)}
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
              onClick={() => handleAction(`节奏调整为 ${pace}`)}
              disabled={isLoading}
              className={buttonBase}
            >
              {pace}
            </button>
          ))}
        </div>

        {/* Cancel */}
        <button
          onClick={() => handleAction("取消当前修改")}
          disabled={isLoading}
          className={buttonBase}
        >
          取消
        </button>
      </div>
    </div>
  );
}
