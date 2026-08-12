"use client";

import { useMemo, useState } from "react";
import { useChatStore } from "@/stores/chatStore";
import {
  searchFlights,
  searchHotels,
  checkTicket,
  type FlightResult,
  type HotelResult,
  type TicketResult,
} from "@/lib/api";
import { cn } from "@/lib/utils";

function todayPlus(days: number): string {
  const d = new Date();
  d.setDate(d.getDate() + days);
  return d.toISOString().slice(0, 10);
}

export function BookingPanel() {
  const store = useChatStore();
  const destination = store.confirmedInfo?.destination || "";
  const tripDays = store.itinerary?.length || 1;

  const pois = useMemo(
    () =>
      Array.from(
        new Set(
          (store.itinerary ?? []).flatMap((d) =>
            (d.activities ?? [])
              .filter((a) => a.category !== "restaurant")
              .map((a) => a.poi_name)
              .filter(Boolean)
          )
        )
      ),
    [store.itinerary]
  );

  const [origin, setOrigin] = useState("北京");
  const [date, setDate] = useState(todayPlus(7));
  const [flights, setFlights] = useState<FlightResult[] | null>(null);
  const [hotels, setHotels] = useState<HotelResult[] | null>(null);
  const [tickets, setTickets] = useState<Record<string, TicketResult>>({});
  const [loading, setLoading] = useState<string | null>(null);
  const [booked, setBooked] = useState<Set<string>>(new Set());

  const markBooked = (key: string) =>
    setBooked((prev) => new Set(prev).add(key));

  const doFlights = async () => {
    setLoading("flights");
    try {
      const r = await searchFlights(origin, destination, date);
      setFlights(r.flights);
    } catch {
      setFlights([]);
    } finally {
      setLoading(null);
    }
  };

  const doHotels = async () => {
    setLoading("hotels");
    try {
      const r = await searchHotels(destination, date, todayPlus(7 + tripDays));
      setHotels(r.hotels);
    } catch {
      setHotels([]);
    } finally {
      setLoading(null);
    }
  };

  const doTicket = async (poi: string) => {
    setLoading(`ticket:${poi}`);
    try {
      const r = await checkTicket(poi, date);
      setTickets((prev) => ({ ...prev, [poi]: r }));
    } finally {
      setLoading(null);
    }
  };

  if (!destination) {
    return (
      <div className="glass-card flex h-full items-center justify-center rounded-4xl p-6 text-center">
        <p className="text-sm text-mute/60">先生成一份行程，这里就能预订机票/酒店/门票。</p>
      </div>
    );
  }

  const sectionCls = "glass-card flex flex-col gap-2.5 rounded-3xl p-4";
  const bookBtn = (key: string, onClick: () => void) => (
    <button
      onClick={onClick}
      disabled={booked.has(key)}
      className={cn(
        "rounded-xl px-3 py-1 text-xs font-semibold transition-all",
        booked.has(key)
          ? "cursor-default bg-hairline text-mute"
          : "bg-primary text-ink hover:bg-primary-active"
      )}
    >
      {booked.has(key) ? "✓ 已预订" : "预订"}
    </button>
  );

  return (
    <div className="flex h-full flex-col gap-3 overflow-y-auto p-1 scrollbar-thin">
      <div className="flex items-center justify-between px-1">
        <div className="flex items-center gap-2">
          <span className="text-lg font-semibold text-ink">预订 · {destination}</span>
          <span className="rounded-lg bg-hairline px-1.5 py-0.5 font-mono text-[10px] text-mute">mock</span>
        </div>
        <button
          onClick={() => store.setActiveView("itinerary")}
          className="rounded-xl border border-hairline px-3 py-1 text-xs text-mute transition-colors hover:text-ink"
        >
          返回
        </button>
      </div>

      {/* 机票 */}
      <div className={sectionCls}>
        <div className="flex items-center justify-between">
          <span className="text-sm font-semibold text-ink">✈️ 机票</span>
          <div className="flex items-center gap-1.5">
            <input
              value={origin}
              onChange={(e) => setOrigin(e.target.value)}
              className="w-16 rounded-lg border border-hairline bg-canvas px-2 py-1 font-mono text-xs text-ink outline-none"
              placeholder="出发"
            />
            <input
              type="date"
              value={date}
              onChange={(e) => setDate(e.target.value)}
              className="rounded-lg border border-hairline bg-canvas px-2 py-1 font-mono text-[11px] text-ink outline-none"
            />
            <button
              onClick={doFlights}
              disabled={loading === "flights" || !origin}
              className="rounded-xl bg-primary px-3 py-1 text-xs font-semibold text-ink hover:bg-primary-active disabled:opacity-50"
            >
              {loading === "flights" ? "搜索中…" : "搜索"}
            </button>
          </div>
        </div>
        {flights?.map((f) => {
          const key = `flight:${f.flight_no}`;
          return (
            <div key={key} className="flex items-center justify-between rounded-xl border border-hairline px-3 py-2">
              <div className="font-mono text-xs text-ink">
                {f.airline} {f.flight_no} · {f.departure}-{f.arrival} · {f.duration}
              </div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs text-primary">¥{f.price}</span>
                {bookBtn(key, () => markBooked(key))}
              </div>
            </div>
          );
        })}
        {flights?.length === 0 && <p className="text-xs text-mute/60">暂无航班</p>}
      </div>

      {/* 酒店 */}
      <div className={sectionCls}>
        <div className="flex items-center justify-between">
          <span className="text-sm font-semibold text-ink">🏨 酒店（{Math.max(tripDays - 1, 0)} 晚）</span>
          <button
            onClick={doHotels}
            disabled={loading === "hotels"}
            className="rounded-xl bg-primary px-3 py-1 text-xs font-semibold text-ink hover:bg-primary-active disabled:opacity-50"
          >
            {loading === "hotels" ? "搜索中…" : "搜索"}
          </button>
        </div>
        {hotels?.map((h) => {
          const key = `hotel:${h.name}`;
          return (
            <div key={key} className="flex items-center justify-between rounded-xl border border-hairline px-3 py-2">
              <div className="font-mono text-xs text-ink">
                {h.name} · {h.district} · ⭐{h.rating}
                {h.has_breakfast ? " · 含早" : ""}
              </div>
              <div className="flex items-center gap-2">
                <span className="font-mono text-xs text-primary">¥{h.price_per_night}/晚</span>
                {bookBtn(key, () => markBooked(key))}
              </div>
            </div>
          );
        })}
        {hotels?.length === 0 && <p className="text-xs text-mute/60">暂无酒店</p>}
      </div>

      {/* 门票 */}
      {pois.length > 0 && (
        <div className={sectionCls}>
          <span className="text-sm font-semibold text-ink">🎫 景点门票</span>
          {pois.map((poi) => {
            const t = tickets[poi];
            const key = `ticket:${poi}`;
            return (
              <div key={poi} className="flex items-center justify-between rounded-xl border border-hairline px-3 py-2">
                <div className="font-mono text-xs text-ink">
                  {poi}
                  {t && (
                    <span className="ml-2 text-mute">
                      ¥{t.ticket_price} · {t.available ? "有票" : "无票"}
                      {t.need_reservation ? " · 需预约" : ""}
                    </span>
                  )}
                </div>
                {!t ? (
                  <button
                    onClick={() => doTicket(poi)}
                    disabled={loading === key}
                    className="rounded-xl border border-hairline px-3 py-1 text-xs text-mute hover:text-ink disabled:opacity-50"
                  >
                    {loading === key ? "查询…" : "查询"}
                  </button>
                ) : (
                  bookBtn(key, () => markBooked(key))
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}
