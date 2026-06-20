// @ts-nocheck
"use client";

import { useEffect, useMemo, useRef } from "react";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import { useChatStore } from "@/stores/chatStore";

// Leaflet 默认图标修复
delete (L.Icon.Default.prototype as any)._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png",
  iconUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png",
  shadowUrl: "https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png",
});

export function TripMap() {
  const store = useChatStore();
  const mapRef = useRef<L.Map | null>(null);
  const containerRef = useRef<HTMLDivElement>(null);

  // Memoize fallback to stable reference so effect deps don't churn.
  const itinerary = useMemo(() => store.itinerary || [], [store.itinerary]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current || itinerary.length === 0) return;

    const map = L.map(containerRef.current).setView([30.57, 104.07], 11);
    L.tileLayer("https://tile.openstreetmap.org/{z}/{x}/{y}.png", {
      attribution: "&copy; OpenStreetMap",
    }).addTo(map);
    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, [itinerary.length]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    map.eachLayer((l) => { if (l instanceof L.Marker || l instanceof L.Polyline) map.removeLayer(l); });

    const markers: L.LatLng[] = [];
    itinerary.forEach((day: any) => {
      (day.activities || []).forEach((act: any) => {
        if (act.lat && act.lng) {
          const pos: L.LatLng = L.latLng(act.lat, act.lng);
          markers.push(pos);
          L.marker(pos).addTo(map).bindPopup(`<b>${act.poi_name}</b>`);
        }
      });
    });
    if (markers.length > 1) L.polyline(markers, { color: "#0e0f0c" }).addTo(map);
    if (markers.length > 0) map.fitBounds(L.latLngBounds(markers), { padding: [30, 30] });
  }, [itinerary]);

  if (itinerary.length === 0) return null;

  return <div ref={containerRef} className="h-64 w-full rounded-2xl" style={{ minHeight: 256 }} />;
}
