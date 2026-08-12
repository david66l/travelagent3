"""Reservation handling: filter fixed reservations and emit reminders."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from vrp_solver_service.models import POIInput, ReminderOutput, ReservationInput

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)


class ReservationHandler:
    """Align POI list with user reservations and emit missing-booking reminders."""

    def filter_and_remind(
        self,
        pois: list[POIInput],
        reservations: list[ReservationInput],
    ) -> tuple[list[POIInput], list[ReminderOutput]]:
        """Return filtered POIs and reminder list."""
        reminders: list[ReminderOutput] = []
        reserved_ids = {r.poi_id for r in reservations}

        # Ensure every reservation has a matching POI
        poi_ids = {p.id for p in pois}
        for r in reservations:
            if r.poi_id not in poi_ids:
                reminders.append(
                    ReminderOutput(
                        type="reservation",
                        message=f"预约 '{r.poi_id}' 未在候选 POI 中找到，请人工确认。",
                        poi_id=r.poi_id,
                        date=r.date,
                    )
                )

        # Mark reserved POIs as must_visit
        for p in pois:
            if p.id in reserved_ids:
                p.must_visit = True
                if p.reservation:
                    reminders.append(
                        ReminderOutput(
                            type="reservation",
                            message=f"{p.name} 需要提前预约：{p.reservation}",
                            poi_id=p.id,
                            poi_name=p.name,
                        )
                    )

        # POIs that require reservation but user did not provide one => reminder
        for p in pois:
            if p.reservation and p.id not in reserved_ids:
                reminders.append(
                    ReminderOutput(
                        type="reservation",
                        message=f"{p.name} 建议提前预约：{p.reservation}",
                        poi_id=p.id,
                        poi_name=p.name,
                    )
                )

        return pois, reminders
