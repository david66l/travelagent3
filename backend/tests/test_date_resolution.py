"""Tests for natural language date resolution."""

import pytest
from datetime import datetime, timedelta
from agents.intent_recognition import IntentRecognitionAgent


class TestDateResolution:
    """Test Chinese natural language date parsing."""

    def setup_method(self):
        self.agent = IntentRecognitionAgent()
        self.today = datetime.now().date()

    def test_next_monday(self):
        result = self.agent._resolve_date("下周一")
        assert result is not None
        assert result.startswith("20")
        # Should be at least 7 days away
        result_date = datetime.fromisoformat(result).date()
        assert (result_date - self.today).days >= 7

    def test_next_week(self):
        result = self.agent._resolve_date("下周")
        assert result is not None
        result_date = datetime.fromisoformat(result).date()
        # Should be next Monday
        assert result_date.weekday() == 0
        assert result_date > self.today

    def test_tomorrow(self):
        result = self.agent._resolve_date("明天")
        expected = (self.today + timedelta(days=1)).isoformat()
        assert result == expected

    def test_day_after_tomorrow(self):
        result = self.agent._resolve_date("后天")
        expected = (self.today + timedelta(days=2)).isoformat()
        assert result == expected

    def test_month_day(self):
        result = self.agent._resolve_date("5月1日")
        year = self.today.year
        expected = f"{year}-05-01"
        # If May 1st has passed, should be next year
        try:
            result_date = datetime.fromisoformat(result).date()
            assert result_date.month == 5
            assert result_date.day == 1
        except ValueError:
            pytest.fail(f"Invalid date result: {result}")

    def test_date_range(self):
        result = self.agent._resolve_date("5月1日到5月5日")
        assert "to" in result
        parts = result.split(" to ")
        assert len(parts) == 2

    def test_iso_date(self):
        result = self.agent._resolve_date("2026-07-15")
        assert result == "2026-07-15"

    def test_empty_string(self):
        result = self.agent._resolve_date("")
        assert result is None

    def test_this_week(self):
        result = self.agent._resolve_date("这周五")
        assert result is not None
        result_date = datetime.fromisoformat(result).date()
        # Should be within current week (0-6 days)
        delta = (result_date - self.today).days
        assert 0 <= delta < 7
