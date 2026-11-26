"""Offline tests for day_bot.py rules. Run: python3 -m unittest test_day_bot -v"""
import unittest
from datetime import datetime, timedelta, date
import day_bot as d

DAY = date(2026, 11, 25)
CFG = dict(d.DEFAULTS)

def bars(spec, day=DAY):
    """spec: list of (HH:MM, o, h, l, c, v) in New York time."""
    out = []
    for hhmm, o, h, l, c, v in spec:
        t = d.at(day, hhmm).astimezone(d.timezone.utc)
        out.append({"t": t.strftime("%Y-%m-%dT%H:%M:%SZ"), "o": o, "h": h, "l": l, "c": c, "v": v})
    return out

RANGE = [("09:30", 10.00, 10.20, 9.90, 10.10, 9000), ("09:35", 10.10, 10.25, 10.00, 10.15, 7000), ("09:40", 10.15, 10.30, 10.05, 10.20, 6000)]

class Session(unittest.TestCase):
    def test_normal_day(self):
        t = d.session_times(DAY, CFG, d.at(DAY, "16:00"))
        self.assertEqual((t["first_entry"].strftime("%H:%M"), t["last_entry"].strftime("%H:%M"), t["flat_by"].strftime("%H:%M")), ("09:45", "15:30", "15:55"))

    def test_early_close(self):
        day = date(2026, 11, 27)
        t = d.session_times(day, CFG, d.at(day, "13:00"))
        self.assertEqual((t["last_entry"].strftime("%H:%M"), t["flat_by"].strftime("%H:%M")), ("12:30", "12:55"))

class Signals(unittest.TestCase):
    def setUp(self): self.t = d.session_times(DAY, CFG, d.at(DAY, "16:00"))

    def test_range_needs_all_three_bars(self):
        self.assertIsNone(d.opening_range(bars(RANGE[:2]), self.t))
        r = d.opening_range(bars(RANGE), self.t)
        self.assertEqual((r["high"], r["low"], round(r["mid"], 3), r["volume"]), (10.30, 9.90, 10.10, 22000))

    def test_waits_for_a_completed_close_above(self):
        b = bars(RANGE + [("09:45", 10.20, 10.40, 10.15, 10.28, 5000), ("09:50", 10.28, 10.45, 10.25, 10.38, 8000)])
        self.assertIsNone(d.find_trigger(b, self.t, d.at(DAY, "09:52")))        # 09:50 bar still forming
        trig = d.find_trigger(b, self.t, d.at(DAY, "09:55"))
        self.assertFalse(trig["expired"]); self.assertEqual((trig["close"], trig["bar_end"].strftime("%H:%M")), (10.38, "09:55"))

    def test_wick_above_is_not_a_trigger(self):
        b = bars(RANGE + [("09:45", 10.20, 10.50, 10.15, 10.29, 5000)])
        self.assertIsNone(d.find_trigger(b, self.t, d.at(DAY, "10:00")))

    def test_breakout_after_cutoff_is_expired(self):
        b = bars(RANGE + [("15:30", 10.2, 10.5, 10.2, 10.4, 5000)])
        self.assertTrue(d.find_trigger(b, self.t, d.at(DAY, "15:40"))["expired"])

class Plan(unittest.TestCase):
    trig = {"close": 10.38, "range": {"high": 10.30, "low": 9.90, "mid": 10.10, "volume": 22000}}

    def test_stop_mid_target_1r(self):
        p, why = d.plan_trade(self.trig, 10.38, CFG)
        self.assertIsNone(why); self.assertEqual((p["stop"], p["risk"], p["target"]), (10.10, 0.28, 10.66))

    def test_sizes_from_the_higher_price(self):
        p, _ = d.plan_trade(self.trig, 10.45, CFG)
        self.assertEqual((p["entry"], p["risk"], p["target"]), (10.45, 0.35, 10.80))

    def test_ran_away(self):
        p, why = d.plan_trade(self.trig, 10.60, CFG)   # 0.22 over the trigger, more than 0.5 x 0.28
        self.assertIsNone(p); self.assertIn("ran away", why)

    def test_below_stop(self):
        p, why = d.plan_trade(self.trig, 10.05, CFG)
        self.assertIsNone(p)

    def test_too_tight(self):
        trig = {"close": 10.31, "range": {"high": 10.30, "low": 10.29, "mid": 10.295, "volume": 1}}
        p, why = d.plan_trade(trig, 10.31, CFG)
        self.assertIsNone(p); self.assertIn("tight", why)

class Size(unittest.TestCase):
    def test_position_cap_binds_small_account(self):
        qty, why = d.size({"entry": 10.00, "risk": 0.30}, CFG, 21.20, 21.20)
        self.assertEqual((qty, why), (1, "position cap"))

    def test_max_loss_binds(self):
        cfg = dict(CFG, budget=1000, max_order_usd=1000)
        qty, why = d.size({"entry": 10.00, "risk": 0.30}, cfg, 1000, 1000)
        self.assertEqual((qty, why), (3, "max loss"))

    def test_unsettled_cash_blocks(self):
        qty, why = d.size({"entry": 10.00, "risk": 0.30}, CFG, 8.00, 21.20)
        self.assertEqual(qty, 0)

    def test_under_min_order(self):
        qty, why = d.size({"entry": 4.00, "risk": 0.30}, dict(CFG, max_loss_per_trade_usd=0.30), 21.20, 21.20)
        self.assertEqual(qty, 1 if 4.0 >= CFG["min_order_usd"] else 0)

class Limits(unittest.TestCase):
    def test_trades(self):
        ok, why = d.day_limits_ok({"trades": 3, "risk": 0}, 0.3, CFG, 21.2, 21.2); self.assertFalse(ok)
    def test_day_risk(self):
        ok, why = d.day_limits_ok({"trades": 1, "risk": 1.0}, 1.01, CFG, 21.2, 21.2); self.assertFalse(ok)
        ok, why = d.day_limits_ok({"trades": 1, "risk": 1.0}, 1.0, CFG, 21.2, 21.2); self.assertTrue(ok)
    def test_equity_drop(self):
        ok, why = d.day_limits_ok({"trades": 1, "risk": 0.3}, 0.3, CFG, 19.2, 21.2); self.assertFalse(ok)

class Exits(unittest.TestCase):
    plan = {"entry": 10.38, "stop": 10.10, "target": 10.66}
    def setUp(self): self.flat = d.at(DAY, "15:55")
    def test_target(self):
        self.assertEqual(d.simulate_exit(bars([("10:00", 10.4, 10.7, 10.3, 10.6, 1)]), self.plan, self.flat), ("target", 10.66))
    def test_stop_first_when_both(self):
        self.assertEqual(d.simulate_exit(bars([("10:00", 10.4, 10.7, 10.0, 10.6, 1)]), self.plan, self.flat)[0], "stop")
    def test_gap_below_stop_fills_worse(self):
        self.assertEqual(d.simulate_exit(bars([("10:00", 10.0, 10.1, 9.9, 10.0, 1)]), self.plan, self.flat), ("stop", 10.0))
    def test_flat_by(self):
        self.assertEqual(d.simulate_exit(bars([("15:55", 10.5, 10.6, 10.4, 10.5, 1)]), self.plan, self.flat), ("flat", 10.5))

class Ids(unittest.TestCase):
    def test_roundtrip(self):
        cid = d.client_id(DAY, "NU", 0.28)
        self.assertLessEqual(len(cid), 48)
        self.assertEqual(d.parse_client_id(cid), ("20261125", "NU", 0.28))
        self.assertIsNone(d.parse_client_id("other-bot-123"))

class Timing(unittest.TestCase):
    def test_checks_right_after_each_bar_close(self):
        n = lambda hhmmss: datetime(2026, 11, 25, *map(int, hhmmss.split(":")), tzinfo=d.ET)
        self.assertEqual(d.next_check(n("09:49:50"), CFG), n("09:50:03"))
        self.assertEqual(d.next_check(n("09:50:03"), CFG), n("09:50:20"))
        self.assertEqual(d.next_check(n("09:50:21"), CFG), n("09:55:03"))
        self.assertEqual(d.next_check(n("09:58:59"), CFG), n("10:00:03"))

if __name__ == "__main__":
    unittest.main()
