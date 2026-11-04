#!/usr/bin/env python3
"""FAST news-reaction test bot (paper account PA3GNIYTI). Standard library only.
Rule (fixed, pre-registered): a fresh stock-specific headline with a positive keyword score
-> buy that stock at market (notional, fractional), sell after HOLD_MIN minutes or before the close.
Back-tests found no edge (net loss about equal to trading cost). This is a paper experiment only.
Usage: python3 fast_bot.py [--dry] [--replay-hours N] [--max-minutes N]
"""
import sys, os, re, json, time, csv, ssl, urllib.request, urllib.parse, urllib.error
from datetime import datetime, timezone, timedelta
from pathlib import Path
HERE = Path(__file__).resolve().parent
LOG = HERE / "bot_logs_fast"; LOG.mkdir(exist_ok=True)
UNIVERSE = "AAPL MSFT NVDA AMZN GOOGL META TSLA AMD NFLX JPM PLTR COIN".split()
HOLD_MIN = 15; MAX_TRADES_DAY = 3; POLL_SEC = 20; MIN_USD = 1.0; COOLDOWN_MIN = 30
POS = r"\b(beats?|tops?|raises?|raised|upgrades?|upgraded|initiates? .{0,30}(buy|outperform|overweight)|record|surges?|soars?|jumps?|rall(y|ies)|wins?|approv(al|ed)|buyback|partnership|strong|accelerat\w+|boosts?)\b"
NEG = r"\b(misses?|cuts?|downgrades?|downgraded|lowers?|lowered|plunges?|falls?|slumps?|tumbles?|drops?|probe|lawsuit|sued|investigation|recall|warns?|weak|delays?|halts?|fines?|layoffs?)\b"
JUNK = r"(stocks? (to|that)|movers|why is|what'?s going on|top \d|options? (activity|trad)|bulls and bears|earnings (preview|scheduled)|price target|trading|premarket|after-?hours|etf|magnificent|this week|today)"
def score(h):
    h = h.lower()
    if re.search(JUNK, h): return None
    return len(re.findall(POS, h)) - len(re.findall(NEG, h))
if os.environ.get("APCA_API_KEY_ID"):
    env = {"APCA_API_KEY_ID": os.environ["APCA_API_KEY_ID"], "APCA_API_SECRET_KEY": os.environ["APCA_API_SECRET_KEY"]}
else:
    env = dict(re.findall(r'export (APCA_\w+)="?([^"\s]+)"?', (HERE / "fast.env").read_text()))
H = {"APCA-API-KEY-ID": env["APCA_API_KEY_ID"], "APCA-API-SECRET-KEY": env["APCA_API_SECRET_KEY"], "Content-Type": "application/json"}
TR = "https://paper-api.alpaca.markets/v2"; DATA = "https://data.alpaca.markets"
CTX = ssl.create_default_context()
try:
    import certifi; CTX.load_verify_locations(certifi.where())   # macOS python.org builds ship no CA certs
except Exception: pass
def call(method, url, body=None, tries=5):
    for k in range(tries):
        req = urllib.request.Request(url, method=method, headers=H, data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
                t = r.read(); return json.loads(t) if t else {}
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300]
            if e.code == 429: time.sleep(3 + 2 * k); continue
            raise RuntimeError(f"{e.code} {method} {url.split('?')[0]}: {msg}")
        except Exception as e:
            if k == tries - 1: raise
            time.sleep(2)
def now(): return datetime.now(timezone.utc)
def log(*a):
    line = f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} " + " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG / "fast.log", "a") as f: f.write(line + "\n")
def wait_fill(oid):
    for _ in range(30):
        o = call("GET", f"{TR}/orders/{oid}")
        if o.get("status") == "filled": return o
        if o.get("status") in ("canceled", "expired", "rejected"): return o
        time.sleep(1)
    return call("GET", f"{TR}/orders/{oid}")
def main():
    dry = "--dry" in sys.argv
    replay = float(sys.argv[sys.argv.index("--replay-hours") + 1]) if "--replay-hours" in sys.argv else 0
    max_min = float(sys.argv[sys.argv.index("--max-minutes") + 1]) if "--max-minutes" in sys.argv else 1e9
    t_start = time.time()
    log("fast bot start", "DRY" if dry else "LIVE", "replay_h", replay)
    clock = call("GET", f"{TR}/clock")
    # wait for the open (scheduler starts us early); give up if the market is closed for the day
    while not clock["is_open"]:
        nxt = datetime.fromisoformat(clock["next_open"]); wait = (nxt - now()).total_seconds()
        if wait > 3 * 3600 or wait < 0 or time.time() - t_start > max_min * 60: log("market closed, exiting"); return
        time.sleep(min(60, max(5, wait))); clock = call("GET", f"{TR}/clock")
    seen_from = now() - timedelta(hours=replay) if replay else now()
    last_id = 0; seen_ids = set(); cool = {}; trades_today = 0; day = datetime.now().date()
    pos = None   # dict(sym, qty_usd, t_in, order)
    while time.time() - t_start < max_min * 60:
        clock = call("GET", f"{TR}/clock"); close_t = datetime.fromisoformat(clock["next_close"]) if clock["is_open"] else None
        if not clock["is_open"]:
            if pos and not dry: exit_pos(pos, "market closed"); pos = None
            log("market closed, exiting"); return
        mins_to_close = (close_t - now()).total_seconds() / 60
        # exit logic
        if pos and (now() - pos["t_in"]).total_seconds() >= HOLD_MIN * 60 or (pos and mins_to_close <= 6):
            exit_pos(pos, "timeout" if mins_to_close > 6 else "near close", dry); pos = None
        # poll news
        q = {"symbols": ",".join(UNIVERSE), "start": seen_from.strftime("%Y-%m-%dT%H:%M:%SZ"), "limit": 50, "sort": "asc", "include_content": "false"}
        try: news = call("GET", f"{DATA}/v1beta1/news?" + urllib.parse.urlencode(q)).get("news", [])
        except Exception as e: log("news error", e); time.sleep(POLL_SEC); continue
        for a in news:
            if a["id"] in seen_ids: continue
            seen_ids.add(a["id"])
            created = datetime.fromisoformat(a["created_at"].replace("Z", "+00:00"))
            syms = [s for s in a.get("symbols", [])]
            if len(syms) > 2: continue
            tick = [s for s in syms if s in UNIVERSE]
            if len(tick) != 1: continue
            sc = score(a["headline"]); s = tick[0]
            if sc is None or sc < 1: continue
            age = (now() - created).total_seconds()
            if not replay and age > 180: log("skip stale", s, f"{age:.0f}s", a["headline"][:70]); continue
            if pos: log("skip busy", s, a["headline"][:70]); continue
            if trades_today >= MAX_TRADES_DAY: log("skip daily cap", s); continue
            if s in cool and (now() - cool[s]).total_seconds() < COOLDOWN_MIN * 60: log("skip cooldown", s); continue
            if mins_to_close < HOLD_MIN + 10: log("skip too close to close", s); continue
            acct = call("GET", f"{TR}/account")
            bp = float(acct.get("non_marginable_buying_power") or acct.get("cash") or 0); usd = round(min(bp, float(acct["cash"])) - 0.01, 2)
            if usd < MIN_USD: log("skip no settled cash", s, f"bp={bp:.2f}"); continue
            log("SIGNAL", s, f"score={sc} age={age:.0f}s usd={usd}", a["headline"][:90])
            if dry: pos = {"sym": s, "t_in": now(), "hl": a["headline"], "usd": usd, "dry": True, "px_in": None}; cool[s] = now(); trades_today += 1; continue
            try:
                o = call("POST", f"{TR}/orders", {"symbol": s, "notional": str(usd), "side": "buy", "type": "market", "time_in_force": "day"})
                o = wait_fill(o["id"])
            except Exception as e: log("order error", e); continue
            if o.get("status") != "filled": log("buy not filled", o.get("status")); continue
            pos = {"sym": s, "t_in": now(), "hl": a["headline"], "usd": usd, "px_in": float(o["filled_avg_price"]), "qty": float(o["filled_qty"]), "score": sc}
            cool[s] = now(); trades_today += 1; log("BOUGHT", s, pos["px_in"], pos["qty"])
        seen_from = max(seen_from, now() - timedelta(minutes=30)) if not replay else seen_from
        time.sleep(POLL_SEC)
    if pos and not dry: exit_pos(pos, "script end")
    log("fast bot end")
def exit_pos(pos, why, dry=False):
    s = pos["sym"]
    if dry or pos.get("dry"): log("DRY EXIT", s, why); return
    try:
        o = call("DELETE", f"{TR}/positions/{s}"); o = wait_fill(o["id"]) if "id" in o else o
        px = float(o.get("filled_avg_price") or 0)
    except Exception as e: log("exit error", e); return
    pnl_bp = (px / pos["px_in"] - 1) * 1e4 if pos["px_in"] and px else 0
    log("SOLD", s, px, f"pnl={pnl_bp:.1f}bp", why)
    new = not (LOG / "trades.csv").exists()
    with open(LOG / "trades.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["time_in", "symbol", "px_in", "px_out", "pnl_bp", "usd", "score", "why_exit", "headline"])
        w.writerow([pos["t_in"].isoformat(), s, pos["px_in"], px, round(pnl_bp, 1), pos["usd"], pos.get("score"), why, pos["hl"]])
if __name__ == "__main__": main()
