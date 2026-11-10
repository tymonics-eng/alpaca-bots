#!/usr/bin/env python3
"""Overnight QQQ bot (fast-bot paper account). Standard library only.
Research basis (research2/, Nov 2026): holding QQQ only from close to next open was the one intraday-type
strategy that stayed positive in the locked 2025-07..2026-10 test (+0.07%/night after costs, t=1.6). Modest, not guaranteed.
  --mode buy  : started ~1h before the close; waits until 5 min before the close, buys QQQ with all cash.
  --mode sell : started before the open; waits for the open (+1 min), sells everything.
  --dry       : log what it would do, place no orders.
"""
import sys, os, re, json, time, csv, ssl, urllib.request, urllib.error
from datetime import datetime, timezone
from pathlib import Path
SYM = "QQQ"; BUY_BEFORE_CLOSE_MIN = int(os.environ.get("TEST_BUY_BEFORE_MIN", 5)); LATEST_BUY_MIN = 2; MAX_WAIT_MIN = 100
HERE = Path(__file__).resolve().parent; LOG = HERE / "bot_logs_overnight"; LOG.mkdir(exist_ok=True)
CTX = ssl.create_default_context()
try:
    import certifi; CTX.load_verify_locations(certifi.where())
except Exception: pass
if os.environ.get("APCA_API_KEY_ID"):
    K, S = os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"]
else:   # local testing only
    env = dict(re.findall(r'export (APCA_\w+)="?([^"\s]+)"?', (HERE.parent / "run.sh").read_text())); K, S = env["APCA_API_KEY_ID"], env["APCA_API_SECRET_KEY"]
H = {"APCA-API-KEY-ID": K, "APCA-API-SECRET-KEY": S, "Content-Type": "application/json"}
TR = "https://paper-api.alpaca.markets/v2"
def call(method, path, body=None):
    for k in range(5):
        req = urllib.request.Request(TR + path, method=method, headers=H, data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
                t = r.read(); return json.loads(t) if t else {}
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300]
            if e.code == 404: return None
            if e.code == 429 or e.code >= 500: time.sleep(3 + 2 * k); continue
            raise RuntimeError(f"{e.code} {method} {path}: {msg}")
        except urllib.error.URLError:
            time.sleep(3)
    raise RuntimeError(f"failed {method} {path}")
def now(): return datetime.now(timezone.utc)
def ts(s): return datetime.fromisoformat(s.replace("Z", "+00:00"))
def log(*a):
    line = now().strftime("%Y-%m-%d %H:%M:%S UTC ") + " ".join(str(x) for x in a); print(line, flush=True)
    with open(LOG / "overnight.log", "a") as f: f.write(line + "\n")
def record(side, o, why):
    acct = call("GET", "/account"); new = not (LOG / "trades.csv").exists()
    with open(LOG / "trades.csv", "a", newline="") as f:
        w = csv.writer(f)
        if new: w.writerow(["time_utc", "side", "symbol", "qty", "price", "notional", "equity_after", "note"])
        w.writerow([now().isoformat(timespec="seconds"), side, o.get("symbol"), o.get("filled_qty"), o.get("filled_avg_price"),
                    round(float(o.get("filled_qty") or 0) * float(o.get("filled_avg_price") or 0), 2), acct.get("equity"), why])
def wait_fill(oid):
    for _ in range(60):
        o = call("GET", f"/orders/{oid}")
        if o and o.get("status") in ("filled", "canceled", "expired", "rejected"): return o
        time.sleep(2)
    return call("GET", f"/orders/{oid}")
def sleep_until(t, label):
    secs = (t - now()).total_seconds()
    if secs > 0: log(f"waiting {secs/60:.1f} min until {label}"); time.sleep(secs)
def do_buy(dry):
    c = call("GET", "/clock")
    if not c["is_open"]: log("market closed, nothing to do"); return
    close = ts(c["next_close"]); left = (close - now()).total_seconds() / 60
    if left > MAX_WAIT_MIN: log(f"close is {left:.0f} min away (other DST slot), exiting"); return
    if left < LATEST_BUY_MIN: log(f"only {left:.1f} min to close, too late, skipping today"); return
    if left > BUY_BEFORE_CLOSE_MIN: sleep_until(close.replace(second=0) - __import__("datetime").timedelta(minutes=BUY_BEFORE_CLOSE_MIN), "5 min before close")
    if call("GET", f"/positions/{SYM}"): log("already holding", SYM, "- no new buy"); return
    call("DELETE", "/orders")
    a = call("GET", "/account"); cash = min(float(a["cash"]), float(a.get("non_marginable_buying_power") or a["cash"]))
    usd = round(cash - 0.05, 2)
    if os.environ.get("TEST_CAP_USD"): usd = min(usd, float(os.environ["TEST_CAP_USD"]))   # testing only
    if usd < 1: log(f"not enough cash ({cash:.2f})"); return
    if dry: log(f"DRY: would buy ${usd} of {SYM}"); return
    o = wait_fill(call("POST", "/orders", {"symbol": SYM, "notional": str(usd), "side": "buy", "type": "market", "time_in_force": "day"})["id"])
    log("BUY", o.get("status"), o.get("filled_qty"), "@", o.get("filled_avg_price")); 
    if o.get("status") == "filled": record("buy", o, "close entry")
def do_sell(dry):
    c = call("GET", "/clock")
    if not c["is_open"]:
        nxt = ts(c["next_open"]); wait = (nxt - now()).total_seconds() / 60
        if wait > MAX_WAIT_MIN or wait < 0: log(f"open is {wait:.0f} min away, exiting"); return
        sleep_until(nxt.replace(second=0) + __import__("datetime").timedelta(minutes=1), "1 min after open")
    pos = call("GET", "/positions") or []
    if os.environ.get("TEST_ONLY_SYM"): pos = [p for p in pos if p["symbol"] == SYM]   # testing only
    if not pos: log("no positions to sell"); return
    call("DELETE", "/orders")
    for p in pos:
        if dry: log("DRY: would sell", p["symbol"], p["qty"]); continue
        o = call("DELETE", f"/positions/{p['symbol']}")
        o = wait_fill(o["id"]) if o and "id" in o else o
        log("SELL", p["symbol"], o.get("status") if o else "?", o.get("filled_qty") if o else "", "@", o.get("filled_avg_price") if o else "")
        if o and o.get("status") == "filled": record("sell", o, "open exit")
if __name__ == "__main__":
    mode = sys.argv[sys.argv.index("--mode") + 1]; dry = "--dry" in sys.argv
    log(f"overnight bot start mode={mode} dry={dry}")
    (do_buy if mode == "buy" else do_sell)(dry)
    log("done")
