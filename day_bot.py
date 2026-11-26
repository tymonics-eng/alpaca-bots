#!/usr/bin/env python3
"""Day-trader bot: opening-range breakout, long only, paper account. Standard library only.
Same rules as the day-trader skill pack (trader-entry, trader-exit, trader-budget, trader-risk):
  * Opening range = the first 15 minutes (9:30-9:45 ET), from 5-minute bars.
  * Entry = the first COMPLETED 5-minute bar after 9:45 that closes above the range high.
  * Stop = the range midpoint (rounded down to the cent). Target = entry + 1R (R = entry - stop).
  * Size = whole shares, the smallest of: max loss / R, position cap, max order, settled cash after the buffer.
  * One bracket order per trade (market entry, stop-loss and take-profit wait at the broker).
  * Day limits: trades per day, total risk at the stop per day, and a stop when today's equity drop hits the limit.
  * No entries after the cutoff; everything the bot opened is closed by flat_by. Early closes come from Alpaca's
    calendar: cutoff = close - 30 min, flat_by = close - 5 min at the latest.
Modes:
  --mode trade    : run the session (started ~9:40 ET). Dry unless --live.
  --mode flatten  : close what the bot opened today (started ~15:50 ET as a backstop). Dry unless --live.
  --replay DATE   : replay a past session from 5-minute bars and print the trades it would have made. No orders.
Orders only with --live, only on the paper endpoint. Every order the bot sends has client_order_id "dtb-...".
"""
import sys, os, re, json, time, csv, ssl, math, argparse, uuid, urllib.request, urllib.parse, urllib.error
from datetime import datetime, timezone, timedelta, date
from pathlib import Path
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
HERE = Path(__file__).resolve().parent
LOG = HERE / "bot_logs_day"
TR = "https://paper-api.alpaca.markets/v2"     # paper only, on purpose
DATA = "https://data.alpaca.markets"
PREFIX = "dtb"
DEFAULTS = {
    "budget": 21.20, "cash_buffer_pct": 10, "max_position_pct": 75, "max_order_usd": 16, "min_order_usd": 5,
    "max_loss_per_trade_usd": 1.0, "max_daily_loss_usd": 2.0, "max_trades_per_day": 3, "min_reward_risk": 1.0,
    "target_r": 1.0, "or_minutes": 15, "no_entries_before": "09:45", "no_entries_after": "15:30", "flat_by": "15:55",
    "min_price": 2.0, "max_price": 15.90, "min_risk_pct": 0.5, "max_chase_r": 0.5, "universe_top": 100,
    "min_or_volume": 5000, "extra_symbols": [], "blocked": [], "poll_seconds": 60, "check_offsets_s": [3, 20], "max_run_minutes": 340,
}

# ---------------------------------------------------------------- pure rules (tested offline)
def cents_down(x): return math.floor(round(x * 100, 6)) / 100
def cents_up(x): return math.ceil(round(x * 100, 6)) / 100
def hm(s): h, m = s.split(":"); return int(h), int(m)
def at(day, s): h, m = hm(s); return datetime(day.year, day.month, day.day, h, m, tzinfo=ET)

def session_times(day, cfg, close_et=None):
    """Open, range end, entry window and flat-by for one day. close_et = the real close (early closes)."""
    open_ = at(day, "09:30")
    or_end = open_ + timedelta(minutes=cfg["or_minutes"])
    first = max(at(day, cfg["no_entries_before"]), or_end)
    last = at(day, cfg["no_entries_after"]); flat = at(day, cfg["flat_by"])
    if close_et is not None:
        last = min(last, close_et - timedelta(minutes=30)); flat = min(flat, close_et - timedelta(minutes=5))
    return {"open": open_, "or_end": or_end, "first_entry": first, "last_entry": last, "flat_by": flat}

def bar_time(b): return datetime.fromisoformat(b["t"].replace("Z", "+00:00")).astimezone(ET)

def opening_range(bars, t):
    """High, low, mid and volume of the bars that start inside [open, or_end). None until the range is complete."""
    rng = [b for b in bars if t["open"] <= bar_time(b) < t["or_end"]]
    need = int((t["or_end"] - t["open"]).total_seconds() // 300)
    if len(rng) < need: return None
    hi = max(b["h"] for b in rng); lo = min(b["l"] for b in rng)
    return {"high": hi, "low": lo, "mid": (hi + lo) / 2, "volume": sum(b["v"] for b in rng)}

def find_trigger(bars, t, now_et):
    """The first completed 5-minute bar after the range that closes above the range high, if it ended in the window."""
    orng = opening_range(bars, t)
    if orng is None: return None
    for b in bars:
        start = bar_time(b); end = start + timedelta(minutes=5)
        if start < t["or_end"] or end > now_et: continue
        if b["c"] > orng["high"]:
            if end < t["first_entry"] or end > t["last_entry"]: return {"expired": True, "bar_end": end, "range": orng}
            return {"expired": False, "bar_end": end, "close": b["c"], "range": orng, "bar": b}
    return None

def plan_trade(trig, price, cfg):
    """Stop, target and R from the trigger and the price we would pay. Returns (plan, None) or (None, why)."""
    stop = cents_down(trig["range"]["mid"])
    entry = max(price, trig["close"])
    risk = round(entry - stop, 4)
    if price <= stop or risk <= 0: return None, f"price {price:.2f} is at or below the stop {stop:.2f}"
    if risk < entry * cfg["min_risk_pct"] / 100: return None, f"stop too tight ({risk:.2f} a share)"
    if price - trig["close"] > cfg["max_chase_r"] * (trig["close"] - stop): return None, f"ran away: {price:.2f} vs trigger {trig['close']:.2f}"
    target = cents_up(entry + cfg["target_r"] * risk)
    rr = (target - entry) / risk
    if rr + 1e-9 < cfg["min_reward_risk"]: return None, f"reward:risk {rr:.2f} under {cfg['min_reward_risk']}"
    return {"entry": round(entry, 2), "stop": stop, "target": target, "risk": round(risk, 2)}, None

def size(plan, cfg, settled_cash, equity):
    """Whole shares inside the four limits. Returns (qty, binding_limit)."""
    p = plan["entry"]
    lim = {
        "max loss": math.floor(cfg["max_loss_per_trade_usd"] / plan["risk"] + 1e-9),
        "position cap": math.floor(min(cfg["budget"], equity) * cfg["max_position_pct"] / 100 / p + 1e-9),
        "max order": math.floor(cfg["max_order_usd"] / p + 1e-9),
        "cash": math.floor((settled_cash - min(cfg["budget"], equity) * cfg["cash_buffer_pct"] / 100) / p + 1e-9),
    }
    name = min(lim, key=lambda k: lim[k]); qty = max(0, lim[name])
    if qty * p < cfg["min_order_usd"]: return 0, f"{name} (order ${qty * p:.2f} under ${cfg['min_order_usd']})"
    return qty, name

def day_limits_ok(today, new_risk_usd, cfg, equity, last_equity):
    """today = {"trades": n, "risk": usd}. Returns (ok, why)."""
    if today["trades"] >= cfg["max_trades_per_day"]: return False, f"{today['trades']} trades today, limit {cfg['max_trades_per_day']}"
    if today["risk"] + new_risk_usd > cfg["max_daily_loss_usd"] + 1e-9:
        return False, f"day risk {today['risk']:.2f} + {new_risk_usd:.2f} over ${cfg['max_daily_loss_usd']}"
    if last_equity and equity - last_equity <= -cfg["max_daily_loss_usd"]:
        return False, f"down {last_equity - equity:.2f} today, limit ${cfg['max_daily_loss_usd']}"
    return True, ""

def simulate_exit(bars_after, plan, flat_by):
    """Replay: walk the bars after the fill. Stop first when a bar touches both (the cautious read)."""
    for b in bars_after:
        if bar_time(b) + timedelta(minutes=5) > flat_by: return "flat", b["o"]
        if b["l"] <= plan["stop"]: return "stop", min(plan["stop"], b["o"])
        if b["h"] >= plan["target"]: return "target", max(plan["target"], b["o"])   # a gap over the target fills at the open
    return "flat", bars_after[-1]["c"] if bars_after else plan["entry"]

def client_id(day, sym, risk_usd):
    return f"{PREFIX}-{day:%Y%m%d}-{sym}-r{int(round(risk_usd * 100))}-{uuid.uuid4().hex[:6]}"

def parse_client_id(cid):
    m = re.match(rf"{PREFIX}-(\d{{8}})-([A-Z.]+)-r(\d+)-", cid or "")
    return (m.group(1), m.group(2), int(m.group(3)) / 100) if m else None

# ---------------------------------------------------------------- Alpaca I/O
CTX = ssl.create_default_context()
try:
    import certifi; CTX.load_verify_locations(certifi.where())
except Exception: pass

def keys():
    if os.environ.get("APCA_API_KEY_ID"): return os.environ["APCA_API_KEY_ID"], os.environ["APCA_API_SECRET_KEY"]
    for f in (HERE.parent / "day.env", HERE.parent / "fast.env"):   # local testing only (gitignored)
        if f.exists():
            env = dict(re.findall(r'(?:export )?(APCA_\w+)="?([^"\s]+)"?', f.read_text()))
            if "APCA_API_KEY_ID" in env: return env["APCA_API_KEY_ID"], env["APCA_API_SECRET_KEY"]
    sys.exit("no Alpaca keys: set APCA_API_KEY_ID / APCA_API_SECRET_KEY")

H = None
def call(method, url, body=None, params=None):
    global H
    if H is None:
        k, s = keys(); H = {"APCA-API-KEY-ID": k, "APCA-API-SECRET-KEY": s, "Content-Type": "application/json"}
    if not url.startswith("http"): url = TR + url
    if params: url += "?" + urllib.parse.urlencode(params)
    for n in range(5):
        req = urllib.request.Request(url, method=method, headers=H, data=json.dumps(body).encode() if body is not None else None)
        try:
            with urllib.request.urlopen(req, timeout=30, context=CTX) as r:
                t = r.read(); return json.loads(t) if t else {}
        except urllib.error.HTTPError as e:
            msg = e.read().decode()[:300]
            if e.code == 404: return None
            if e.code == 429 or e.code >= 500: time.sleep(3 + 2 * n); continue
            raise RuntimeError(f"{e.code} {method} {url.split('?')[0]}: {msg}")
        except urllib.error.URLError:
            time.sleep(3)
    raise RuntimeError(f"failed {method} {url.split('?')[0]}")

def bars_5m(symbols, start, end):
    out, token = {}, None
    for i in range(0, len(symbols), 100):
        chunk = symbols[i:i + 100]; token = None
        while True:
            p = {"symbols": ",".join(chunk), "timeframe": "5Min", "start": start.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"),
                 "end": end.astimezone(timezone.utc).isoformat().replace("+00:00", "Z"), "feed": "iex", "limit": 10000, "adjustment": "raw"}
            if token: p["page_token"] = token
            r = call("GET", DATA + "/v2/stocks/bars", params=p) or {}
            for s, b in (r.get("bars") or {}).items(): out.setdefault(s, []).extend(b)
            token = r.get("next_page_token")
            if not token: break
    for s in out: out[s].sort(key=lambda b: b["t"])
    return out

def latest_prices(symbols):
    r = call("GET", DATA + "/v2/stocks/trades/latest", params={"symbols": ",".join(symbols), "feed": "iex"}) or {}
    return {s: t["p"] for s, t in (r.get("trades") or {}).items()}

def universe(cfg):
    r = call("GET", DATA + "/v1beta1/screener/stocks/most-actives", params={"by": "volume", "top": cfg["universe_top"]}) or {}
    syms = [x["symbol"] for x in r.get("most_actives", [])] + list(cfg["extra_symbols"])
    syms = [s for s in dict.fromkeys(syms) if re.fullmatch(r"[A-Z]{1,5}", s) and s not in cfg["blocked"]]
    return syms

def close_time(day):
    c = call("GET", "/calendar", params={"start": day.isoformat(), "end": day.isoformat()}) or []
    if not c: return None
    h, m = hm(c[0]["close"]); return datetime(day.year, day.month, day.day, h, m, tzinfo=ET)

def today_orders(day):
    start = datetime(day.year, day.month, day.day, tzinfo=ET).astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    os_ = call("GET", "/orders", params={"status": "all", "after": start, "nested": "true", "limit": 500}) or []
    return [o for o in os_ if (o.get("client_order_id") or "").startswith(f"{PREFIX}-{day:%Y%m%d}-")]

def today_usage(day):
    used = {"trades": 0, "risk": 0.0, "symbols": set()}
    for o in today_orders(day):
        if o.get("side") != "buy": continue
        meta = parse_client_id(o["client_order_id"])
        if not meta: continue
        used["symbols"].add(meta[1])
        if o.get("status") in ("filled", "partially_filled", "new", "accepted", "pending_new"):
            used["trades"] += 1; used["risk"] += meta[2]
    return used

# ---------------------------------------------------------------- logging
def log(*a):
    LOG.mkdir(exist_ok=True)
    line = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC ") + " ".join(str(x) for x in a)
    print(line, flush=True)
    with open(LOG / "day.log", "a") as f: f.write(line + "\n")

def record(row):
    LOG.mkdir(exist_ok=True); f = LOG / "trades.csv"; new = not f.exists()
    with open(f, "a", newline="") as fh:
        w = csv.writer(fh)
        if new: w.writerow(["time_utc", "symbol", "side", "qty", "entry", "stop", "target", "risk_usd", "status", "fill", "note"])
        w.writerow([datetime.now(timezone.utc).isoformat(timespec="seconds")] + row)

def load_cfg(path):
    cfg = dict(DEFAULTS)
    if path and Path(path).exists(): cfg.update(json.loads(Path(path).read_text()))
    return cfg

# ---------------------------------------------------------------- modes
def wait_fill(oid):
    for _ in range(30):
        o = call("GET", f"/orders/{oid}")
        if o and o.get("status") in ("filled", "canceled", "expired", "rejected"): return o
        time.sleep(2)
    return call("GET", f"/orders/{oid}")

def flatten(day, live):
    mine = today_orders(day)
    syms = {parse_client_id(o["client_order_id"])[1] for o in mine if parse_client_id(o["client_order_id"])}
    pos = {p["symbol"]: p for p in (call("GET", "/positions") or []) if p["symbol"] in syms}
    open_ids = []
    for o in mine:
        for x in [o] + (o.get("legs") or []):
            if x.get("status") in ("new", "accepted", "held", "pending_new", "partially_filled"): open_ids.append((x["id"], x["symbol"], x.get("type")))
    if not pos and not open_ids: log("flatten: nothing of the bot's is open"); return
    for oid, sym, typ in open_ids:
        if live: call("DELETE", f"/orders/{oid}")
        log(("" if live else "DRY: would ") + f"cancel {typ} order {sym} {oid[:8]}")
    if live and open_ids: time.sleep(2)
    for sym, p in pos.items():
        if not live: log(f"DRY: would sell {p['qty']} {sym} at market"); continue
        o = call("DELETE", f"/positions/{sym}")
        o = wait_fill(o["id"]) if o and "id" in o else o
        log(f"FLAT {sym} {p['qty']}: {o.get('status') if o else '?'} @ {o.get('filled_avg_price') if o else ''}")
        record([sym, "sell", p["qty"], "", "", "", "", o.get("status") if o else "?", o.get("filled_avg_price") if o else "", "flat by close"])

def trade(cfg, live):
    started = datetime.now(ET); day = started.date()
    clock = call("GET", "/clock")
    cl = close_time(day)
    if cl is None: log("market holiday, nothing to do"); return
    t = session_times(day, cfg, cl)
    stop_at = min(t["flat_by"], started + timedelta(minutes=cfg["max_run_minutes"]))
    log(f"day bot start live={live} close={cl:%H:%M} entries {t['first_entry']:%H:%M}-{t['last_entry']:%H:%M} flat_by {t['flat_by']:%H:%M} ET")
    if not clock["is_open"] and started > cl: log("market already closed"); return
    syms, tried = None, set()
    while True:
        now = datetime.now(ET)
        if now >= t["flat_by"]: flatten(day, live); break
        if now >= stop_at: log("run budget reached; brackets stay at the broker, the flatten job closes the rest"); break
        if now < t["open"] + timedelta(minutes=cfg["or_minutes"]) or now > t["last_entry"] + timedelta(minutes=5):
            time.sleep(min(cfg["poll_seconds"], max(5, (stop_at - now).total_seconds()))); continue
        if syms is None:
            syms = universe(cfg)
            px = latest_prices(syms) if syms else {}
            a = call("GET", "/account")
            cash = min(float(a["cash"]), float(a.get("non_marginable_buying_power") or a["cash"]))
            top = min(cfg["max_price"], cash * (1 - cfg["cash_buffer_pct"] / 100))   # one share must fit in settled cash
            syms = [s for s in syms if cfg["min_price"] <= px.get(s, 0) <= top]
            log(f"settled cash {cash:.2f}; universe: {len(syms)} symbols priced ${cfg['min_price']}-${top:.2f}: {' '.join(syms[:40])}")
            if not syms: log("no symbols in the price band"); break
        try:
            step(cfg, live, day, t, syms, tried, now)
        except RuntimeError as e:
            log("error:", e)
        sleep_s = (next_check(datetime.now(ET), cfg) - datetime.now(ET)).total_seconds()
        time.sleep(max(1, min(sleep_s, (stop_at - datetime.now(ET)).total_seconds() + 1)))
    log("day bot done")

def next_check(now, cfg):
    """The next moment worth looking: a few seconds after each 5-minute bar closes (the skill's trigger moment),
    plus one later look in case the bar reached the data feed late."""
    base = now.replace(second=0, microsecond=0) - timedelta(minutes=now.minute % 5)
    for k in range(3):
        for off in cfg["check_offsets_s"]:
            c = base + timedelta(minutes=5 * k, seconds=off)
            if c > now: return c
    return now + timedelta(seconds=60)

def step(cfg, live, day, t, syms, tried, now):
    used = today_usage(day)
    bars = bars_5m(syms, t["open"], now)
    cands = []
    for s in syms:
        if s in used["symbols"] or s in tried: continue
        b = bars.get(s, [])
        trig = find_trigger(b, t, now)
        if not trig: continue
        if trig["expired"]: tried.add(s); continue
        if trig["range"]["volume"] < cfg["min_or_volume"]: tried.add(s); continue
        cands.append((s, trig))
    if not cands: return
    cands.sort(key=lambda x: -(x[1]["close"] / x[1]["range"]["high"]))
    px = latest_prices([s for s, _ in cands])
    acct = call("GET", "/account")
    equity = float(acct["equity"]); last_eq = float(acct.get("last_equity") or equity)
    settled = min(float(acct["cash"]), float(acct.get("non_marginable_buying_power") or acct["cash"]))
    for s, trig in cands:
        tried.add(s)
        plan, why = plan_trade(trig, px.get(s, trig["close"]), cfg)
        if not plan: log(f"{s} triggered {trig['bar_end']:%H:%M} at {trig['close']:.2f}, skipped: {why}"); continue
        qty, binding = size(plan, cfg, settled, equity)
        if qty < 1: log(f"{s} triggered, skipped: size 0, bound by {binding}"); continue
        risk_usd = round(qty * plan["risk"], 2)
        ok, why = day_limits_ok(used, risk_usd, cfg, equity, last_eq)
        if not ok: log(f"{s} triggered, BLOCK: {why}"); return
        body = {"symbol": s, "qty": str(qty), "side": "buy", "type": "market", "time_in_force": "day", "order_class": "bracket",
                "take_profit": {"limit_price": f"{plan['target']:.2f}"}, "stop_loss": {"stop_price": f"{plan['stop']:.2f}"},
                "client_order_id": client_id(day, s, risk_usd)}
        log(f"{s} TRIGGER {trig['bar_end']:%H:%M} close {trig['close']:.2f} > range high {trig['range']['high']:.2f}; "
            f"buy {qty} ~{plan['entry']:.2f} stop {plan['stop']:.2f} target {plan['target']:.2f} risk ${risk_usd} (bound by {binding})")
        if not live:
            log("DRY: would send", json.dumps(body)); record([s, "buy", qty, plan["entry"], plan["stop"], plan["target"], risk_usd, "dry", "", binding]); return
        o = call("POST", "/orders", body); o = wait_fill(o["id"])
        log(f"{s} order {o.get('status')} {o.get('filled_qty')} @ {o.get('filled_avg_price')}")
        record([s, "buy", qty, plan["entry"], plan["stop"], plan["target"], risk_usd, o.get("status"), o.get("filled_avg_price"), binding])
        return   # one entry per pass; the next pass re-reads cash and limits

def yahoo_bars(symbols, start, end):
    """Keyless 5-minute bars (Yahoo chart JSON), for replays only. Consolidated volume, not IEX."""
    out = {}
    for s in symbols:
        url = (f"https://query2.finance.yahoo.com/v8/finance/chart/{urllib.parse.quote(s)}?interval=5m"
               f"&period1={int(start.timestamp())}&period2={int(end.timestamp())}&includePrePost=false")
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30, context=CTX) as r: res = json.loads(r.read())["chart"]["result"][0]
        except Exception as e:
            print(f"  {s}: no Yahoo data ({e})"); continue
        q = res["indicators"]["quote"][0]; rows = []
        for i, ts_ in enumerate(res.get("timestamp") or []):
            if None in (q["open"][i], q["high"][i], q["low"][i], q["close"][i]): continue
            rows.append({"t": datetime.fromtimestamp(ts_, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "o": q["open"][i], "h": q["high"][i],
                         "l": q["low"][i], "c": q["close"][i], "v": q["volume"][i] or 0})
        out[s] = rows; time.sleep(0.3)
    return out

def replay(cfg, day, symbols, source="alpaca", close_hm=None, start_cash=21.20):
    cl = at(day, close_hm) if close_hm else close_time(day)
    if cl is None: sys.exit(f"{day} is not a trading day")
    t = session_times(day, cfg, cl)
    syms = symbols or universe(cfg)
    bars = (yahoo_bars if source == "yahoo" else bars_5m)(syms, t["open"], cl)
    first = {s: b[0]["o"] for s, b in bars.items() if b}
    top = min(cfg["max_price"], start_cash * (1 - cfg["cash_buffer_pct"] / 100))
    syms = [s for s in syms if cfg["min_price"] <= first.get(s, 0) <= top]
    print(f"Replay {day}: close {cl:%H:%M}, entries {t['first_entry']:%H:%M}-{t['last_entry']:%H:%M}, flat by {t['flat_by']:%H:%M} ET; "
          f"{len(syms)} symbols in ${cfg['min_price']}-${top:.2f}")
    cash = start_cash; settled = cash; used = {"trades": 0, "risk": 0.0}; pnl = 0.0; open_until = None
    # Under $2,000 Alpaca treats the account like a cash account: sale proceeds settle the next day (T+1),
    # so each buy uses up settled cash for the rest of the day, as the live bot sees it.
    events = []
    for s in syms:
        b = bars.get(s, [])
        trig = find_trigger(b, t, cl)
        if trig and not trig["expired"] and trig["range"]["volume"] >= cfg["min_or_volume"]: events.append((trig["bar_end"], s, trig))
    events.sort(key=lambda e: (e[0], -(e[2]["close"] / e[2]["range"]["high"])))
    for end, s, trig in events:
        if open_until and end < open_until: print(f"  {end:%H:%M} {s:6} triggered while the bot was in another trade"); continue
        after = [x for x in bars[s] if bar_time(x) >= end]
        if not after: continue
        fill = after[0]["o"]
        plan, why = plan_trade(trig, fill, cfg)
        if not plan: print(f"  {end:%H:%M} {s:6} skipped: {why}"); continue
        qty, binding = size(plan, cfg, settled, cash)
        if qty < 1: print(f"  {end:%H:%M} {s:6} skipped: size 0 ({binding})"); continue
        risk = round(qty * plan["risk"], 2)
        ok, why = day_limits_ok(used, risk, cfg, cash + pnl, cash)
        if not ok: print(f"  {end:%H:%M} {s:6} BLOCK: {why}"); break
        how, px = simulate_exit(after, plan, t["flat_by"])
        exit_i = next((i for i, x in enumerate(after) if (how == "stop" and x["l"] <= plan["stop"]) or (how == "target" and x["h"] >= plan["target"])), len(after) - 1)
        open_until = bar_time(after[exit_i]) + timedelta(minutes=5)
        p = round(qty * (px - plan["entry"]), 2); pnl += p; used["trades"] += 1; used["risk"] += risk; settled -= qty * plan["entry"]
        print(f"  {end:%H:%M} {s:6} buy {qty} @ {plan['entry']:.2f} stop {plan['stop']:.2f} target {plan['target']:.2f} risk ${risk:.2f} -> {how} {px:.2f}  P&L {p:+.2f}")
    print(f"Trades {used['trades']}, P&L {pnl:+.2f} on ${start_cash:.2f} (before fees; fills at next bar open; stop first when a bar touches both)")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--mode", choices=["trade", "flatten"])
    ap.add_argument("--replay", metavar="YYYY-MM-DD")
    ap.add_argument("--symbols", help="comma-separated, replay only (default: Alpaca most-actives now)")
    ap.add_argument("--yahoo", action="store_true", help="replay from keyless Yahoo bars (needs --symbols and --close)")
    ap.add_argument("--close", metavar="HH:MM", help="replay: the session close in New York time (skips Alpaca's calendar)")
    ap.add_argument("--cash", type=float, default=21.20, help="replay: starting cash (default 21.20)")
    ap.add_argument("--config", default=str(HERE / "day_bot_config.json"))
    ap.add_argument("--live", action="store_true", help="place orders (paper endpoint). Without it: dry run")
    a = ap.parse_args(); cfg = load_cfg(a.config)
    if a.yahoo and not (a.symbols and a.close): ap.error("--yahoo needs --symbols and --close")
    if a.replay: replay(cfg, date.fromisoformat(a.replay), [s.strip().upper() for s in a.symbols.split(",")] if a.symbols else None,
                        "yahoo" if a.yahoo else "alpaca", a.close, a.cash)
    elif a.mode == "trade": trade(cfg, a.live)
    elif a.mode == "flatten": flatten(datetime.now(ET).date(), a.live)
    else: ap.error("pass --mode trade, --mode flatten or --replay DATE")
