#!/usr/bin/env python3
"""
ETF trend rotation (default) and intraday momentum scanner for an Alpaca PAPER account (v3).

SETUP
    pip install alpaca-py pandas numpy
    export APCA_API_KEY_ID="your paper key"
    export APCA_API_SECRET_KEY="your paper secret"

USAGE, TREND MODE (default; the only strategy that held up on the 2017-2020 locked test)
    python alpaca_aggressive_bot.py                        # prints today's target portfolio and orders, sends nothing
    python alpaca_aggressive_bot.py --live                 # sends fractional market orders to the PAPER account
    Run it once a day while the market is open (15:35-22:00 Warsaw time). It usually trades
    only every few days, when one of its four staggered tranches rebalances.

USAGE, MOMENTUM MODE (old intraday strategy; lost money on 15 years of history, kept for reference)
    python alpaca_aggressive_bot.py --mode momentum                  # scan only
    python alpaca_aggressive_bot.py --mode momentum --live           # bracket orders to PAPER account
    python alpaca_aggressive_bot.py --backtest --archive ~/Downloads/archive.zip --years 15

TREND MODE RULES (found by the research run, tested once on 2017-04 to 2020-04)
    * Universe: 19 fixed ETFs (US/intl stocks, 9 sectors, bonds, gold).
    * Score = average of the 1, 3, 6 and 12-month returns. Keep the top 9 with a positive score.
    * Weight them by inverse 63-day volatility, scale the whole portfolio so its recent
      volatility is about 10% a year, and put whatever is left in IEF (7-10y Treasuries).
    * Four staggered sub-portfolios each rebalance every 21 trading days; the target is their average.
    Locked-test result at 10 bps costs: 3.1%/yr, Sharpe 0.31, worst drawdown -21%,
    versus SPY buy and hold 3.2%/yr, Sharpe 0.26, worst drawdown -34% (period includes March 2020).
    It roughly matched SPY's return with less risk. It did NOT beat SPY's return, and it is not a
    profit-every-day system (about 55% green days).

WHAT CHANGED IN V2
    * Backtest mode with green-day %, expectancy, drawdown, and a walk-forward
      sweep of MIN_SCORE (tune on the first 60% of history, judge on the last 40%).
    * Regime filter: sits out days when SPY is choppy or too volatile, and only
      trades in the direction of SPY's trend. A flat day is better than a forced loss.
    * Higher conviction threshold (MIN_SCORE 5.0).
    * Small-account mode: allows a 1-share minimum when the risk is acceptable.
    * Pattern day trader guard for margin accounts under $25k.
    * No new entries late in the day, and positions are flattened before the close
      (the strategy is an intraday one, matching how the backtest exits).
    * Data feed setting (IEX for free plans) so requests do not fail on SIP limits.

IMPORTANT
    * Paper only. The client is hard-coded to paper=True.
    * No strategy wins every day. The goal here is a positive edge and a high share
      of green days, and the backtest tells you whether that exists.
    * The backtest covers the daily-bar part of the score only (no VWAP or news),
      uses today's watchlist (survivorship bias) and assumes next-day-open entries.
      Treat results as a rough guide, not proof.
    * Research tool, not financial advice. You run it, you own the results.
"""

import argparse
import csv
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

import zipfile

try:
    from alpaca.data.enums import DataFeed
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.historical.news import NewsClient
    from alpaca.data.requests import NewsRequest, StockBarsRequest, StockSnapshotRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.trading.client import TradingClient
    from alpaca.trading.enums import OrderClass, OrderSide, QueryOrderStatus, TimeInForce
    from alpaca.trading.requests import (
        GetOrdersRequest,
        MarketOrderRequest,
        StopLossRequest,
        TakeProfitRequest,
    )
    HAVE_ALPACA = True
except ImportError:
    HAVE_ALPACA = False     # archive backtests work without alpaca-py installed

try:
    from alpaca.data.historical.screener import ScreenerClient
    from alpaca.data.requests import MarketMoversRequest, MostActivesRequest
    HAVE_SCREENER = True
except Exception:
    HAVE_SCREENER = False

# ----------------------------------------------------------------------------
# CONFIG (edit these)
# ----------------------------------------------------------------------------
WATCHLIST = [
    "SPY", "QQQ", "IWM", "NVDA", "TSLA", "AMD", "AAPL", "MSFT", "META", "AMZN",
    "GOOGL", "PLTR", "COIN", "MSTR", "SMCI", "AVGO", "NFLX", "SOFI", "RIVN", "HOOD",
]

DATA_FEED = "iex"               # "iex" for free plans, "sip" if you pay for full market data
DAILY_TARGET_PCT = 2.0          # stop opening new trades once up this much on the day
DAILY_MAX_LOSS_PCT = 3.0        # kill switch: stop trading when down this much
RISK_PER_TRADE = 0.0075         # 0.75% of equity risked per trade (distance to stop)
MAX_POSITION_PCT = 0.25         # no single position above 25% of equity
MAX_OPEN_POSITIONS = 6
MAX_NEW_TRADES_PER_SCAN = 3
ALLOW_SHORTS = True
MIN_PRICE = 2.0
MIN_AVG_DOLLAR_VOLUME = 20_000_000   # 20d average dollar volume (auto-scaled down on IEX)
MIN_SCORE = 5.0                 # minimum absolute score to trade (higher = pickier)
ATR_STOP_MULT = 1.5             # stop distance = 1.5 x ATR
REWARD_RISK = 2.0               # take profit = 2 x stop distance
UNIVERSE_MAX = 80               # cap on symbols analysed per scan
NEWS_LOOKBACK_HOURS = 24
LOG_DIR = Path("bot_logs")

# Regime filter (sit out bad days)
REGIME_FILTER = True
SPY_MAX_ATR_PCT = 2.5           # sit out if SPY's daily ATR% is above this (too wild)
SPY_MIN_TREND_PCT = 0.3         # sit out if |EMA9 - EMA21| of SPY is below this % (chop)
TREND_ALIGN = True              # longs only when SPY above EMA21, shorts only when below

# Small-account mode
SMALL_ACCOUNT_MIN_ONE_SHARE = True
MAX_ONE_SHARE_RISK_PCT = 0.10   # allow 1 share if its stop distance is under 10% of the stake

# Virtual stake: the bot may only use this much of the account, no matter how big the balance is.
# Day 1 it can use STAKE_START_USD. Every later day it can use STAKE_START_USD plus the total
# profit or loss made since the first run (so the whole stake if flat, more after wins, less after
# losses). Open positions are counted at market value. The stake never uses leverage.
# Use this account only for the bot, because all account P/L is treated as the bot's P/L.
# To restart the experiment from $20, delete bot_logs/stake_state.json.
STAKE_ENABLED = True
STAKE_START_USD = 20.0
STATE_FILE = LOG_DIR / "stake_state.json"

# Pattern day trader guard (real margin accounts under $25k are limited to 3 day trades per 5 days)
PDT_GUARD = True

# Intraday timing
NO_NEW_TRADES_LAST_MIN = 45     # no new entries in the last 45 minutes of the session
FLATTEN_ENABLED = True          # close everything shortly before the close
FLATTEN_BEFORE_CLOSE_MIN = 10

# Backtest assumptions
BACKTEST_SLIPPAGE_BPS = 5       # per side
BACKTEST_START_EQUITY = 10_000
SWEEP_SCORES = [3.0, 4.0, 5.0, 6.0, 7.0]

# Trend mode settings (the research values; changing them means the backtest no longer applies)
TREND_RISKY = ["SPY", "QQQ", "IWM", "DIA", "EFA", "EEM", "XLB", "XLE", "XLF", "XLI", "XLK", "XLP",
               "XLU", "XLV", "XLY", "TLT", "IEF", "LQD", "GLD"]
TREND_FALLBACK = "IEF"
TREND_LOOKBACKS = (21, 63, 126, 252)
TREND_TOP_N = 9
TREND_VOL_TARGET = 0.10
TREND_VOL_WINDOW = 63
TREND_TRANCHES = 4
TREND_PERIOD = 21
MIN_ORDER_USD = 1.00            # Alpaca's minimum notional order; smaller differences are skipped

POSITIVE_WORDS = {"beat", "beats", "surge", "soar", "upgrade", "record", "raises", "growth",
                  "approval", "partnership", "bullish", "outperform", "buyback", "wins"}
NEGATIVE_WORDS = {"miss", "misses", "plunge", "downgrade", "lawsuit", "probe", "recall", "cuts",
                  "bearish", "underperform", "fraud", "investigation", "halt", "bankruptcy", "warns"}


def feed():
    return DataFeed(DATA_FEED.lower())


def min_dollar_volume():
    return MIN_AVG_DOLLAR_VOLUME * (0.03 if DATA_FEED.lower() == "iex" else 1.0)


# ----------------------------------------------------------------------------
# Clients
# ----------------------------------------------------------------------------
def make_clients():
    if not HAVE_ALPACA:
        sys.exit("alpaca-py is not installed. Run: pip install alpaca-py")
    key = os.environ.get("APCA_API_KEY_ID")
    secret = os.environ.get("APCA_API_SECRET_KEY")
    if not key or not secret:
        sys.exit("Set APCA_API_KEY_ID and APCA_API_SECRET_KEY (paper keys).")
    trading = TradingClient(key, secret, paper=True)  # PAPER ONLY
    data = StockHistoricalDataClient(key, secret)
    news = NewsClient(key, secret)
    screener = ScreenerClient(key, secret) if HAVE_SCREENER else None
    return trading, data, news, screener


# ----------------------------------------------------------------------------
# Data pulls
# ----------------------------------------------------------------------------
def build_universe(screener):
    symbols = list(WATCHLIST)
    notes = {}
    if screener is not None:
        try:
            actives = screener.get_most_actives(MostActivesRequest(top=30))
            for a in actives.most_actives:
                symbols.append(a.symbol)
                notes[a.symbol] = "active"
        except Exception as e:
            print(f"[warn] most actives failed: {e}")
        try:
            movers = screener.get_market_movers(MarketMoversRequest(top=20))
            for m in movers.gainers + movers.losers:
                symbols.append(m.symbol)
                notes[m.symbol] = "mover"
        except Exception as e:
            print(f"[warn] movers failed: {e}")
    seen, out = set(), []
    for s in symbols:
        if s not in seen and s.isalpha():   # skip warrants, rights, odd tickers
            seen.add(s)
            out.append(s)
    out = out[:UNIVERSE_MAX]
    if "SPY" not in out:
        out.append("SPY")                   # needed for the regime filter
    return out, notes


def get_daily_bars(data, symbols, calendar_days=200):
    start = datetime.now(timezone.utc) - timedelta(days=calendar_days)
    req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start,
                           adjustment="split", feed=feed())
    return data.get_stock_bars(req).df


def get_intraday_bars(data, symbols):
    start = datetime.now(timezone.utc) - timedelta(days=3)
    req = StockBarsRequest(symbol_or_symbols=symbols,
                           timeframe=TimeFrame(5, TimeFrameUnit.Minute), start=start, feed=feed())
    return data.get_stock_bars(req).df


def get_snapshots(data, symbols):
    return data.get_stock_snapshot(StockSnapshotRequest(symbol_or_symbols=symbols, feed=feed()))


def get_news_scores(news_client, symbols):
    """Very simple keyword sentiment over recent headlines. Returns {symbol: (score, count)}."""
    start = datetime.now(timezone.utc) - timedelta(hours=NEWS_LOOKBACK_HOURS)
    scores = {s: [0, 0] for s in symbols}
    chunk = 20
    for i in range(0, len(symbols), chunk):
        batch = symbols[i:i + chunk]
        try:
            res = news_client.get_news(NewsRequest(symbols=",".join(batch), start=start, limit=50))
            articles = res.data.get("news", []) if hasattr(res, "data") else []
        except Exception as e:
            print(f"[warn] news failed: {e}")
            continue
        for art in articles:
            text = f"{art.headline} {art.summary or ''}".lower()
            words = set(text.replace(",", " ").replace(".", " ").split())
            s = len(words & POSITIVE_WORDS) - len(words & NEGATIVE_WORDS)
            for sym in art.symbols:
                if sym in scores:
                    scores[sym][0] += s
                    scores[sym][1] += 1
    return {k: tuple(v) for k, v in scores.items()}


# ----------------------------------------------------------------------------
# Indicators (shared by live scanner and backtest)
# ----------------------------------------------------------------------------
def rsi(series, n=14):
    delta = series.diff()
    up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    down = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    rs = up / down.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(df, n=14):
    prev_close = df["close"].shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def add_indicators(daily):
    """Takes one symbol's daily bars, returns them with indicators and the core daily score."""
    d = daily.copy()
    c = d["close"]
    d["ema9"] = c.ewm(span=9, adjust=False).mean()
    d["ema21"] = c.ewm(span=21, adjust=False).mean()
    d["ema50"] = c.ewm(span=50, adjust=False).mean()
    d["rsi"] = rsi(c)
    d["atr"] = atr(d)
    d["mom5"] = (c / c.shift(5) - 1) * 100
    d["mom20"] = (c / c.shift(20) - 1) * 100
    d["gap"] = (d["open"] / c.shift(1) - 1) * 100
    d["rvol"] = d["volume"] / d["volume"].rolling(20).mean().shift(1)

    up = (d["ema9"] > d["ema21"]) & (d["ema21"] > d["ema50"])
    dn = (d["ema9"] < d["ema21"]) & (d["ema21"] < d["ema50"])
    s = pd.Series(np.where(up, 1.0, np.where(dn, -1.0, 0.0)), index=d.index)
    s = s + pd.Series(np.where(c > d["ema21"], 1.0, -1.0), index=d.index)
    s = s + (d["mom5"] / 3).clip(-2, 2) + (d["mom20"] / 8).clip(-1.5, 1.5) + (d["gap"] / 2).clip(-1, 1)
    mult = 1.0 + (d["rvol"] - 1.5).clip(lower=0, upper=2) * 0.15   # volume confirms the direction
    s = s * mult.fillna(1.0)
    s = s - (d["rsi"] > 80).astype(float) + (d["rsi"] < 20).astype(float)
    d["core"] = s
    return d


def regime_from_spy(spy_daily):
    """Returns a DataFrame indexed like SPY with reg_ok (trade today?) and bias (+1 / -1)."""
    d = add_indicators(spy_daily)
    trend_pct = (d["ema9"] - d["ema21"]).abs() / d["close"] * 100
    atr_pct = d["atr"] / d["close"] * 100
    out = pd.DataFrame(index=d.index)
    out["trend_pct"] = trend_pct
    out["atr_pct"] = atr_pct
    out["reg_ok"] = (trend_pct >= SPY_MIN_TREND_PCT) & (atr_pct <= SPY_MAX_ATR_PCT)
    out["bias"] = np.where(d["close"] > d["ema21"], 1, -1)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize()
    return out


def analyse_symbol(sym, daily, intraday, snap, news):
    if daily is None or len(daily) < 55:
        return None
    d = add_indicators(daily)
    last = d.iloc[-1]
    if pd.isna(last["core"]):
        return None
    price = float(snap.latest_trade.price) if snap and snap.latest_trade else float(last["close"])
    if price < MIN_PRICE:
        return None
    avg_dollar_vol = float((d["close"] * d["volume"]).tail(20).mean())
    if avg_dollar_vol < min_dollar_volume():
        return None

    high20 = float(d["high"].tail(20).max())
    low20 = float(d["low"].tail(20).min())
    vwap = np.nan
    if intraday is not None and len(intraday) > 0:
        dates = intraday.index.date
        today = intraday[dates == dates.max()]
        if len(today) > 0:
            v = today["volume"].sum()
            vwap = (today["close"] * today["volume"]).sum() / v if v else np.nan

    score = float(last["core"])
    if not np.isnan(vwap):
        score += 1.0 if price > vwap else -1.0
    n_score, n_count = news
    score += float(np.clip(n_score * 0.5, -1.5, 1.5))

    rvol = last["rvol"] if not pd.isna(last["rvol"]) else 0.0
    return {
        "symbol": sym, "price": round(price, 2), "score": round(score, 2),
        "rsi": round(float(last["rsi"]), 1), "atr": round(float(last["atr"]), 3),
        "atr_pct": round(float(last["atr"]) / price * 100, 2),
        "gap_pct": round(float(last["gap"]), 2), "rel_vol": round(float(rvol), 2),
        "mom5": round(float(last["mom5"]), 2), "mom20": round(float(last["mom20"]), 2),
        "vwap": round(float(vwap), 2) if not np.isnan(vwap) else None,
        "dist_high20": round((price / high20 - 1) * 100, 2),
        "dist_low20": round((price / low20 - 1) * 100, 2),
        "avg_dollar_vol_m": round(avg_dollar_vol / 1e6, 1),
        "news_score": n_score, "news_count": n_count,
    }


# ----------------------------------------------------------------------------
# Sizing and orders
# ----------------------------------------------------------------------------
def plan_trade(row, equity, buying_power):
    side = OrderSide.BUY if row["score"] > 0 else OrderSide.SELL
    if side == OrderSide.SELL and not ALLOW_SHORTS:
        return None
    price = row["price"]
    stop_dist = max(row["atr"] * ATR_STOP_MULT, price * 0.005)
    risk_dollars = equity * RISK_PER_TRADE
    qty_by_risk = int(risk_dollars // stop_dist)
    qty_by_cap = int((equity * MAX_POSITION_PCT) // price)
    qty_by_bp = int((buying_power * 0.9) // price)
    qty = min(qty_by_risk, qty_by_cap, qty_by_bp)
    if qty < 1 and SMALL_ACCOUNT_MIN_ONE_SHARE:
        # tiny account: allow one share if it fits the buying power and the loss is bearable
        if price <= buying_power * 0.9 and stop_dist <= equity * MAX_ONE_SHARE_RISK_PCT:
            qty = 1
    if qty < 1:
        return None
    if side == OrderSide.BUY:
        stop = round(price - stop_dist, 2)
        take = round(price + stop_dist * REWARD_RISK, 2)
    else:
        stop = round(price + stop_dist, 2)
        take = round(price - stop_dist * REWARD_RISK, 2)
    return {"symbol": row["symbol"], "side": side, "qty": qty, "entry": price,
            "stop": stop, "take": take, "risk_usd": round(qty * stop_dist, 2)}


def submit(trading, plan):
    req = MarketOrderRequest(
        symbol=plan["symbol"], qty=plan["qty"], side=plan["side"],
        time_in_force=TimeInForce.DAY, order_class=OrderClass.BRACKET,
        take_profit=TakeProfitRequest(limit_price=plan["take"]),
        stop_loss=StopLossRequest(stop_price=plan["stop"]),
    )
    return trading.submit_order(req)


# ----------------------------------------------------------------------------
# Logging
# ----------------------------------------------------------------------------
def log_rows(name, rows):
    if not rows:
        return
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"{name}_{datetime.now().strftime('%Y%m%d')}.csv"
    new = not path.exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        for r in rows:
            r = {k: (v.value if hasattr(v, "value") else v) for k, v in r.items()}
            w.writerow(r)


# ----------------------------------------------------------------------------
# Live scan
# ----------------------------------------------------------------------------
def load_stake_state(acct_equity):
    """Creates the stake state on the first run and returns it."""
    LOG_DIR.mkdir(exist_ok=True)
    if STATE_FILE.exists():
        return json.loads(STATE_FILE.read_text())
    state = {"baseline_equity": acct_equity, "stake_start": STAKE_START_USD,
             "created_date": datetime.now().strftime("%Y-%m-%d")}
    STATE_FILE.write_text(json.dumps(state, indent=2))
    return state


def run_scan(trading, data, news_client, screener, live):
    clock = trading.get_clock()
    acct = trading.get_account()
    acct_equity = float(acct.equity)
    last_equity = float(acct.last_equity)

    if STAKE_ENABLED:
        state = load_stake_state(acct_equity)
        baseline = float(state["baseline_equity"])
        stake = max(0.0, float(state["stake_start"]) + (acct_equity - baseline))
        if state["created_date"] == datetime.now().strftime("%Y-%m-%d"):
            day_start_stake = float(state["stake_start"])        # day 1
        else:
            day_start_stake = max(0.0, float(state["stake_start"]) + (last_equity - baseline))
        equity = stake                                           # sizing uses the stake, not the account
        day_pl = acct_equity - last_equity
        day_pl_pct = day_pl / day_start_stake * 100 if day_start_stake else 0.0
        label = f"stake ${stake:,.2f} (started ${float(state['stake_start']):,.2f})"
    else:
        equity = acct_equity
        day_pl_pct = (acct_equity / last_equity - 1) * 100 if last_equity else 0.0
        label = f"equity ${acct_equity:,.2f}"
    print(f"\n=== {datetime.now():%Y-%m-%d %H:%M:%S} | market open: {clock.is_open} | "
          f"{label} | day P/L {day_pl_pct:+.2f}% ===")
    if STAKE_ENABLED and stake < 1.0:
        print("Stake is used up (below $1). Bot is stopped. Delete bot_logs/stake_state.json to restart.")
        return

    mins_to_close = None
    if clock.is_open:
        mins_to_close = (clock.next_close - clock.timestamp).total_seconds() / 60

    if live and FLATTEN_ENABLED and mins_to_close is not None and mins_to_close <= FLATTEN_BEFORE_CLOSE_MIN:
        trading.close_all_positions(cancel_orders=True)
        print("Closing time: flattened all positions and cancelled open orders.")
        return

    if day_pl_pct <= -DAILY_MAX_LOSS_PCT:
        print("KILL SWITCH: daily loss limit hit. No new trades.")
        return
    if day_pl_pct >= DAILY_TARGET_PCT:
        print("Daily target reached. No new trades (edit DAILY_TARGET_PCT to change).")
        return
    if live and not clock.is_open:
        print("Market closed. Scanning only, no orders.")
        live = False
    if mins_to_close is not None and mins_to_close <= NO_NEW_TRADES_LAST_MIN:
        print("Too close to the close for new entries. Scanning only.")
        live = False

    if PDT_GUARD and acct_equity < 25_000:
        try:
            if int(acct.daytrade_count) >= 3:
                print("PDT guard: 3 day trades already used in the rolling 5 days. No new entries.")
                live = False
        except Exception:
            pass

    positions = {p.symbol: p for p in trading.get_all_positions()}
    open_orders = trading.get_orders(GetOrdersRequest(status=QueryOrderStatus.OPEN))
    busy = set(positions) | {o.symbol for o in open_orders}
    print(f"Open positions: {len(positions)} | open orders: {len(open_orders)}")

    symbols, notes = build_universe(screener)
    print(f"Universe: {len(symbols)} symbols (feed: {DATA_FEED})")

    daily = get_daily_bars(data, symbols)
    intraday = get_intraday_bars(data, symbols)
    snaps = get_snapshots(data, symbols)
    news = get_news_scores(news_client, symbols)

    # Regime filter from SPY
    regime_ok, bias = True, 0
    if REGIME_FILTER and "SPY" in daily.index.get_level_values("symbol"):
        reg = regime_from_spy(daily.xs("SPY", level="symbol"))
        r = reg.iloc[-1]
        regime_ok, bias = bool(r["reg_ok"]), int(r["bias"])
        print(f"Regime: SPY trend {r['trend_pct']:.2f}% (min {SPY_MIN_TREND_PCT}), "
              f"ATR {r['atr_pct']:.2f}% (max {SPY_MAX_ATR_PCT}), bias {'LONG' if bias > 0 else 'SHORT'} "
              f"-> {'TRADE' if regime_ok else 'SIT OUT'}")

    rows = []
    symbols_in_daily = set(daily.index.get_level_values("symbol"))
    symbols_in_intra = set(intraday.index.get_level_values("symbol")) if len(intraday) else set()
    for sym in symbols:
        try:
            if sym not in symbols_in_daily:
                continue
            d = daily.xs(sym, level="symbol")
            i = intraday.xs(sym, level="symbol") if sym in symbols_in_intra else None
            row = analyse_symbol(sym, d, i, snaps.get(sym), news.get(sym, (0, 0)))
            if row:
                row["source"] = notes.get(sym, "watchlist")
                row["ts"] = datetime.now().isoformat(timespec="seconds")
                rows.append(row)
        except Exception as e:
            print(f"[warn] {sym}: {e}")

    rows.sort(key=lambda r: abs(r["score"]), reverse=True)
    log_rows("scan", rows)

    print("\nTop candidates:")
    print(f"{'SYM':6} {'PRICE':>9} {'SCORE':>6} {'RSI':>5} {'GAP%':>6} {'RVOL':>5} {'MOM5':>6} {'MOM20':>6} {'NEWS':>5}")
    for r in rows[:15]:
        print(f"{r['symbol']:6} {r['price']:9.2f} {r['score']:6.2f} {r['rsi']:5.1f} {r['gap_pct']:6.2f} "
              f"{r['rel_vol']:5.2f} {r['mom5']:6.2f} {r['mom20']:6.2f} {r['news_score']:5}")

    if REGIME_FILTER and not regime_ok:
        print("\nSitting out: market regime filter says conditions are poor. A flat day beats a forced loss.")
        return

    slots = MAX_OPEN_POSITIONS - len(positions)
    if slots <= 0:
        print("\nMax open positions reached.")
        return

    buying_power = float(acct.buying_power)
    if STAKE_ENABLED:
        exposure = sum(abs(float(p.market_value)) for p in positions.values())
        buying_power = min(buying_power, max(0.0, stake - exposure))   # cash only, no leverage
        print(f"Stake in use: ${exposure:,.2f} | available for new trades: ${buying_power:,.2f}")
    plans, decisions = [], []
    for r in rows:
        if len(plans) >= min(slots, MAX_NEW_TRADES_PER_SCAN):
            break
        if abs(r["score"]) < MIN_SCORE or r["symbol"] in busy:
            continue
        if REGIME_FILTER and TREND_ALIGN and r["score"] * bias < 0:
            continue    # against the market trend
        p = plan_trade(r, equity, buying_power)
        if p:
            plans.append(p)
            buying_power -= p["qty"] * p["entry"]

    print(f"\nTrade plan ({len(plans)} trades):")
    for p in plans:
        line = (f"{p['side'].value.upper():4} {p['qty']:>5} {p['symbol']:6} @ ~{p['entry']:.2f} "
                f"stop {p['stop']:.2f} take {p['take']:.2f} risk ${p['risk_usd']:.2f}")
        status = "PLANNED"
        if live:
            try:
                o = submit(trading, p)
                status = f"SUBMITTED {o.id}"
            except Exception as e:
                status = f"FAILED {e}"
        print(f"  {line} -> {status}")
        decisions.append({**p, "status": status, "ts": datetime.now().isoformat(timespec="seconds")})
    log_rows("decisions", decisions)
    if not plans:
        print("  nothing clears the score, trend and sizing filters right now.")
    if not live:
        print("\nNo orders sent (dry run, or trading is paused by a guard above). Use --live to send.")


# ----------------------------------------------------------------------------
# Trend mode (ETF rotation)
# ----------------------------------------------------------------------------
def trend_daily_weights(close):
    """close: wide DataFrame of adjusted daily closes (index = dates, columns = ETFs).
    Returns the daily target weights before tranching. Uses only data up to each row."""
    univ = [s for s in TREND_RISKY if s in close.columns]
    c = close[univ].ffill()
    rets = c.pct_change(fill_method=None)
    score = sum(c / c.shift(L) - 1 for L in TREND_LOOKBACKS) / len(TREND_LOOKBACKS)
    valid = c.notna() & c.shift(max(TREND_LOOKBACKS)).notna()
    score = score.where(valid)
    sel = (score.rank(axis=1, ascending=False) <= TREND_TOP_N) & (score > 0)
    vol = rets.rolling(TREND_VOL_WINDOW).std()
    iv = (1 / vol).where(sel)
    w = iv.div(iv.sum(axis=1), axis=0).mul(sel.sum(axis=1) / TREND_TOP_N, axis=0).fillna(0.0)
    R, W = rets.fillna(0.0).values, w.values
    scale = np.ones(len(w))
    for i in range(TREND_VOL_WINDOW, len(w)):
        pv = (R[i - TREND_VOL_WINDOW + 1:i + 1] @ W[i]).std() * np.sqrt(252)
        if pv > 0:
            scale[i] = min(1.0, TREND_VOL_TARGET / pv)
    w = w.mul(scale, axis=0)
    w[TREND_FALLBACK] = w.get(TREND_FALLBACK, 0.0) + (1 - w.sum(axis=1))
    w[~valid.any(axis=1)] = 0.0
    return w


def trend_target(close):
    """Averages TREND_TRANCHES staggered sub-portfolios. Each tranche rebalances on the first
    trading day of each of its TREND_PERIOD-weekday buckets, counted from 2000-01-03, so the
    schedule is the same whatever window of history was downloaded and survives holidays."""
    wd = trend_daily_weights(close)
    idx = wd.index.tz_localize(None) if wd.index.tz is not None else wd.index
    pos = np.busday_count(np.datetime64("2000-01-03"), idx.values.astype("datetime64[D]"))
    step = TREND_PERIOD // TREND_TRANCHES
    acc = 0
    rebal_today = []
    for k in range(TREND_TRANCHES):
        bucket = pd.Series((pos - k * step) // TREND_PERIOD, index=wd.index)
        m = bucket != bucket.shift(1)
        m.iloc[0] = False                                   # first row has no prior bucket
        acc = acc + wd.where(m).ffill().fillna(0.0)
        rebal_today.append(bool(m.iloc[-1]))
    return acc / TREND_TRANCHES, any(rebal_today)


def get_trend_closes(data, clock):
    start = datetime.now(timezone.utc) - timedelta(days=560)
    req = StockBarsRequest(symbol_or_symbols=TREND_RISKY, timeframe=TimeFrame.Day, start=start,
                           adjustment="all", feed=feed())
    df = data.get_stock_bars(req).df.reset_index()
    df["date"] = pd.to_datetime(df["timestamp"]).dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    close = df.pivot_table(index="date", columns="symbol", values="close").sort_index()
    if clock.is_open:
        today = pd.Timestamp(clock.timestamp).tz_convert("America/New_York").normalize().tz_localize(None)
        close = close[close.index < today]      # signal uses yesterday's close; today's bar is partial
    return close


def run_trend(trading, data, live):
    clock = trading.get_clock()
    acct = trading.get_account()
    acct_equity = float(acct.equity)
    if STAKE_ENABLED:
        state = load_stake_state(acct_equity)
        capital = max(0.0, float(state["stake_start"]) + (acct_equity - float(state["baseline_equity"])))
        label = f"stake ${capital:,.2f} (started ${float(state['stake_start']):,.2f})"
    else:
        capital = acct_equity
        label = f"equity ${acct_equity:,.2f}"
    print(f"\n=== TREND MODE {datetime.now():%Y-%m-%d %H:%M:%S} | market open: {clock.is_open} | {label} ===")
    if capital < MIN_ORDER_USD:
        print("Capital below the $1 minimum order. Nothing to do.")
        return

    close = get_trend_closes(data, clock)
    missing = [s for s in TREND_RISKY if s not in close.columns]
    if missing:
        print(f"[warn] no data for {missing}; they are skipped this run.")
    if len(close) < max(TREND_LOOKBACKS) + TREND_VOL_WINDOW:
        sys.exit(f"Only {len(close)} days of history downloaded; need about 320. Check the data feed.")
    target, rebal_day = trend_target(close)
    w = target.iloc[-1]
    w = w[w > 0.0005].sort_values(ascending=False)
    print(f"Signal date: {close.index[-1].date()} | tranche rebalance on this signal: {'yes' if rebal_day else 'no'}")
    print("Target portfolio:")
    for s, x in w.items():
        print(f"  {s:5} {x * 100:6.2f}%   ${x * capital:8.2f}")

    positions = {p.symbol: p for p in trading.get_all_positions()}
    other = [s for s in positions if s not in TREND_RISKY]
    if other:
        print(f"[warn] positions outside the trend universe are left alone: {other}")
    current = {s: float(positions[s].market_value) for s in TREND_RISKY if s in positions}
    held_total = sum(current.values())

    orders = []
    for s in sorted(set(w.index) | set(current)):
        diff = float(w.get(s, 0.0)) * capital - current.get(s, 0.0)
        if abs(diff) < MIN_ORDER_USD:
            continue
        orders.append((s, round(diff, 2), float(w.get(s, 0.0)) == 0.0))
    sells = [o for o in orders if o[1] < 0]
    buys = [o for o in orders if o[1] > 0]
    cash_after_sells = capital - held_total + sum(-o[1] for o in sells)
    cash_after_sells = min(cash_after_sells, float(acct.cash) + sum(-o[1] for o in sells))
    total_buys = sum(o[1] for o in buys)
    if total_buys > cash_after_sells and total_buys > 0:
        f = max(0.0, cash_after_sells) / total_buys        # never buy more than the cash available
        buys = [(s, round(d * f, 2), z) for s, d, z in buys if d * f >= MIN_ORDER_USD]

    print(f"\nOrders ({len(sells)} sells, {len(buys)} buys), held now ${held_total:,.2f}:")
    if not sells and not buys:
        print("  none: the portfolio is already within $1 of target.")
    can_trade = live and clock.is_open
    if live and not clock.is_open:
        print("  market closed, so nothing is sent. Run again during market hours.")
    decisions = []
    for s, d, close_all in sells + buys:
        side = "SELL" if d < 0 else "BUY"
        desc = f"close position" if close_all and d < 0 else f"${abs(d):.2f}"
        status = "PLANNED"
        if can_trade:
            try:
                if close_all and d < 0:
                    o = trading.close_position(s)
                else:
                    o = trading.submit_order(MarketOrderRequest(
                        symbol=s, notional=abs(d), side=OrderSide.SELL if d < 0 else OrderSide.BUY,
                        time_in_force=TimeInForce.DAY))
                status = f"SUBMITTED {getattr(o, 'id', '')}"
            except Exception as e:
                status = f"FAILED {e}"
        print(f"  {side:4} {s:5} {desc:>15} -> {status}")
        decisions.append({"ts": datetime.now().isoformat(timespec="seconds"), "mode": "trend",
                          "symbol": s, "side": side, "usd": abs(d), "status": status})
        if can_trade and side == "SELL" and status.startswith("SUBMITTED"):
            time.sleep(1)                                   # let sells fill before buying
    log_rows("trend_decisions", decisions)
    if not live:
        print("\nDry run: no orders sent. Add --live to send them to the paper account.")


# ----------------------------------------------------------------------------
# Backtest
# ----------------------------------------------------------------------------
def build_candidates(daily_all, symbols, floor=0.0):
    """One row per (symbol, signal day): score at the close, outcome prices on the next day."""
    frames = []
    have = set(daily_all.index.get_level_values("symbol"))
    for sym in symbols:
        if sym not in have:
            continue
        d = add_indicators(daily_all.xs(sym, level="symbol"))
        nxt = d[["open", "high", "low", "close"]].shift(-1)
        idx = d.index.tz_localize(None) if d.index.tz is not None else d.index
        idx = idx.normalize()
        next_date = pd.Series(idx, index=d.index).shift(-1)
        out = pd.DataFrame({
            "sig_date": idx,
            "date": next_date.values,
            "symbol": sym,
            "score": d["core"].values,
            "atr": d["atr"].values,
            "px": (d["raw_close"] if "raw_close" in d.columns else d["close"]).values,
            "dollar_vol": ((d["raw_close"] if "raw_close" in d.columns else d["close"])
                           * d["volume"]).rolling(20).mean().values,
            "open": nxt["open"].values, "high": nxt["high"].values,
            "low": nxt["low"].values, "close": nxt["close"].values,
        })
        out = out.dropna()
        out = out[(out["score"].abs() >= floor) & (out["px"] >= MIN_PRICE)
                  & (out["dollar_vol"] >= min_dollar_volume())]
        frames.append(out)
    cand = pd.concat(frames, ignore_index=True)
    cand = cand[(cand["px"] >= MIN_PRICE) & (cand["dollar_vol"] >= min_dollar_volume())]
    return cand


def simulate(cand, regime, min_score):
    slip = BACKTEST_SLIPPAGE_BPS / 10_000
    c = cand[cand["score"].abs() >= min_score].copy()
    if REGIME_FILTER and regime is not None:
        c = c.join(regime[["reg_ok", "bias"]], on="sig_date")
        c = c[c["reg_ok"].fillna(False)]
        if TREND_ALIGN:
            c = c[(c["score"] * c["bias"]) > 0]
    if not ALLOW_SHORTS:
        c = c[c["score"] > 0]
    c["abs_score"] = c["score"].abs()
    c = c.sort_values(["date", "abs_score"], ascending=[True, False])
    c = c.groupby("date").head(MAX_NEW_TRADES_PER_SCAN)

    equity = BACKTEST_START_EQUITY
    day_rows, trades = [], []
    for date, g in c.groupby("date"):
        start_eq, day_pnl = equity, 0.0
        for r in g.itertuples():
            side = 1 if r.score > 0 else -1
            entry = r.open * (1 + side * slip)
            stop_dist = max(r.atr * ATR_STOP_MULT, entry * 0.005)
            stop = entry - side * stop_dist
            take = entry + side * stop_dist * REWARD_RISK
            if side == 1:
                hit_stop, hit_take = r.low <= stop, r.high >= take
            else:
                hit_stop, hit_take = r.high >= stop, r.low <= take
            if hit_stop:                  # if both are hit, assume the stop came first (conservative)
                exit_px = stop
            elif hit_take:
                exit_px = take
            else:
                exit_px = r.close
            exit_px = exit_px * (1 - side * slip)
            shares = min(start_eq * RISK_PER_TRADE / stop_dist, start_eq * MAX_POSITION_PCT / entry)
            pnl = side * (exit_px - entry) * shares
            day_pnl += pnl
            trades.append({"date": date, "symbol": r.symbol, "side": side, "score": r.score,
                           "pnl": pnl, "r_mult": pnl / (shares * stop_dist)})
        equity += day_pnl
        day_rows.append({"date": date, "pnl": day_pnl, "equity": equity,
                         "ret_pct": day_pnl / start_eq * 100})
    return pd.DataFrame(day_rows), pd.DataFrame(trades)


def summarize(days, trades, all_days):
    if len(trades) == 0:
        return {"trades": 0, "traded_days": 0, "green_pct": 0, "win_rate": 0, "avg_r": 0,
                "profit_factor": 0, "return_pct": 0, "max_dd_pct": 0, "worst_day": 0, "best_day": 0,
                "flat_days": all_days}
    gross_win = trades.loc[trades["pnl"] > 0, "pnl"].sum()
    gross_loss = -trades.loc[trades["pnl"] < 0, "pnl"].sum()
    eq = days["equity"]
    peak = eq.cummax()
    return {
        "trades": len(trades),
        "traded_days": len(days),
        "flat_days": all_days - len(days),
        "green_pct": round((days["pnl"] > 0).mean() * 100, 1),
        "win_rate": round((trades["pnl"] > 0).mean() * 100, 1),
        "avg_r": round(trades["r_mult"].mean(), 3),
        "profit_factor": round(gross_win / gross_loss, 2) if gross_loss > 0 else float("inf"),
        "return_pct": round((eq.iloc[-1] / BACKTEST_START_EQUITY - 1) * 100, 1),
        "max_dd_pct": round(((eq - peak) / peak).min() * 100, 1),
        "worst_day": round(days["ret_pct"].min(), 2),
        "best_day": round(days["ret_pct"].max(), 2),
    }


def load_archive(zip_path, years, min_median_dollar_vol=2_000_000):
    """Reads the Kaggle-style archive (stocks/*.csv, etfs/SPY.csv) straight from the zip.
    Prices are adjusted for splits and dividends using Adj Close. Raw close is kept for the
    price and liquidity filters. Returns (DataFrame indexed by symbol/timestamp, last date)."""
    global DATA_FEED
    DATA_FEED = "sip"           # archive volumes are full-market, so use the full dollar-volume filter
    zf = zipfile.ZipFile(zip_path)
    spy = pd.read_csv(zf.open("etfs/SPY.csv"), parse_dates=["Date"])
    data_end = spy["Date"].max()
    start = data_end - pd.Timedelta(days=int(years * 365) + 150)   # extra history for indicator warm-up
    names = [n for n in zf.namelist() if n.startswith("stocks/") and n.endswith(".csv")]
    names.append("etfs/SPY.csv")
    frames, skipped = {}, 0
    for n in names:
        sym = os.path.basename(n)[:-4]
        if not sym.isalpha():
            continue
        try:
            df = pd.read_csv(zf.open(n), parse_dates=["Date"])
            df = df[df["Date"] >= start].dropna()
            df = df[(df["Close"] > 0) & (df["Adj Close"] > 0)]
            if len(df) < 300:
                skipped += 1
                continue
            if (df["Close"] * df["Volume"]).median() < min_median_dollar_vol:
                skipped += 1
                continue
            f = df["Adj Close"] / df["Close"]
            out = pd.DataFrame({
                "open": (df["Open"] * f).values, "high": (df["High"] * f).values,
                "low": (df["Low"] * f).values, "close": df["Adj Close"].values,
                "volume": df["Volume"].values, "raw_close": df["Close"].values,
            }, index=pd.DatetimeIndex(df["Date"].values, tz="UTC", name="timestamp"))
            frames[sym] = out.sort_index()
        except Exception:
            skipped += 1
    daily_all = pd.concat(frames, names=["symbol", "timestamp"])
    print(f"Loaded {len(frames)} symbols from the archive ({skipped} skipped as illiquid or too short). "
          f"Data ends {data_end.date()}.")
    return daily_all, data_end


def run_backtest(data, years, archive=None):
    floor = min(SWEEP_SCORES + [MIN_SCORE])
    if archive:
        daily_all, data_end = load_archive(archive, years)
        symbols = sorted(set(daily_all.index.get_level_values("symbol")))
        cutoff = pd.Timestamp(data_end) - pd.Timedelta(days=int(years * 365))
    else:
        symbols = list(dict.fromkeys(WATCHLIST))
        start = datetime.now(timezone.utc) - timedelta(days=int(years * 365) + 120)
        print(f"Downloading {len(symbols)} symbols, about {years} years of daily bars (feed: {DATA_FEED})...")
        req = StockBarsRequest(symbol_or_symbols=symbols, timeframe=TimeFrame.Day, start=start,
                               adjustment="split", feed=feed())
        daily_all = data.get_stock_bars(req).df
        cutoff = pd.Timestamp.now().normalize() - pd.Timedelta(days=int(years * 365))
    if "SPY" not in daily_all.index.get_level_values("symbol"):
        sys.exit("SPY data missing, needed for the regime filter.")
    regime = regime_from_spy(daily_all.xs("SPY", level="symbol"))
    print("Computing indicators and signals...")
    cand = build_candidates(daily_all, symbols, floor=floor)
    cand = cand[cand["date"] >= cutoff]
    trading_days = list(regime.index[regime.index >= cutoff])      # every market day in the window
    split = trading_days[int(len(trading_days) * 0.6)]
    train, test = cand[cand["date"] < split], cand[cand["date"] >= split]
    n_train = sum(1 for x in trading_days if x < split)
    n_test = len(trading_days) - n_train
    print(f"{len(trading_days)} trading days | train {n_train} days | test {n_test} days "
          f"| slippage {BACKTEST_SLIPPAGE_BPS} bps per side\n")

    cols = ["trades", "traded_days", "flat_days", "green_pct", "win_rate", "avg_r",
            "profit_factor", "return_pct", "max_dd_pct", "worst_day", "best_day"]

    def table(label, c, n_days):
        print(f"--- {label} ---")
        print(f"{'MIN_SCORE':>9} " + " ".join(f"{k:>13}" for k in cols))
        results = {}
        for ms in SWEEP_SCORES:
            d, t = simulate(c, regime, ms)
            s = summarize(d, t, n_days)
            results[ms] = s
            print(f"{ms:9.1f} " + " ".join(f"{s[k]:>13}" for k in cols))
        print()
        return results

    res_train = table("TRAIN (first 60%, used to pick the setting)", train, n_train)
    table("TEST (last 40%, never used for picking)", test, n_test)

    ok = {k: v for k, v in res_train.items() if v["trades"] >= 30 and v["return_pct"] > 0}
    if ok:
        best = max(ok, key=lambda k: (ok[k]["green_pct"], ok[k]["avg_r"]))
        print(f"Best MIN_SCORE on TRAIN by green-day % (min 30 trades, positive return): {best}")
        print("Check that the TEST row for that value is also profitable. If not, the edge is probably noise.")
    else:
        print("No setting made money on the training period with enough trades. No evidence of an edge.")

    d, t = simulate(cand, regime, MIN_SCORE)
    LOG_DIR.mkdir(exist_ok=True)
    d.to_csv(LOG_DIR / "backtest_equity_curve.csv", index=False)
    t.to_csv(LOG_DIR / "backtest_trades.csv", index=False)
    print(f"\nSaved equity curve and trades for MIN_SCORE={MIN_SCORE} to {LOG_DIR}/")
    print("Reminder: watchlist is today's list (survivorship bias), entries assume the next open, "
          "and VWAP/news are not part of the backtest.")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["trend", "momentum"], default="trend",
                    help="trend = ETF rotation (default), momentum = old intraday scanner")
    ap.add_argument("--live", action="store_true", help="submit orders to the PAPER account")
    ap.add_argument("--loop", type=int, default=0, help="rescan every N seconds")
    ap.add_argument("--backtest", action="store_true", help="run the historical backtest and exit")
    ap.add_argument("--years", type=float, default=3.0, help="years of history for the backtest")
    ap.add_argument("--archive", type=str, default=None,
                    help="path to archive.zip (stocks/ and etfs/ daily CSVs) to backtest on instead of Alpaca data")
    args = ap.parse_args()

    if args.backtest and args.archive:
        run_backtest(None, args.years, archive=args.archive)
        return
    trading, data, news_client, screener = make_clients()
    if args.backtest:
        run_backtest(data, args.years)
        return
    if args.mode == "trend":
        run_trend(trading, data, args.live)
        return
    while True:
        try:
            run_scan(trading, data, news_client, screener, args.live)
        except Exception as e:
            print(f"[error] scan failed: {e}")
        if not args.loop:
            break
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
