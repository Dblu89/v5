"""
SMC OPTIMIZER v5 - PROFESSIONAL
WDO Dolar Mini B3 - 32 vCPUs
6 Estrategias SMC + Grid 10.000+ combos
"""

import pandas as pd
import numpy as np
import json, sys, os, time, warnings, itertools
from datetime import datetime
from multiprocessing import Pool, cpu_count
from smartmoneyconcepts import smc as SMC
warnings.filterwarnings("ignore")

# ================================================================

# CONFIGURACOES

# ================================================================

CSV_PATH   = "/workspace/strategy_composer/wdo_clean.csv"
OUTPUT_DIR = "/workspace/param_opt_output"
N_CORES    = min(32, cpu_count())
CAPITAL    = 50_000.0
MULT_WDO   = 10.0
COMISSAO   = 5.0
SLIPPAGE   = 2.0

# Grid de parametros - 10.000+ combos

GRID = {
    "swing_length":  [3, 5, 7, 10, 14],
    "rr_min":        [1.5, 2.0, 2.5, 3.0, 3.5],
    "atr_mult_sl":   [0.3, 0.5, 0.7, 1.0],
    "poi_janela":    [10, 20, 30, 50, 80],
    "choch_janela":  [10, 20, 30, 50, 80],
    "estrategia":    [1, 2, 3, 4, 5, 6],
    "close_break":   [True, False],
}

# 5x5x4x5x5x6x2 = 15.000 combos

MIN_TRADES = 20
MIN_PF     = 0.0
MAX_DD     = -99.0

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================================================================

# VARIAVEIS GLOBAIS (compartilhadas entre workers)

# ================================================================

_DF_GLOBAL = None
_IND_CACHE = {}

# ================================================================

# CARREGAMENTO

# ================================================================

def carregar():
    print(f"[DATA] Carregando {CSV_PATH}…")
    df = pd.read_csv(CSV_PATH, parse_dates=["datetime"], index_col="datetime")
    df.columns = [c.lower().strip() for c in df.columns]
    df = df[["open","high","low","close","volume"]].copy()
    df = df[df.index.dayofweek < 5]
    df = df[(df.index.hour >= 9) & (df.index.hour < 18)]
    df = df.dropna()
    df = df[df["close"] > 0]
    df = df.sort_index()
    df = df[~df.index.duplicated(keep="last")]
    print(f"[DATA] OK {len(df):,} candles | {df.index[0].date()} -> {df.index[-1].date()}")
    return df

# ================================================================

# PREPARAR INDICADORES SMC (lib oficial)

# ================================================================

def preparar_smc(df, swing_length=5, close_break=True):
    swings    = SMC.swing_highs_lows(df, swing_length=swing_length)
    estrutura = SMC.bos_choch(df, swings, close_break=close_break)
    fvg       = SMC.fvg(df)
    ob        = SMC.ob(df, swings)
    liq       = SMC.liquidity(df, swings)

    h, l, cp = df["high"], df["low"], df["close"].shift(1)
    tr  = pd.concat([h-l, (h-cp).abs(), (l-cp).abs()], axis=1).max(axis=1)
    atr = tr.rolling(14).mean()

    # ATR expandindo (volatilidade crescente)
    atr_expanding = atr > atr.rolling(20).mean()

    # Zonas premium/discount
    swing_high = df["high"].rolling(50).max()
    swing_low  = df["low"].rolling(50).min()
    rng        = swing_high - swing_low
    mid        = swing_low + rng * 0.5
    premium    = df["close"] > mid       # acima do meio = premium
    discount   = df["close"] < mid       # abaixo do meio = discount

    # Previous day high/low
    daily_high = df["high"].resample("D").max().reindex(df.index, method="ffill").shift(1)
    daily_low  = df["low"].resample("D").min().reindex(df.index, method="ffill").shift(1)

    # Sessao London 09h-12h BRT / NY 13h-17h BRT
    london = (df.index.hour >= 9) & (df.index.hour < 12)
    ny     = (df.index.hour >= 13) & (df.index.hour < 17)

    r = df.copy()
    r["choch"]         = estrutura["CHOCH"].values
    r["bos"]           = estrutura["BOS"].values
    r["fvg"]           = fvg["FVG"].values
    r["fvg_top"]       = fvg["Top"].values
    r["fvg_bot"]       = fvg["Bottom"].values
    r["ob"]            = ob["OB"].values
    r["ob_top"]        = ob["Top"].values
    r["ob_bot"]        = ob["Bottom"].values
    r["liq"]           = liq["Liquidity"].values if "Liquidity" in liq.columns else 0
    r["liq_lvl"]       = liq["Level"].values if "Level" in liq.columns else np.nan
    r["liq_swept"]     = liq["Swept"].values if "Swept" in liq.columns else np.nan
    r["atr"]           = atr.values
    r["atr_expanding"] = atr_expanding.values
    r["premium"]       = premium.values
    r["discount"]      = discount.values
    r["pdh"]           = daily_high.values
    r["pdl"]           = daily_low.values
    r["london"]        = london
    r["ny"]            = ny

    # fillna nos sinais
    for col in ["choch","bos","fvg","ob","liq"]:
        r[col] = pd.to_numeric(r[col], errors="coerce").fillna(0)

    return r

# ================================================================

# 6 ESTRATEGIAS SMC

# ================================================================

def verificar_entrada(i, row, cols, close, atr,
    fvgs_bull, fvgs_bear,
    obs_bull, obs_bear,
    liq_bull, liq_bear,
    ult_choch_bull, ult_choch_bear,
    choch_janela, estrategia):
    """
    Retorna (sinal, poi, tipo_entrada) ou (None, None, None)

    Estrategias:
    1 - CHoCH + FVG/OB (base)
    2 - CHoCH + FVG/OB somente em zona Discount/Premium
    3 - CHoCH + FVG/OB somente na sessao London/NY
    4 - CHoCH + confluencia OB+FVG (ambos na mesma zona)
    5 - Liquidity Sweep + CHoCH + FVG/OB
    6 - CHoCH + FVG/OB com filtro ATR expandindo
    """

    def v(col):
        return row[cols[col]]

    premium  = bool(v("premium"))
    discount = bool(v("discount"))
    london   = bool(v("london"))
    ny       = bool(v("ny"))
    atr_exp  = bool(v("atr_expanding"))

    sinal = poi = tipo = None

    # Filtros por estrategia
    if estrategia == 2:
        # Discount para long, Premium para short
        pode_bull = discount
        pode_bear = premium
    elif estrategia == 3:
        # Somente London ou NY
        pode_bull = pode_bear = (london or ny)
    elif estrategia == 6:
        # ATR expandindo (momentum)
        pode_bull = pode_bear = atr_exp
    else:
        pode_bull = pode_bear = True

    # Estrategia 4: confluencia OB + FVG
    if estrategia == 4:
        if (i - ult_choch_bull) <= choch_janela and pode_bull:
            for ob in obs_bull:
                for fv in fvgs_bull:
                    # Sobreposicao entre OB e FVG
                    overlap_top = min(ob["top"], fv["top"])
                    overlap_bot = max(ob["bot"], fv["bot"])
                    if overlap_top > overlap_bot and overlap_bot <= close <= overlap_top:
                        sinal = 1
                        poi = {"top": overlap_top, "bot": overlap_bot, "tipo": "OB+FVG"}
                        break
                if sinal:
                    break

        if sinal is None and (i - ult_choch_bear) <= choch_janela and pode_bear:
            for ob in obs_bear:
                for fv in fvgs_bear:
                    overlap_top = min(ob["top"], fv["top"])
                    overlap_bot = max(ob["bot"], fv["bot"])
                    if overlap_top > overlap_bot and overlap_bot <= close <= overlap_top:
                        sinal = -1
                        poi = {"top": overlap_top, "bot": overlap_bot, "tipo": "OB+FVG"}
                        break
                if sinal:
                    break
        return sinal, poi

    # Estrategia 5: Liquidity Sweep + CHoCH + FVG/OB
    if estrategia == 5:
        liq_swept = v("liq_swept")
        liq_dir   = v("liq")
        # Sweep de liquidez bearish (varreu shorts) -> espera reversao bull
        if not np.isnan(liq_swept) and liq_dir == -1:
            pode_bull = True
        # Sweep de liquidez bullish (varreu longs) -> espera reversao bear
        if not np.isnan(liq_swept) and liq_dir == 1:
            pode_bear = True

    # Logica base: CHoCH + FVG/OB
    if (i - ult_choch_bull) <= choch_janela and pode_bull:
        for p in reversed(fvgs_bull + obs_bull):
            if close <= p["top"] and close >= p["bot"] * 0.998:
                sinal = 1
                poi = p
                break

    if sinal is None and (i - ult_choch_bear) <= choch_janela and pode_bear:
        for p in reversed(fvgs_bear + obs_bear):
            if close >= p["bot"] and close <= p["top"] * 1.002:
                sinal = -1
                poi = p
                break

    return sinal, poi

# ================================================================

# ENGINE DE BACKTEST

# ================================================================

def backtest(df_ind, rr_min=2.0, atr_mult_sl=0.5,
    poi_janela=20, choch_janela=20,
    estrategia=1, capital=CAPITAL):

    trades = []
    equity = [capital]
    cap    = capital
    em_pos = False
    trade  = None

    fvgs_bull, fvgs_bear = [], []
    obs_bull,  obs_bear  = [], []
    liq_bull,  liq_bear  = [], []
    ult_choch_bull = ult_choch_bear = -9999

    arr  = df_ind.values
    cols = {c: i for i, c in enumerate(df_ind.columns)}

    def v(row, col):
        return row[cols[col]]

    for i in range(20, len(df_ind)):
        row   = arr[i]
        close = v(row, "close")
        atr   = v(row, "atr")
        if np.isnan(atr) or atr <= 0:
            atr = 5.0

        # Gerenciar posicao aberta
        if em_pos and trade:
            d, sl, tp, en = trade["d"], trade["sl"], trade["tp"], trade["entry"]
            lo, hi = v(row, "low"), v(row, "high")
            hit_sl = (d == 1 and lo <= sl) or (d == -1 and hi >= sl)
            hit_tp = (d == 1 and hi >= tp) or (d == -1 and lo <= tp)
            if hit_sl or hit_tp:
                saida = sl if hit_sl else tp
                pts   = (saida - en) * d
                brl   = pts * MULT_WDO - COMISSAO - SLIPPAGE * MULT_WDO * 0.5
                cap  += brl
                equity.append(round(cap, 2))
                trade["saida"]     = round(saida, 2)
                trade["pnl_pts"]   = round(pts, 2)
                trade["pnl_brl"]   = round(brl, 2)
                trade["resultado"] = "WIN" if hit_tp else "LOSS"
                trade["saida_dt"]  = str(df_ind.index[i])[:16]
                trades.append(trade)
                em_pos = False
                trade  = None
            continue

        # Coletar CHoCH
        choch = v(row, "choch")
        if choch == 1:
            ult_choch_bull = i
            fvgs_bull.clear(); obs_bull.clear()
        elif choch == -1:
            ult_choch_bear = i
            fvgs_bear.clear(); obs_bear.clear()

        # Coletar POIs
        fvg_v = v(row, "fvg")
        if fvg_v == 1 and not np.isnan(v(row, "fvg_top")):
            fvgs_bull.append({"top": v(row,"fvg_top"), "bot": v(row,"fvg_bot"), "tipo": "FVG"})
        elif fvg_v == -1 and not np.isnan(v(row, "fvg_top")):
            fvgs_bear.append({"top": v(row,"fvg_top"), "bot": v(row,"fvg_bot"), "tipo": "FVG"})

        ob_v = v(row, "ob")
        if ob_v == 1 and not np.isnan(v(row, "ob_top")):
            obs_bull.append({"top": v(row,"ob_top"), "bot": v(row,"ob_bot"), "tipo": "OB"})
        elif ob_v == -1 and not np.isnan(v(row, "ob_top")):
            obs_bear.append({"top": v(row,"ob_top"), "bot": v(row,"ob_bot"), "tipo": "OB"})

        # Expirar POIs antigos
        fvgs_bull = fvgs_bull[-poi_janela:]
        fvgs_bear = fvgs_bear[-poi_janela:]
        obs_bull  = obs_bull[-poi_janela:]
        obs_bear  = obs_bear[-poi_janela:]

        # Verificar entrada
        sinal, poi = verificar_entrada(
            i, row, cols, close, atr,
            fvgs_bull, fvgs_bear, obs_bull, obs_bear,
            liq_bull, liq_bear,
            ult_choch_bull, ult_choch_bear,
            choch_janela, estrategia
        )

        if sinal is None or poi is None:
            continue

        # Calcular SL e TP
        slip = SLIPPAGE * 0.5
        if sinal == 1:
            entry = close + slip
            sl    = poi["bot"] - atr * atr_mult_sl
        else:
            entry = close - slip
            sl    = poi["top"] + atr * atr_mult_sl

        risk = abs(entry - sl)
        if risk <= 0:
            continue

        tp   = entry + sinal * risk * rr_min
        rr_r = abs(tp - entry) / risk

        if rr_r < rr_min * 0.95:
            continue
        if risk * MULT_WDO / cap > 0.05:
            continue

        em_pos = True
        trade  = {
            "entry_dt":   str(df_ind.index[i])[:16],
            "d":          sinal,
            "entry":      round(entry, 2),
            "sl":         round(sl, 2),
            "tp":         round(tp, 2),
            "rr":         round(rr_r, 2),
            "poi_tipo":   poi.get("tipo", "?"),
            "estrategia": estrategia,
            "capital_pre":round(cap, 2),
        }

    if em_pos and trade:
        last = float(arr[-1][cols["close"]])
        pts  = (last - trade["entry"]) * trade["d"]
        brl  = pts * MULT_WDO - COMISSAO
        cap += brl
        trade.update({"saida": last, "pnl_pts": round(pts,2),
                      "pnl_brl": round(brl,2), "resultado": "ABERTO",
                      "saida_dt": str(df_ind.index[-1])[:16]})
        trades.append(trade)
        equity.append(round(cap, 2))

    return trades, equity

# ================================================================

# METRICAS

# ================================================================

def metricas(trades, equity, capital=CAPITAL):
    fechados = [t for t in trades if t.get("resultado") in ("WIN","LOSS")]
    if len(fechados) < MIN_TRADES:
        return None
    df_t  = pd.DataFrame(fechados)
    wins  = df_t[df_t["resultado"] == "WIN"]
    loses = df_t[df_t["resultado"] == "LOSS"]
    n     = len(df_t)
    wr    = len(wins) / n * 100
    avg_w = float(wins["pnl_brl"].mean())  if len(wins)  else 0
    avg_l = float(loses["pnl_brl"].mean()) if len(loses) else -1
    pf    = abs(float(wins["pnl_brl"].sum()) / float(loses["pnl_brl"].sum())) if float(loses["pnl_brl"].sum()) != 0 else 9999
    pnl   = float(df_t["pnl_brl"].sum())

    eq   = pd.Series(equity)
    peak = eq.cummax()
    dd   = (eq - peak) / peak * 100
    mdd  = float(dd.min())

    if mdd < MAX_DD:
        return None

    rets    = eq.pct_change().dropna()
    sharpe  = float(rets.mean() / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
    neg     = rets[rets < 0]
    sortino = float(rets.mean() / neg.std() * np.sqrt(252)) if len(neg) > 1 else 0

    poi_tipos = df_t["poi_tipo"].value_counts().to_dict()

    return {
        "total_trades": n,
        "wins":         int(len(wins)),
        "losses":       int(len(loses)),
        "win_rate":     round(wr, 2),
        "profit_factor":round(pf, 3),
        "sharpe_ratio": round(sharpe, 3),
        "sortino_ratio":round(sortino, 3),
        "avg_win_brl":  round(avg_w, 2),
        "avg_loss_brl": round(avg_l, 2),
        "expectancy_brl":round((wr/100*avg_w)+((1-wr/100)*avg_l), 2),
        "total_pnl_brl":round(pnl, 2),
        "retorno_pct":  round(pnl/capital*100, 2),
        "max_drawdown_pct": round(mdd, 2),
        "capital_final":round(capital+pnl, 2),
        "poi_tipos":    poi_tipos,
    }

# ================================================================

# WORKER (multiprocessing - sem serializar df)

# ================================================================

def init_worker(df_ins):
    global _DF_GLOBAL
    _DF_GLOBAL = df_ins

def worker(params):
    global _DF_GLOBAL
    sw, rr, am, pj, cj, est, cb = params
    try:
        df_ind = preparar_smc(_DF_GLOBAL, swing_length=sw, close_break=cb)
        trades, equity = backtest(df_ind, rr_min=rr, atr_mult_sl=am,
            poi_janela=pj, choch_janela=cj,
            estrategia=est)
        m = metricas(trades, equity)
        if not m:
            return None
        if m["profit_factor"] < MIN_PF:
            return None

        pf_s = min(m["profit_factor"], 10) / 10
        sh_s = min(max(m["sharpe_ratio"], 0), 8) / 8
        so_s = min(max(m["sortino_ratio"], 0), 10) / 10
        wr_s = m["win_rate"] / 100
        tr_s = min(m["total_trades"], 500) / 500

        score = pf_s*0.35 + sh_s*0.25 + so_s*0.15 + wr_s*0.15 + tr_s*0.10

        return {
            "swing_length": sw, "rr_min": rr, "atr_mult_sl": am,
            "poi_janela": pj, "choch_janela": cj,
            "estrategia": est, "close_break": cb,
            "score": round(float(score), 6),
            **m
        }
    except Exception as e:
        return None

# ================================================================

# GRID SEARCH

# ================================================================

def grid_search(df_ins, mini=False):
    g = GRID
    combos = list(itertools.product(
        g["swing_length"], g["rr_min"], g["atr_mult_sl"],
        g["poi_janela"], g["choch_janela"],
        g["estrategia"], g["close_break"]
    ))

    if mini:
        combos = [(5, 2.0, 0.5, 20, 20, 1, True),
                  (5, 2.0, 0.5, 20, 20, 2, True),
                  (5, 2.0, 0.5, 20, 20, 3, True)]
        print(f"\n[GRID] Modo MINI - {len(combos)} combos (1 por estrategia base)")
    else:
        print(f"\n[GRID] Grid Search v5 - {len(combos):,} combos | {N_CORES} cores")
        print(f"       6 estrategias SMC | parametros expandidos")

    t0 = time.time()

    if mini:
        resultados = [r for r in [worker(c) for c in combos] if r and "score" in r]
    else:
        print(f"[GRID] Iniciando Pool com initializer (df compartilhado)...")
        with Pool(processes=N_CORES,
                  initializer=init_worker,
                  initargs=(df_ins,)) as pool:
            raw = pool.map(worker, combos, chunksize=max(1, len(combos)//N_CORES//4))
        resultados = [r for r in raw if r and "score" in r]

    elapsed = time.time() - t0
    resultados.sort(key=lambda x: -x["score"])

    print(f"\n[GRID] OK {elapsed:.1f}s | {len(resultados):,} validos de {len(combos):,}")

    if resultados:
        exibir_top(resultados)

    return {
        "melhor":       resultados[0] if resultados else None,
        "top20":        resultados[:20],
        "por_estrategia": agrupar_por_estrategia(resultados),
        "total_combos": len(combos),
        "validos":      len(resultados),
        "elapsed_s":    round(elapsed, 1),
    }

def agrupar_por_estrategia(resultados):
    nomes = {
        1: "CHoCH+FVG/OB Base",
        2: "CHoCH+FVG/OB Discount/Premium",
        3: "CHoCH+FVG/OB London/NY",
        4: "CHoCH+OB+FVG Confluencia",
        5: "Liquidity Sweep+CHoCH",
        6: "CHoCH+FVG/OB ATR Filter",
    }
    grupos = {}
    for est in range(1, 7):
        subset = [r for r in resultados if r["estrategia"] == est]
        if subset:
            melhor = subset[0]
            grupos[nomes[est]] = {
                "melhor_score": melhor["score"],
                "melhor_pf":    melhor["profit_factor"],
                "melhor_wr":    melhor["win_rate"],
                "melhor_trades":melhor["total_trades"],
                "config":       {k: melhor[k] for k in
                    ["swing_length","rr_min","atr_mult_sl",
                     "poi_janela","choch_janela","close_break"]},
                "total_validos":len(subset),
            }
    return grupos

def exibir_top(resultados, n=20):
    nomes_est = {1:"Base", 2:"Disc/Prem", 3:"London/NY",
        4:"OB+FVG", 5:"Liq.Sweep", 6:"ATR Filt"}
    print(f"\n{'='*82}")
    print(f"  TOP {min(n,len(resultados))} CONFIGURACOES - SMC OPTIMIZER v5")
    print(f"{'='*82}")
    print(f"  {'#':>2} {'EST':>10} {'SW':>3} {'RR':>4} {'ATR':>4} {'POI':>4} {'CHoCH':>5} "
        f"{'PF':>6} {'Sharpe':>7} {'WR%':>6} {'Trades':>7} {'DD%':>6} {'Score':>7}")
    print(f"  {'-'*80}")
    for i, r in enumerate(resultados[:n], 1):
        star = "*" if i == 1 else " "
        nome = nomes_est.get(r["estrategia"], "?")
        print(f"  {star}{i:>2} {nome:>10} {r['swing_length']:>3} {r['rr_min']:>4} "
            f"{r['atr_mult_sl']:>4} {r['poi_janela']:>4} {r['choch_janela']:>5} "
            f"{r['profit_factor']:>6.3f} {r['sharpe_ratio']:>7.3f} "
            f"{r['win_rate']:>6.1f} {r['total_trades']:>7} "
            f"{r['max_drawdown_pct']:>6.1f} {r['score']:>7.4f}")
    print(f"{'='*82}")

    # Resumo por estrategia
    print(f"\n  MELHOR POR ESTRATEGIA:")
    for est, nome in {1:"Base",2:"Disc/Prem",3:"London/NY",
                      4:"OB+FVG",5:"Liq.Sweep",6:"ATR Filt"}.items():
        subset = [r for r in resultados if r["estrategia"] == est]
        if subset:
            b = subset[0]
            print(f"  {nome:>12}: PF={b['profit_factor']:.3f} | WR={b['win_rate']:.1f}% | "
                  f"Trades={b['total_trades']} | Score={b['score']:.4f}")

# ================================================================

# WALK-FORWARD

# ================================================================

def walk_forward(df, config, n_splits=5):
    print(f"\n[WF] Walk-Forward: {n_splits} splits")
    resultados = []
    step = len(df) // n_splits

    for i in range(n_splits - 1):
        ini   = i * step
        fim   = (i + 2) * step
        split = ini + int((fim - ini) * 0.7)
        df_tr = df.iloc[ini:split]
        df_te = df.iloc[split:fim]
        if len(df_tr) < 500 or len(df_te) < 100:
            continue

        d0 = df_tr.index[0].strftime("%Y-%m-%d")
        d1 = df_tr.index[-1].strftime("%Y-%m-%d")
        d2 = df_te.index[0].strftime("%Y-%m-%d")
        d3 = df_te.index[-1].strftime("%Y-%m-%d")

        try:
            sw  = config["swing_length"]
            cb  = config["close_break"]
            est = config["estrategia"]
            bt  = {k: v for k, v in config.items()
                   if k in ["rr_min","atr_mult_sl","poi_janela","choch_janela","estrategia"]}
            df_tr_ind = preparar_smc(df_tr, swing_length=sw, close_break=cb)
            df_te_ind = preparar_smc(df_te, swing_length=sw, close_break=cb)
            tr_t, tr_e = backtest(df_tr_ind, **bt)
            te_t, te_e = backtest(df_te_ind, **bt)
            m_tr = metricas(tr_t, tr_e) or {}
            m_te = metricas(te_t, te_e) or {}

            print(f"\n  Split {i+1}: Train [{d0}->{d1}] | Test [{d2}->{d3}]")
            if m_tr:
                print(f"    TRAIN -> WR:{m_tr.get('win_rate',0)}% | "
                      f"PF:{m_tr.get('profit_factor',0)} | "
                      f"Trades:{m_tr.get('total_trades',0)} | "
                      f"PnL:R${m_tr.get('total_pnl_brl',0):,.0f}")
            if m_te:
                print(f"    TEST  -> WR:{m_te.get('win_rate',0)}% | "
                      f"PF:{m_te.get('profit_factor',0)} | "
                      f"Trades:{m_te.get('total_trades',0)} | "
                      f"PnL:R${m_te.get('total_pnl_brl',0):,.0f}")

            resultados.append({"split": i+1, "train": m_tr, "test": m_te,
                                "train_start": d0, "train_end": d1,
                                "test_start": d2, "test_end": d3})
        except Exception as e:
            print(f"    Split {i+1} erro: {e}")

    lucrativos = sum(1 for r in resultados if r["test"].get("total_pnl_brl",0) > 0)
    print(f"\n[WF] OK {lucrativos}/{len(resultados)} splits out-of-sample lucrativos")
    return resultados

# ================================================================

# MONTE CARLO

# ================================================================

def monte_carlo(trades, n_sim=2000, capital=CAPITAL):
    fechados = [t for t in trades if t.get("resultado") in ("WIN","LOSS")]
    if len(fechados) < 10:
        return {}
    print(f"\n[MC] Monte Carlo: {n_sim:,} simulacoes…")
    pnls = np.array([t["pnl_brl"] for t in fechados])
    np.random.seed(42)
    rets, dds, ruinas = [], [], 0
    for _ in range(n_sim):
        seq = np.random.choice(pnls, size=len(pnls), replace=True)
        eq  = np.insert(capital + np.cumsum(seq), 0, capital)
        pk  = np.maximum.accumulate(eq)
        dd  = ((eq - pk) / pk * 100).min()
        r   = (eq[-1] - capital) / capital * 100
        rets.append(r); dds.append(dd)
        if eq[-1] < capital * 0.5:
            ruinas += 1
    rf, md = np.array(rets), np.array(dds)
    res = {
        "n_simulacoes":    n_sim,
        "prob_lucro_pct":  round(float((rf > 0).mean() * 100), 1),
        "retorno_mediana": round(float(np.median(rf)), 2),
        "retorno_p10":     round(float(np.percentile(rf, 10)), 2),
        "retorno_p25":     round(float(np.percentile(rf, 25)), 2),
        "retorno_p75":     round(float(np.percentile(rf, 75)), 2),
        "retorno_p90":     round(float(np.percentile(rf, 90)), 2),
        "dd_mediano":      round(float(np.median(md)), 2),
        "dd_p90":          round(float(np.percentile(md, 90)), 2),
        "dd_pior":         round(float(md.min()), 2),
        "prob_ruina_pct":  round(float(ruinas/n_sim*100), 2),
        "prob_dd_10":      round(float((md < -10).mean() * 100), 1),
        "prob_dd_20":      round(float((md < -20).mean() * 100), 1),
    }
    print(f"[MC] OK Prob.lucro:{res['prob_lucro_pct']}% | "
        f"DD mediano:{res['dd_mediano']}% | Ruina:{res['prob_ruina_pct']}%")
    return res

# ================================================================

# RELATORIO

# ================================================================

def relatorio(m, mc=None, titulo="RESULTADO", config=None):
    if not m:
        return
    sep = "=" * 62
    def L(lb, vl):
        print(f"  {lb:<32} {str(vl):>26}")
    print(f"\n{sep}")
    print(f"  SMC OPTIMIZER v5 – {titulo}")
    print(sep)
    if config:
        est_nomes = {1:"Base",2:"Disc/Prem",3:"London/NY",
            4:"OB+FVG",5:"Liq.Sweep",6:"ATR Filt"}
        print(f"  Estrategia : {est_nomes.get(config.get('estrategia',1),'?')}")
        print(f"  Params     : SW={config.get('swing_length')} "
            f"RR={config.get('rr_min')} ATR={config.get('atr_mult_sl')} "
            f"POI={config.get('poi_janela')} CHoCH={config.get('choch_janela')}")
        print(f"  {'-'*58}")
    L("Total Trades",     m["total_trades"])
    L("Wins / Losses",    f"{m['wins']} W  /  {m['losses']} L")
    L("Win Rate",         f"{m['win_rate']}%")
    L("Profit Factor",    m["profit_factor"])
    L("Sharpe Ratio",     m["sharpe_ratio"])
    L("Sortino Ratio",    m["sortino_ratio"])
    L("Expectancy",       f"R$ {m['expectancy_brl']:,.2f}")
    L("Total PnL",        f"R$ {m['total_pnl_brl']:,.2f}")
    L("Retorno %",        f"{m['retorno_pct']}%")
    L("Max Drawdown",     f"{m['max_drawdown_pct']}%")
    L("Capital Final",    f"R$ {m['capital_final']:,.2f}")
    if mc:
        print(f"  {'-'*58}")
        print(f"  MONTE CARLO ({mc['n_simulacoes']:,} simulacoes)")
        L("Prob. Lucro",      f"{mc['prob_lucro_pct']}%")
        L("Retorno Mediana",  f"{mc['retorno_mediana']}%")
        L("Retorno P10/P90",  f"{mc['retorno_p10']}% / {mc['retorno_p90']}%")
        L("DD Mediano",       f"{mc['dd_mediano']}%")
        L("DD Pior",          f"{mc['dd_pior']}%")
        L("Risco Ruina",      f"{mc['prob_ruina_pct']}%")
        L("Prob DD > 10%",    f"{mc['prob_dd_10']}%")
        L("Prob DD > 20%",    f"{mc['prob_dd_20']}%")
    print(sep)

# ================================================================

# EXPORTAR JSON

# ================================================================

def exportar(grid, m_full, m_oos, wf, mc, config, trades, equity):
    out = {
        "versao":        "v5",
        "gerado_em":     datetime.now().isoformat(),
        "config_melhor": config,
        "metricas_full": m_full,
        "metricas_oos":  m_oos,
        "walk_forward":  wf,
        "monte_carlo":   mc,
        "grid_top20":    grid.get("top20", []),
        "grid_por_estrategia": grid.get("por_estrategia", {}),
        "grid_stats": {
            "total_combos": grid.get("total_combos"),
            "validos":      grid.get("validos"),
            "elapsed_s":    grid.get("elapsed_s"),
        },
        "trades":        trades,
        "equity_curve":  equity,
    }
    path = f"{OUTPUT_DIR}/resultado_v5.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\n[OK] Resultado salvo em {path}")
    return path

# ================================================================

# MAIN

# ================================================================

def main():
    MINI = "–mini" in sys.argv

    print("=" * 70)
    print("  SMC OPTIMIZER v5 -- 6 ESTRATEGIAS SMC | 15.000 COMBOS")
    print("  WDO Dolar Mini B3 | Biblioteca oficial smartmoneyconcepts")
    print("=" * 70)

    # 1. Carregar dados
    df = carregar()
    split  = int(len(df) * 0.70)
    df_ins = df.iloc[:split]
    df_oos = df.iloc[split:]
    print(f"  In-sample : {len(df_ins):,} | {df_ins.index[0].date()} -> {df_ins.index[-1].date()}")
    print(f"  Out-sample: {len(df_oos):,} | {df_oos.index[0].date()} -> {df_oos.index[-1].date()}")

    # Inicializar worker com df (compartilhado)
    init_worker(df_ins)

    if MINI:
        print("\n[MINI] Testando 3 combos (1 por estrategia base)...")
        grid = grid_search(df_ins, mini=True)
        if grid["validos"] > 0:
            print(f"\nOK {grid['validos']} estrategia(s) valida(s)!")
            for k, v in grid.get("por_estrategia", {}).items():
                if v:
                    print(f"  {k}: PF={v['melhor_pf']:.3f} | "
                          f"WR={v['melhor_wr']:.1f}% | Trades={v['melhor_trades']}")
        else:
            print("AVISO: Nenhum resultado valido no mini.")
        return

    # 2. Grid search
    grid = grid_search(df_ins, mini=False)

    if not grid["melhor"]:
        print("\n[ERRO] Nenhuma configuracao valida.")
        return

    melhor = grid["melhor"]
    CONFIG = {k: melhor[k] for k in
              ["swing_length","rr_min","atr_mult_sl",
               "poi_janela","choch_janela","estrategia","close_break"]}

    # 3. Backtest Out-of-Sample
    print("\n[OOS] Backtest Out-of-Sample...")
    df_oos_ind = preparar_smc(df_oos, swing_length=CONFIG["swing_length"],
                               close_break=CONFIG["close_break"])
    bt_params  = {k: v for k, v in CONFIG.items()
                  if k in ["rr_min","atr_mult_sl","poi_janela","choch_janela","estrategia"]}
    t_oos, e_oos = backtest(df_oos_ind, **bt_params)
    m_oos = metricas(t_oos, e_oos)
    relatorio(m_oos, titulo="OUT-OF-SAMPLE", config=CONFIG)

    # 4. Backtest completo
    print("\n[FULL] Backtest dataset completo...")
    df_full_ind = preparar_smc(df, swing_length=CONFIG["swing_length"],
                                close_break=CONFIG["close_break"])
    t_full, e_full = backtest(df_full_ind, **bt_params)
    m_full = metricas(t_full, e_full)

    # 5. Walk-Forward
    wf = walk_forward(df, CONFIG, n_splits=5)

    # 6. Monte Carlo
    mc = monte_carlo(t_full, n_sim=2000)
    relatorio(m_full, mc, titulo="COMPLETO + MONTE CARLO", config=CONFIG)

    # 7. Exportar
    exportar(grid, m_full, m_oos, wf, mc, CONFIG, t_full, e_full)

    print(f"\nCONCLUIDO!")
    print(f"  Melhor estrategia : {melhor['estrategia']}")
    print(f"  Profit Factor     : {melhor['profit_factor']}")
    print(f"  Win Rate          : {melhor['win_rate']}%")
    print(f"  Score             : {melhor['score']}")
    print(f"  Trades            : {melhor['total_trades']}")

if __name__ == "__main__":
    main()
