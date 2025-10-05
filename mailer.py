# -*- coding: utf-8 -*-
"""
Tahmin Botu — tek parça mailer.py
(GÜNCEL: Elo + OppAdj Form + Hava + Odds + API-Football ipucu + High-Alert ayrı mail
 + Otomatik Öğrenme (w_mkt & goal_scale) + TableAdj (standings) + Streak (W/L))

Ücretsiz kaynaklar:
- football-data.org (Fixtures/sonuçlar/standings) -> X-Auth-Token: FOOTBALL_DATA_TOKEN
- OpenLigaDB (fallback)                            -> anahtar gerekmez
- Open-Meteo (hava)                                -> anahtar gerekmez
- The Odds API (opsiyonel oranlar)                 -> ODDS_API_KEY varsa kullanılır
- API-Football (opsiyonel, free tier)              -> APIFOOTBALL_KEY varsa kart/korner ipucu

Modlar:
- MODE=PREDICT  -> 10:00 TR “Günün Tahminleri”
- MODE=RESULTS  -> 23:59 TR “Günün Sonuçları”
- MODE=AUTO     -> Saat 07 UTC ise PREDICT, aksi ise RESULTS

Zorunlu Secrets: GMAIL_USER, GMAIL_PASS, GMAIL_TO
Önerilen Secrets: FOOTBALL_DATA_TOKEN
Opsiyonel Secrets: ODDS_API_KEY, APIFOOTBALL_KEY

Opsiyonel ENV (varsayılanlar):
- TOP_N=5, MIN_CONF=0, HIGH_ALERT=90
- OLD_LEAGUES="bundesliga,bundesliga2"
- ODDS_TTL_MIN=15
- ELO_K=24, ELO_HOME_ADV=60
- FORM_LOOKBACK=10, FORM_DAYS=120
- ALLOW_STATE_FILE=1
- SPLIT_HIGH_ALERT_MAIL=0
- W_MKT_INIT=0.35         # model ↔ market harman başlatma
- LEARN_RATE=0.05         # w_mkt için öğrenme hızı
- GOAL_LR=0.02            # lig bazlı gol tabanı ölçek learning rate
- PRED_MATCH_WINDOW_HRS=48  # tahmin-sonuç eşleşme toleransı
- TABLE_WEIGHT=0.12       # standings etkisi (±%12 sınır)
- STREAK_UNIT=0.02        # tek W/L adımı etkisi
- STREAK_MAX=0.08         # W/L toplam mutlak sınır
"""

import os, math, time, json, smtplib, traceback
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

import requests

# --- Ortak yardımcılar -------------------------------------------------------

TR_TZ = timezone(timedelta(hours=3))  # Türkiye
HEADERS_JSON = {"Accept": "application/json"}

def log(msg): print(f"[mailer] {msg}", flush=True)

def http_get(url, headers=None, params=None, timeout=25):
    try:
        r = requests.get(url, headers=headers or {}, params=params or {}, timeout=timeout)
        if r.status_code == 200:
            try:
                return r.json()
            except Exception:
                return None
        log(f"GET {url} -> {r.status_code}")
    except Exception as e:
        log(f"GET ERROR {url}: {e}")
    return None

def to_dt_utc(s):
    try:
        if not s:
            return None
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None

def safe_float(x, default=None):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def clamp(x, a, b): return max(a, min(b, x))
def norm_team(x: str): return (x or "").lower().replace(".", " ").replace("-", " ").replace(" fc","").strip()

def season_for_today():
    now = datetime.now(TR_TZ)
    y = now.year
    return y if now.month >= 7 else (y - 1)

# --- Secrets / ortam ---------------------------------------------------------

GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_PASS")
GMAIL_TO   = os.getenv("GMAIL_TO")
FD_TOKEN   = os.getenv("FOOTBALL_DATA_TOKEN")
ODDS_KEY   = os.getenv("ODDS_API_KEY")
APIFOOT    = (os.getenv("APIFOOTBALL_KEY") or "").strip()
MODE_ENV   = (os.getenv("MODE") or "AUTO").upper().strip()

TOP_N        = int(os.getenv("TOP_N", "5"))
MIN_CONF     = int(os.getenv("MIN_CONF", "0"))
HIGH_ALERT   = int(os.getenv("HIGH_ALERT", "90"))
OLD_LEAGUES  = [x.strip() for x in (os.getenv("OLD_LEAGUES", "bundesliga,bundesliga2").split(",")) if x.strip()]
ODDS_TTL_MIN = int(os.getenv("ODDS_TTL_MIN", "15"))
SPLIT_HIGH   = (os.getenv("SPLIT_HIGH_ALERT_MAIL", "0") == "1")

# Elo / Form ayarları
ELO_K            = float(os.getenv("ELO_K", "24"))
ELO_HOME_ADV     = float(os.getenv("ELO_HOME_ADV", "60"))
FORM_LOOKBACK    = int(os.getenv("FORM_LOOKBACK", "10"))
FORM_DAYS        = int(os.getenv("FORM_DAYS", "120"))
ALLOW_STATE_FILE = (os.getenv("ALLOW_STATE_FILE", "1") == "1")

# Otomatik öğrenme ayarları
W_MKT_INIT  = float(os.getenv("W_MKT_INIT", "0.35"))
LEARN_RATE  = float(os.getenv("LEARN_RATE", "0.05"))
GOAL_LR     = float(os.getenv("GOAL_LR", "0.02"))
PRED_MATCH_WINDOW_HRS = int(os.getenv("PRED_MATCH_WINDOW_HRS", "48"))

# Table/Streak ayarları
TABLE_WEIGHT = float(os.getenv("TABLE_WEIGHT", "0.12"))
STREAK_UNIT  = float(os.getenv("STREAK_UNIT",  "0.02"))
STREAK_MAX   = float(os.getenv("STREAK_MAX",   "0.08"))

if not (GMAIL_USER and GMAIL_PASS and GMAIL_TO):
    raise SystemExit("GMAIL_USER/GMAIL_PASS/GMAIL_TO secrets eksik.")

# --- State (kalibrasyon/Elo + Öğrenme) --------------------------------------

STATE_PATH = "model_state.json"

def _state_load():
    try:
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                st = json.load(f)
                st.setdefault("elo", {})
                st.setdefault("goal_scale", {})     # area -> ölçek (1.0)
                st.setdefault("w_mkt", W_MKT_INIT)  # harman ağırlık
                st.setdefault("pred_store", {})     # match_key -> tahmin/teşhis
                st.setdefault("metrics", {})        # kümülatif metrikler (opsiyonel)
                st.setdefault("last_saved", None)
                return st
    except Exception as e:
        log(f"state load err: {e}")
    return {"elo": {}, "goal_scale": {}, "w_mkt": W_MKT_INIT,
            "pred_store": {}, "metrics": {}, "last_saved": None}

def _state_save(st):
    if not ALLOW_STATE_FILE:
        log("STATE kaydı kapalı (ALLOW_STATE_FILE=0)")
        return
    try:
        st["last_saved"] = datetime.utcnow().isoformat() + "Z"
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=2)
        log(f"state saved -> {STATE_PATH}")
    except Exception as e:
        log(f"state save err: {e}")

STATE = _state_load()

def _team_key(area, name):
    a = (area or "Europe").strip().lower()
    n = (name or "").strip().lower()
    return f"{a}:{n}"

def elo_get(area, name):
    key = _team_key(area, name)
    return float(STATE.get("elo", {}).get(key, 1500.0))

def elo_set(area, name, val):
    key = _team_key(area, name)
    STATE.setdefault("elo", {})[key] = float(val)

def elo_expect(elo_a, elo_b, home_adv=0.0):
    diff = (elo_a + home_adv) - elo_b
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

def elo_update(area, home_name, away_name, result_hw):
    Eh = elo_get(area, home_name)
    Ea = elo_get(area, away_name)
    ph = elo_expect(Eh, Ea, ELO_HOME_ADV)
    pa = 1.0 - ph
    Eh_new = Eh + ELO_K * (result_hw - ph)
    Ea_new = Ea + ELO_K * ((1.0 - result_hw) - pa)
    elo_set(area, home_name, Eh_new)
    elo_set(area, away_name, Ea_new)

def get_goal_scale(area):
    return float(STATE["goal_scale"].get(area or "Europe", 1.0))

def set_goal_scale(area, val):
    STATE["goal_scale"][area or "Europe"] = float(val)

def get_w_mkt():
    try:
        return float(STATE.get("w_mkt", W_MKT_INIT))
    except:
        return W_MKT_INIT

def set_w_mkt(val):
    STATE["w_mkt"] = float(clamp(val, 0.0, 0.8))

# --- Odds sport-key haritalaması --------------------------------------------

def guess_sport_key(area: str, comp: str):
    a = (area or "").lower()
    c = (comp or "").lower()
    s = f"{a} {c}"
    mapping = {
        # Germany
        "bundesliga 3": "soccer_germany_bundesliga3",
        "bundesliga 2": "soccer_germany_bundesliga2",
        "bundesliga":   "soccer_germany_bundesliga",
        "dfb":          "soccer_germany_dfb_pokal",
        # Turkey
        "super lig":    "soccer_turkey_super_league",
        # England
        "premier":      "soccer_epl",
        "championship": "soccer_efl_championship",
        # Spain
        "la liga":      "soccer_spain_la_liga",
        # Italy
        "serie a":      "soccer_italy_serie_a",
        # France
        "ligue 1":      "soccer_france_ligue_one",
        # UEFA
        "champions":    "soccer_uefa_champs_league",
        "europa":       "soccer_uefa_europa_league",
        # Brazil
        "campeonato brasileiro": "soccer_brazil_campeonato",
        "brasileirao":           "soccer_brazil_campeonato",
        "brasileiro":            "soccer_brazil_campeonato",
    }
    for k, v in mapping.items():
        if k in s:
            return v
    return None

# --- Hava: takım -> şehir eşleşmesi -----------------------------------------

def guess_city_from_team(team_name: str):
    t = (team_name or "").lower()
    overrides = {
        # Türkiye
        "galatasaray": "Istanbul", "fenerbahce": "Istanbul", "beşiktaş": "Istanbul",
        "besiktas": "Istanbul", "basaksehir": "Istanbul", "trabzonspor": "Trabzon",
        # Almanya
        "bayern": "Munich", "dortmund": "Dortmund", "leipzig": "Leipzig",
        "leverkusen": "Leverkusen", "schalke": "Gelsenkirchen", "st. pauli": "Hamburg",
        # İspanya
        "real madrid": "Madrid", "barcelona": "Barcelona", "atlético": "Madrid", "atletico": "Madrid",
        # İtalya
        "juventus": "Turin", "inter": "Milan", "milan": "Milan", "roma": "Rome", "lazio": "Rome", "napoli": "Naples",
        # Fransa
        "psg": "Paris", "paris saint-germain": "Paris", "lyon": "Lyon", "marseille": "Marseille",
        # Brezilya
        "corinthians": "Sao Paulo", "palmeiras": "Sao Paulo", "santos": "Santos",
        "flamengo": "Rio de Janeiro", "fluminense": "Rio de Janeiro", "botafogo": "Rio de Janeiro",
        "gremio": "Porto Alegre", "grêmio": "Porto Alegre",
        "internacional": "Porto Alegre", "atletico mineiro": "Belo Horizonte",
    }
    for k, city in overrides.items():
        if k in t:
            return city
    return team_name  # fallback

def fetch_weather_note(home_team):
    city = guess_city_from_team(home_team)
    geo = http_get("https://geocoding-api.open-meteo.com/v1/search",
                   params={"name": city, "count": 1, "language": "en"})
    if not geo or not geo.get("results"):
        return None
    lat = geo["results"][0]["latitude"]; lon = geo["results"][0]["longitude"]
    wx = http_get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "timezone": "auto"
    })
    if not wx:
        return None
    try:
        temps = wx["hourly"]["temperature_2m"][:6]
        prec  = wx["hourly"]["precipitation"][:6]
        wind  = wx["hourly"]["wind_speed_10m"][:6]
        tavg  = sum(temps)/len(temps); pavg = sum(prec)/len(prec); wavg = sum(wind)/len(wind)
        return f"Hava: {tavg:.0f}°C, yağış {pavg:.1f}mm, rüzgâr {wavg:.0f} km/s"
    except Exception:
        return None

def parse_weather(wx_text):
    if not wx_text: return (None, None)
    wind = None; precip = None
    try:
        if "rüzgâr" in wx_text:
            wind = safe_float(wx_text.split("rüzgâr")[1].split("km/s")[0].strip().split()[-1], None)
        if "yağış" in wx_text:
            precip = safe_float(wx_text.split("yağış")[1].split("mm")[0].strip().split()[-1], None)
    except Exception:
        pass
    return (wind, precip)

# --- Football-Data (fixtures) + OpenLigaDB (fallback) ------------------------

def fetch_fd_fixtures(date_str):
    if not FD_TOKEN:
        return []
    url = "https://api.football-data.org/v4/matches"
    headers = {"X-Auth-Token": FD_TOKEN, **HEADERS_JSON}

    tr_day = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=TR_TZ)
    from_utc = (tr_day - timedelta(days=1)).astimezone(timezone.utc).strftime("%Y-%m-%d")
    to_utc   = (tr_day + timedelta(days=1)).astimezone(timezone.utc).strftime("%Y-%m-%d")

    data = http_get(url, headers=headers, params={"dateFrom": from_utc, "dateTo": to_utc})
    out = []
    if data and data.get("matches"):
        for m in data["matches"]:
            dtp = to_dt_utc(m.get("utcDate"))
            if not dtp:
                continue
            if dtp.astimezone(TR_TZ).strftime("%Y-%m-%d") != date_str:
                continue
            if m.get("status") not in ("SCHEDULED", "TIMED"):
                continue
            comp = m.get("competition", {}) or {}
            area = (comp.get("area", {}) or {}).get("name", "")
            ht = m.get("homeTeam", {}) or {}
            at = m.get("awayTeam", {}) or {}
            out.append({
                "source": "FD",
                "utc_kickoff": dtp,
                "home": ht.get("name"),
                "away": at.get("name"),
                "home_id": ht.get("id"),
                "away_id": at.get("id"),
                "area": area,
                "competition": comp.get("name", ""),
                "competition_id": comp.get("id"),  # standings için
                "id": m.get("id"),
            })
    log(f"FD fixtures (TR={date_str}) -> {len(out)}")
    return out

def fetch_openligadb_day(date_str):
    y, m, d = date_str.split("-")
    leagues = [(lg.strip(), int(y)) for lg in OLD_LEAGUES if lg.strip()]
    out = []
    for lg, season in leagues:
        url = f"https://www.openligadb.de/api/getmatchdata/{lg}/{season}"
        data = http_get(url)
        if not data:
            continue
        for mm in data:
            dt = mm.get("MatchDateTimeUTC") or mm.get("MatchDateTime")
            dtp = to_dt_utc(dt) if dt else None
            if not dtp:
                continue
            if dtp.astimezone(TR_TZ).strftime("%Y-%m-%d") != date_str:
                continue
            comp = mm.get("LeagueName") or lg
            out.append({
                "source":"OLD",
                "utc_kickoff": dtp,
                "home": mm.get("Team1",{}).get("TeamName"),
                "away": mm.get("Team2",{}).get("TeamName"),
                "home_id": None, "away_id": None,
                "area": "Germany",
                "competition": comp,
                "competition_id": None,
                "id": f"old:{mm.get('MatchID')}",
            })
    log(f"OpenLigaDB fixtures (TR={date_str}) -> {len(out)}")
    return out

# --- 3. Fallback: The Odds API'den fikstür listesi --------------------------

_odds_cache = {}  # {skey: {"ts": epoch, "data": list}}

def _fetch_odds_sport_cached(skey):
    now = time.time()
    ent = _odds_cache.get(skey)
    if ent and (now - ent["ts"] < ODDS_TTL_MIN*60):
        return ent["data"]
    url = f"https://api.the-odds-api.com/v4/sports/{skey}/odds/"
    params = {"regions":"eu,uk","markets":"h2h","oddsFormat":"decimal","apiKey":ODDS_KEY}
    data = http_get(url, params=params) if ODDS_KEY else None
    _odds_cache[skey] = {"ts": now, "data": data or []}
    return _odds_cache[skey]["data"]

def fetch_odds_fixtures(date_str):
    if not ODDS_KEY:
        return []
    keys = [
        ("soccer_epl", "England", "Premier League"),
        ("soccer_spain_la_liga", "Spain", "La Liga"),
        ("soccer_italy_serie_a", "Italy", "Serie A"),
        ("soccer_france_ligue_one", "France", "Ligue 1"),
        ("soccer_germany_bundesliga", "Germany", "Bundesliga"),
        ("soccer_turkey_super_league", "Turkey", "Super Lig"),
        ("soccer_brazil_campeonato", "Brazil", "Campeonato Brasileiro"),
        ("soccer_efl_championship", "England", "Championship"),
        ("soccer_uefa_champs_league", "Europe", "UEFA Champions League"),
    ]
    out = []
    for skey, area, comp in keys:
        data = _fetch_odds_sport_cached(skey)
        if not data or not isinstance(data, list):
            continue
        for ev in data:
            dtp = to_dt_utc(ev.get("commence_time"))
            if not dtp:
                continue
            if dtp.astimezone(TR_TZ).strftime("%Y-%m-%d") != date_str:
                continue
            out.append({
                "source": "ODDS",
                "utc_kickoff": dtp,
                "home": ev.get("home_team"),
                "away": ev.get("away_team"),
                "home_id": None, "away_id": None,
                "area": area,
                "competition": comp,
                "competition_id": None,
                "id": f"odds:{skey}:{ev.get('id','')}",
            })
    log(f"OddsAPI fixtures (TR={date_str}) -> {len(out)}")
    return out

def fetch_fixtures(date_str):
    fixtures = fetch_fd_fixtures(date_str)
    if not fixtures:
        log("FD boş → OpenLigaDB fallback deneniyor…")
        fixtures = fetch_openligadb_day(date_str)
    if not fixtures:
        log("OpenLigaDB de boş → The Odds API event fallback deneniyor…")
        fixtures = fetch_odds_fixtures(date_str)
    return fixtures

# --- Oranlar (avg) -----------------------------------------------------------

def fetch_odds_avg(area, comp, home, away):
    if not ODDS_KEY:
        return None
    skey = guess_sport_key(area, comp)
    if not skey:
        return None
    data = _fetch_odds_sport_cached(skey)
    if not data or not isinstance(data, list):
        return None
    def norm(x): return (x or "").lower().replace(".", "").replace("-", " ").replace(" fc","").strip()
    h, a = norm(home), norm(away)
    best = None
    for ev in data:
        comps = ev.get("bookmakers", [])
        if not comps:
            continue
        t1 = norm(ev.get("home_team")); t2 = norm(ev.get("away_team"))
        if (t1.startswith(h) and t2.startswith(a)) or (h.startswith(t1) and a.startswith(t2)):
            prices = {"home":[], "draw":[], "away":[]}
            for bk in comps:
                for mk in bk.get("markets", []):
                    for o in mk.get("outcomes", []):
                        nm = (o.get("name") or "").lower()
                        price = safe_float(o.get("price"), 0)
                        if nm in ("home","1"): prices["home"].append(price)
                        elif nm in ("draw","x"): prices["draw"].append(price)
                        elif nm in ("away","2"): prices["away"].append(price)
            if prices["home"] and prices["draw"] and prices["away"]:
                best = (
                    sum(prices["home"])/len(prices["home"]),
                    sum(prices["draw"])/len(prices["draw"]),
                    sum(prices["away"])/len(prices["away"]),
                )
                break
    if not best:
        return None
    o1, ox, o2 = best
    inv = (1/o1) + (1/ox) + (1/o2)
    p1, px, p2 = (1/o1)/inv, (1/ox)/inv, (1/o2)/inv
    return {"odds": (o1, ox, o2), "probs": (p1, px, p2)}

# --- Poisson + Lig tabanı ----------------------------------------------------

LEAGUE_GOAL_BASE = {
    "Turkey": 2.60, "England": 2.75, "Spain": 2.50, "Italy": 2.70,
    "France": 2.55, "Germany": 3.05, "Brazil": 2.35, "Europe": 2.70,
}

def base_total_goals(area):
    base = 2.60
    for k, v in LEAGUE_GOAL_BASE.items():
        if k.lower() in (area or "").lower():
            base = v
            break
    scale = get_goal_scale(area)
    return clamp(base * scale, 1.5, 3.8)

def poisson_prob(lambda_h, lambda_a):
    def pois(m, lam): return (lam**m) * math.exp(-lam) / math.factorial(m)
    p_home = p_draw = p_away = 0.0
    for gh in range(0, 11):
        ph = pois(gh, lambda_h)
        for ga in range(0, 11):
            pa = pois(ga, lambda_a)
            if gh > ga:  p_home += ph*pa
            elif gh == ga: p_draw += ph*pa
            else:         p_away += ph*pa
    return p_home, p_draw, p_away

def poisson_over_prob(lam, line_int):
    lam = max(0.05, lam)
    start = int(line_int) + 1
    s = 0.0; m = start
    while m <= start + 40:
        s += (lam**m) * math.exp(-lam) / math.factorial(m)
        m += 1
    return clamp(s, 0.0, 1.0)

def blend_model_market(model_probs, market_probs):
    if not market_probs:
        return model_probs
    w = get_w_mkt()
    return tuple((1-w)*m + w*mk for m, mk in zip(model_probs, market_probs))

# --- Kart / Korner — lig tabanı + tempo + hava + (opsiyonel) API-Football ----

LEAGUE_CARD_BASE = {
    "Germany": 4.7, "Turkey": 5.1, "England": 4.1, "Spain": 5.0, "Italy": 4.8,
    "France": 4.3, "Brazil": 5.2, "Europe": 4.6
}
LEAGUE_CORNER_BASE = {
    "Germany": 9.4, "Turkey": 9.2, "England": 10.1, "Spain": 9.1, "Italy": 9.5,
    "France": 9.0, "Brazil": 8.7, "Europe": 9.2
}

def base_from_area(area, table, default):
    for k, v in table.items():
        if k.lower() in (area or "").lower():
            return v
    return default

# --- API-Football: opsiyonel istatistik ipucu --------------------------------

APIFOOT_BASE = "https://v3.football.api-sports.io"
_API_LEAGUE_MAP = {
    "England|Premier League": 39,
    "England|Championship":   40,
    "England|League One":     41,
    "England|League Two":     42,
    "Spain|La Liga":          140,
    "Spain|La Liga 2":        141,
    "Italy|Serie A":          135,
    "Italy|Serie B":          136,
    "France|Ligue 1":         61,
    "France|Ligue 2":         62,
    "Germany|Bundesliga":     78,
    "Germany|2. Bundesliga":  79,
    "Germany|DFB-Pokal":      81,
    "Turkey|Super Lig":       203,
    "Turkey|1. Lig":          204,
    "Brazil|Campeonato Brasileiro": 71,
    "Brazil|Serie B":         72,
}
_apifoot_team_cache = {}   # search_name.lower() -> team_id
_apifoot_stat_cache = {}   # (league_id, season, team_id) -> stats_json

def _apifoot_get(path, params):
    if not APIFOOT:
        return None
    headers = {"x-apisports-key": APIFOOT}
    try:
        data = http_get(f"{APIFOOT_BASE}{path}", headers=headers, params=params)
        return (data or {}).get("response", None)
    except Exception as e:
        log(f"apifoot GET err: {e}")
        return None

def _apifoot_find_team_id(team_name):
    if not team_name:
        return None
    key = (team_name or "").strip().lower()
    ent = _apifoot_team_cache.get(key)
    if ent is not None:
        return ent
    resp = _apifoot_get("/teams", {"search": team_name})
    tid = None
    try:
        if resp:
            tid = ((resp[0] or {}).get("team") or {}).get("id")
    except Exception:
        tid = None
    _apifoot_team_cache[key] = tid
    return tid

def _apifoot_team_statistics(league_id, season, team_id):
    if not (league_id and season and team_id):
        return None
    cache_key = (league_id, season, team_id)
    if cache_key in _apifoot_stat_cache:
        return _apifoot_stat_cache[cache_key]
    resp = _apifoot_get("/teams/statistics", {
        "league": league_id, "season": season, "team": team_id
    })
    _apifoot_stat_cache[cache_key] = resp[0] if (resp and len(resp)>0) else None
    return _apifoot_stat_cache[cache_key]

def _apifoot_hint_cards_corners(area, comp, home, away):
    if not APIFOOT:
        return None
    lig_key = f"{(area or '').strip()}|{(comp or '').strip()}"
    lig_id = _API_LEAGUE_MAP.get(lig_key)
    if not lig_id:
        return None
    ssn = season_for_today()
    try:
        h_id = _apifoot_find_team_id(home)
        a_id = _apifoot_find_team_id(away)
        h_stat = _apifoot_team_statistics(lig_id, ssn, h_id) if h_id else None
        a_stat = _apifoot_team_statistics(lig_id, ssn, a_id) if a_id else None
        if not (h_stat or a_stat):
            return None

        def cards_per_game(stat):
            if not stat: return None
            played = (((stat.get("fixtures") or {}).get("played") or {}).get("total")) or 0
            played = int(played) if played else 0
            if played <= 0: return None
            cards = (stat.get("cards") or {})
            total = 0.0
            for v in cards.values():
                t = v.get("total")
                if t is None: continue
                t = safe_float(t, None)
                if t is not None: total += t
            return total / played if total > 0 else None

        def corners_per_game(stat):
            if not stat: return None
            played = (((stat.get("fixtures") or {}).get("played") or {}).get("total")) or 0
            played = int(played) if played else 0
            if played <= 0: return None
            corners = (stat.get("corners") or {}).get("total")
            if corners is None: return None
            c = safe_float(corners, None)
            if c is None: return None
            return c / played

        h_cards = cards_per_game(h_stat)
        a_cards = cards_per_game(a_stat)
        h_corners = corners_per_game(h_stat)
        a_corners = corners_per_game(a_stat)

        out = {}
        if h_cards or a_cards:
            vals = [v for v in [h_cards, a_cards] if v]
            if vals:
                out["mu_cards_hint"] = sum(vals)/len(vals) * 2.0 * 0.5
        if h_corners or a_corners:
            vals = [v for v in [h_corners, a_corners] if v]
            if vals:
                out["mu_corners_hint"] = sum(vals)/len(vals) * 2.0 * 0.5
        return out if out else None
    except Exception as e:
        log(f"apifoot hint err: {e}")
        return None

# --- API-Football: Standings/Streak Fallback (YENİ) -------------------------

_APIF_STANDINGS_CACHE = {}  # (league_id, season) -> {"total_teams":N, "by_id": {team_id: {"position":pos}}}
_APIF_FIXTURES_CACHE  = {}  # (team_id, league_id, season, limit) -> ["W","D","L",...]

def _apifoot_standings(league_id, season):
    """/standings -> lig tablosu"""
    if not APIFOOT or not league_id or not season:
        return None
    key = (league_id, season)
    if key in _APIF_STANDINGS_CACHE:
        return _APIF_STANDINGS_CACHE[key]
    resp = _apifoot_get("/standings", {"league": league_id, "season": season})
    by_id = {}; total = None
    try:
        lg = ((resp or [{}])[0] or {}).get("league", {}) if resp else {}
        groups = lg.get("standings") or []
        table = groups[0] if groups else []
        total = len(table)
        for row in table:
            tid  = ((row.get("team") or {}).get("id"))
            rank = safe_float(row.get("rank"), None)
            if tid is not None and rank is not None:
                by_id[int(tid)] = {"position": int(rank)}
        if by_id and total:
            _APIF_STANDINGS_CACHE[key] = {"total_teams": total, "by_id": by_id}
        else:
            _APIF_STANDINGS_CACHE[key] = None
    except Exception as e:
        log(f"apif standings parse err: {e}")
        _APIF_STANDINGS_CACHE[key] = None
    return _APIF_STANDINGS_CACHE[key]

def _apif_get_position_by_name(league_id, season, team_name):
    """Takım adından pozisyon bul (API-Football team_id araması yapar)."""
    st = _apifoot_standings(league_id, season)
    if not (st and team_name):
        return (None, None)
    tid = _apifoot_find_team_id(team_name)
    if not tid:
        return (None, st.get("total_teams"))
    row = st["by_id"].get(int(tid))
    if not row:
        return (None, st.get("total_teams"))
    return (row.get("position"), st.get("total_teams"))

def _apif_recent_outcomes(team_id, league_id, season, limit=6):
    """Takımın son maçlarını çekip W/D/L listesi döndürür (yalnızca ilgili lig)."""
    if not (APIFOOT and team_id and league_id and season):
        return []
    key = (int(team_id), int(league_id), int(season), int(limit))
    if key in _APIF_FIXTURES_CACHE:
        return _APIF_FIXTURES_CACHE[key]
    resp = _apifoot_get("/fixtures", {
        "team": team_id, "league": league_id, "season": season,
        "status": "FT", "last": limit
    })
    outcomes = []
    try:
        for f in (resp or []):
            teams = (f.get("teams") or {})
            home  = (teams.get("home") or {})
            away  = (teams.get("away") or {})
            gh = safe_float((f.get("goals") or {}).get("home"), None)
            ga = safe_float((f.get("goals") or {}).get("away"), None)
            if gh is None or ga is None:
                continue
            is_home = (int(home.get("id") or -1) == int(team_id))
            if is_home:
                outcomes.append("W" if gh>ga else "D" if gh==ga else "L")
            else:
                outcomes.append("W" if ga>gh else "D" if ga==gh else "L")
    except Exception as e:
        log(f"apif recent outcomes parse err: {e}")
    _APIF_FIXTURES_CACHE[key] = outcomes
    return outcomes

def _apif_streak_for_team(team_name, league_id, season):
    """Takım adıyla W/L streak hesapla (API-Football)."""
    tid = _apifoot_find_team_id(team_name)
    if not tid:
        return (0.0, "")
    outs = _apif_recent_outcomes(tid, league_id, season, limit=6)
    if not outs:
        return (0.0, "")
    streak_char = None; count = 0
    for res in outs:  # son maçtan geri
        if streak_char is None:
            if res == "D":
                return (0.0, "D0")
            streak_char = res; count = 1
        else:
            if res == streak_char:
                count += 1
            else:
                break
        if count >= 6:
            break
    if streak_char == "W":
        score = min(count, 5) * STREAK_UNIT; txt = f"W{count}"
    elif streak_char == "L":
        score = -min(count, 5) * STREAK_UNIT; txt = f"L{count}"
    else:
        score = 0.0; txt = "D0"
    score = clamp(score, -STREAK_MAX, STREAK_MAX)
    return (score, txt)

def _table_adj_from_any(fix):
    """Önce FD standings, değilse API-Football standings."""
    # FD ile (varsa)
    if fix.get("competition_id") and fix.get("home_id") and fix.get("away_id"):
        h_rankpct, h_pos, N = _table_strength(fix["competition_id"], fix["home_id"])
        a_rankpct, a_pos, _ = _table_strength(fix["competition_id"], fix["away_id"])
        if (h_rankpct is not None) and (a_rankpct is not None):
            diff = h_rankpct - a_rankpct
            table_adj = clamp(diff * TABLE_WEIGHT, -TABLE_WEIGHT, TABLE_WEIGHT)
            txt = f" | Table {h_pos}/{N} vs {a_pos}/{N} (Adj {int(table_adj*100)}%)"
            return table_adj, txt

    # API-Football fallback
    lig_key = f"{(fix.get('area') or '').strip()}|{(fix.get('competition') or '').strip()}"
    lig_id = _API_LEAGUE_MAP.get(lig_key)
    if not lig_id:
        return 0.0, ""
    ssn = season_for_today()
    h_pos, N = _apif_get_position_by_name(lig_id, ssn, fix.get("home"))
    a_pos, _ = _apif_get_position_by_name(lig_id, ssn, fix.get("away"))
    if h_pos and a_pos and N:
        h_rankpct = (N - h_pos) / (N - 1) if N > 1 else 0.5
        a_rankpct = (N - a_pos) / (N - 1) if N > 1 else 0.5
        diff = h_rankpct - a_rankpct
        table_adj = clamp(diff * TABLE_WEIGHT, -TABLE_WEIGHT, TABLE_WEIGHT)
        txt = f" | Table {h_pos}/{N} vs {a_pos}/{N} (Adj {int(table_adj*100)}%)"
        return table_adj, txt
    return 0.0, ""

def _streak_from_any(fix):
    """Önce FD streak, değilse API-Football streak."""
    # FD ile (varsa)
    if fix.get("home_id") and fix.get("away_id"):
        hs, hs_txt  = _team_streak(fix.get("home_id"), fix.get("home"))
        as_, as_txt = _team_streak(fix.get("away_id"), fix.get("away"))
        net = clamp(hs - as_, -STREAK_MAX, STREAK_MAX)
        if hs_txt or as_txt:
            return net, f" | Streak {hs_txt}/{as_txt} (Adj {int(net*100)}%)"

    # API-Football fallback
    lig_key = f"{(fix.get('area') or '').strip()}|{(fix.get('competition') or '').strip()}"
    lig_id = _API_LEAGUE_MAP.get(lig_key)
    if not lig_id:
        return 0.0, ""
    ssn = season_for_today()
    hs, hs_txt  = _apif_streak_for_team(fix.get("home"), lig_id, ssn)
    as_, as_txt = _apif_streak_for_team(fix.get("away"), lig_id, ssn)
    net = clamp(hs - as_, -STREAK_MAX, STREAK_MAX)
    if hs_txt or as_txt:
        return net, f" | Streak {hs_txt}/{as_txt} (Adj {int(net*100)}%)"
    return 0.0, ""

def model_cards_corners(area, lam_h, lam_a, wx_text, apifoot_hint=None):
    cards_base  = base_from_area(area, LEAGUE_CARD_BASE, 4.6)
    corner_base = base_from_area(area, LEAGUE_CORNER_BASE, 9.2)

    wind, precip = parse_weather(wx_text)
    tempo_factor = clamp((lam_h + lam_a) / 2.6, 0.7, 1.4)

    cards   = cards_base
    corners = corner_base * tempo_factor

    if apifoot_hint:
        if apifoot_hint.get("mu_cards_hint"):
            cards = 0.6 * cards + 0.4 * max(0.1, apifoot_hint["mu_cards_hint"])
        if apifoot_hint.get("mu_corners_hint"):
            corners = 0.6 * corners + 0.4 * max(0.1, apifoot_hint["mu_corners_hint"])

    if precip is not None:
        cards   *= 1.00 + min(0.10, 0.02  * max(0.0, precip))
        corners *= 1.00 - min(0.10, 0.015 * max(0.0, precip))
    if wind is not None:
        corners *= 1.00 + min(0.08, 0.003 * max(0.0, wind))
        cards   *= 1.00 + min(0.05, 0.002 * max(0.0, wind))

    p_corners_8_5 = poisson_over_prob(max(0.1, corners), 8.5)
    p_corners_9_5 = poisson_over_prob(max(0.1, corners), 9.5)
    p_cards_3_5   = poisson_over_prob(max(0.1, cards),   3.5)
    p_cards_4_5   = poisson_over_prob(max(0.1, cards),   4.5)

    return {
        "mu_cards": cards, "mu_corners": corners,
        "p_over_cards_3_5": p_cards_3_5, "p_over_cards_4_5": p_cards_4_5,
        "p_over_corners_8_5": p_corners_8_5, "p_over_corners_9_5": p_corners_9_5
    }

# --- Opponent-adjusted form (FD team-id ile) ---------------------------------

def _fd_team_matches(team_id, days=120):
    if not (FD_TOKEN and team_id):
        return []
    headers = {"X-Auth-Token": FD_TOKEN, **HEADERS_JSON}
    to_dt   = datetime.utcnow().date()
    from_dt = to_dt - timedelta(days=days)
    url = f"https://api.football-data.org/v4/teams/{team_id}/matches"
    data = http_get(url, headers=headers, params={
        "status": "FINISHED",
        "dateFrom": from_dt.isoformat(),
        "dateTo": to_dt.isoformat()
    })
    if not data or not data.get("matches"):
        return []
    ms = []
    for m in data["matches"]:
        dtp = to_dt_utc(m.get("utcDate"))
        if not dtp:
            continue
        ms.append((dtp, m))
    ms.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in ms]

_FORM_CACHE = {}  # team_id -> (adj, txt)

def _form_adjust_from_matches(team_id, area, team_name):
    if team_id in _FORM_CACHE:
        return _FORM_CACHE[team_id]

    matches = _fd_team_matches(team_id, days=FORM_DAYS)
    if not matches:
        _FORM_CACHE[team_id] = (0.0, "")
        return (0.0, "")

    n = 0
    score_sum = 0.0
    for m in matches:
        ht = (m.get("homeTeam", {}) or {}).get("name", "")
        at = (m.get("awayTeam", {}) or {}).get("name", "")
        score = (m.get("score", {}) or {}).get("fullTime", {}) or {}

        gh = score.get("home"); ga = score.get("away")
        if gh is None or ga is None:
            continue
        gh = int(gh); ga = int(ga)

        is_home = (ht == team_name)
        gf = gh if is_home else ga
        ga_ = ga if is_home else gh

        opp_name = at if is_home else ht
        opp_elo = elo_get(area, opp_name)

        w = 1.0 + clamp((opp_elo - 1500.0) / 1000.0, -0.2, 0.2)
        score_sum += (gf - ga_) * w
        n += 1
        if n >= FORM_LOOKBACK:
            break

    adj = 0.0 if n == 0 else clamp(math.tanh((score_sum / n) / 3.0) * 0.12, -0.15, 0.15)
    txt = f"FormAdj {('+' if adj>=0 else '')}{int(adj*100)}%"
    _FORM_CACHE[team_id] = (adj, txt)
    return (adj, txt)

def build_form_cache_for_date(fixtures):
    seen = set()
    for fx in fixtures:
        for tid in (fx.get("home_id"), fx.get("away_id")):
            if tid and tid not in seen:
                seen.add(tid)
                try:
                    _form_adjust_from_matches(
                        tid, fx.get("area","Europe"),
                        fx.get("home") if tid==fx.get("home_id") else fx.get("away")
                    )
                    time.sleep(0.05)
                except Exception as e:
                    log(f"form cache err team_id={tid}: {e}")

# --- Standings (TableAdj) ----------------------------------------------------

_STANDINGS_CACHE = {}  # competition_id -> {"total_teams":N, "by_id": {team_id: {...}}}

def _fd_competition_standings(comp_id):
    if not (FD_TOKEN and comp_id):
        return None
    if comp_id in _STANDINGS_CACHE:
        return _STANDINGS_CACHE[comp_id]
    headers = {"X-Auth-Token": FD_TOKEN, **HEADERS_JSON}
    url = f"https://api.football-data.org/v4/competitions/{comp_id}/standings"
    data = http_get(url, headers=headers)
    by_id = {}
    total_teams = None
    try:
        for st in (data.get("standings") or []):
            if (st.get("type") or "").upper() != "TOTAL":
                continue
            table = st.get("table") or []
            total_teams = len(table)
            for row in table:
                team = (row.get("team") or {})
                tid  = team.get("id")
                pos  = safe_float(row.get("position"), None)
                played = safe_float(row.get("playedGames"), None)
                pts    = safe_float(row.get("points"), None)
                ppg    = (pts/played) if (pts is not None and played and played>0) else None
                if tid is not None:
                    by_id[int(tid)] = {"position": int(pos) if pos else None,
                                       "played": int(played) if played else None,
                                       "points": int(pts) if pts else None,
                                       "ppg": ppg}
        if by_id and total_teams:
            _STANDINGS_CACHE[comp_id] = {"total_teams": total_teams, "by_id": by_id}
            return _STANDINGS_CACHE[comp_id]
    except Exception as e:
        log(f"standings parse err: {e}")
    return None

def _table_strength(comp_id, team_id):
    st = _fd_competition_standings(comp_id)
    if not (st and team_id):
        return (None, None, None)
    row = (st["by_id"].get(int(team_id))) if isinstance(team_id, int) or str(team_id).isdigit() else None
    if not row or not row.get("position"):
        return (None, None, None)
    N = st["total_teams"] or 20
    pos = row["position"]
    if N <= 1:
        rank_pct = 0.5
    else:
        # 1 (lider) -> 1.0, N (son) -> 0.0
        rank_pct = (N - pos) / (N - 1)
    return (rank_pct, pos, N)

# --- Streak (W/L) ------------------------------------------------------------

_STREAK_CACHE = {}  # team_id -> (score, txt)

def _team_outcome_for(m, team_name):
    score = (m.get("score", {}) or {}).get("fullTime", {}) or {}
    gh = score.get("home"); ga = score.get("away")
    if gh is None or ga is None:
        return None
    ht = (m.get("homeTeam", {}) or {}).get("name", "")
    is_home = (ht == team_name)
    if is_home:
        if gh > ga: return "W"
        if gh == ga: return "D"
        return "L"
    else:
        if ga > gh: return "W"
        if ga == gh: return "D"
        return "L"

def _team_streak(team_id, team_name):
    if not team_id:
        return (0.0, "")
    if team_id in _STREAK_CACHE:
        return _STREAK_CACHE[team_id]
    ms = _fd_team_matches(team_id, days=min(FORM_DAYS, 120))
    if not ms:
        _STREAK_CACHE[team_id] = (0.0, "")
        return (0.0, "")
    streak_char = None
    count = 0
    for m in ms:
        res = _team_outcome_for(m, team_name)
        if res is None:
            continue
        if streak_char is None:
            streak_char = res
            if res == "D":
                count = 0
                break
            count = 1
        else:
            if res == streak_char:
                count += 1
            else:
                break
        if count >= 6:
            break
    if streak_char == "W":
        score = min(count, 5) * STREAK_UNIT
        txt = f"W{count}"
    elif streak_char == "L":
        score = -min(count, 5) * STREAK_UNIT
        txt = f"L{count}"
    else:
        score = 0.0
        txt = "D0"
    score = max(-STREAK_MAX, min(STREAK_MAX, score))
    _STREAK_CACHE[team_id] = (score, txt)
    return (score, txt)

# --- Tahmin/sonuç eşleşme & öğrenme -----------------------------------------

def match_key_from_fixture(fx):
    if fx.get("id"):
        return f"{fx.get('source','?')}:{fx['id']}"
    dt = (fx.get("utc_kickoff") or datetime.now(timezone.utc)).strftime("%Y%m%d")
    return f"{norm_team(fx.get('home'))}|{norm_team(fx.get('away'))}|{dt}"

def match_key_from_result(r):
    if r.get("id"):
        return f"FD:{r['id']}"
    return f"{norm_team(r.get('home'))}|{norm_team(r.get('away'))}|{(r.get('utc_kickoff') or datetime.utcnow()).strftime('%Y%m%d')}"

def brier_score(probs, outcome_idx):
    if probs is None: return None
    y = [0.0, 0.0, 0.0]; y[outcome_idx] = 1.0
    return sum((p - t)**2 for p, t in zip(probs, y)) / 3.0

def record_prediction(fx, rated, model_probs, market_probs, blended_probs, wx_adj, elo_adj, net_form):
    mk = match_key_from_fixture(fx)
    STATE["pred_store"][mk] = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "area": fx.get("area","Europe"),
        "competition": fx.get("competition",""),
        "home": fx.get("home"), "away": fx.get("away"),
        "utc_kickoff": (fx.get("utc_kickoff") or datetime.now(timezone.utc)).isoformat(),
        "probs_model": model_probs,
        "probs_market": market_probs,
        "probs_blend": blended_probs,
        "pick": rated["pick"],
        "conf_pct": rated["confidence"],
        "lam_h": rated["lambda_h"], "lam_a": rated["lambda_a"],
        "wx_adj": wx_adj, "elo_adj": elo_adj, "net_form": net_form,
        "w_mkt_used": get_w_mkt()
    }

def analyze_and_learn(results):
    if not results:
        return ""

    analyzed = 0
    correct = 0
    sum_brier_model = 0.0
    sum_brier_market = 0.0
    sum_brier_blend = 0.0
    market_helped = 0
    goal_updates = {}

    for r in results:
        h, a = r.get("home_goals"), r.get("away_goals")
        if h is None or a is None:
            continue
        if h > a: out_idx = 0
        elif h == a: out_idx = 1
        else: out_idx = 2

        mk_id = match_key_from_result(r)
        cand = STATE["pred_store"].get(mk_id)
        if not cand:
            best_mk = None; best_dt = None; res_dt = r.get("utc_kickoff")
            if res_dt is None:
                res_dt = datetime.utcnow().replace(tzinfo=timezone.utc)
            else:
                if isinstance(res_dt, str):
                    res_dt = to_dt_utc(res_dt) or datetime.utcnow().replace(tzinfo=timezone.utc)
            for mk, ent in STATE["pred_store"].items():
                if norm_team(ent.get("home")) == norm_team(r.get("home")) and norm_team(ent.get("away")) == norm_team(r.get("away")):
                    try:
                        pdt = to_dt_utc(ent.get("utc_kickoff"))
                        if not pdt:
                            continue
                        if abs((pdt - res_dt).total_seconds()) <= PRED_MATCH_WINDOW_HRS*3600:
                            if (best_dt is None) or (abs((pdt-res_dt).total_seconds()) < abs((best_dt-res_dt).total_seconds())):
                                best_dt = pdt; best_mk = mk
                    except:
                        pass
            cand = STATE["pred_store"].get(best_mk) if best_mk else None

        if not cand:
            continue

        analyzed += 1
        probs_m = cand.get("probs_model")
        probs_k = cand.get("probs_market")
        probs_b = cand.get("probs_blend")

        bs_m = brier_score(probs_m, out_idx)
        bs_k = brier_score(probs_k, out_idx) if probs_k else None
        bs_b = brier_score(probs_b, out_idx)

        if bs_m is not None: sum_brier_model += bs_m
        if bs_b is not None: sum_brier_blend += bs_b
        if bs_k is not None: sum_brier_market += bs_k

        pred_pick = cand.get("pick")
        if (out_idx == 0 and pred_pick=="1") or (out_idx==1 and pred_pick=="X") or (out_idx==2 and pred_pick=="2"):
            correct += 1

        if bs_k is not None and bs_m is not None:
            delta = clamp(bs_m - bs_k, -0.4, 0.4)  # >0 → market daha iyi
            if delta > 0.0:
                set_w_mkt(get_w_mkt() + LEARN_RATE * delta)
                market_helped += 1
            else:
                set_w_mkt(get_w_mkt() + LEARN_RATE * delta)

        area = r.get("area","Europe")
        lam_sum = safe_float(cand.get("lam_h"), 1.3) + safe_float(cand.get("lam_a"), 1.3)
        lam_sum = max(0.5, lam_sum)
        realized = float(h + a)
        ratio = realized / lam_sum
        s_old = get_goal_scale(area)
        s_new = clamp(s_old * (1.0 + GOAL_LR*(ratio - 1.0)), 0.85, 1.20)
        if abs(s_new - s_old) > 1e-6:
            set_goal_scale(area, s_new)
            goal_updates[area] = s_new

    if analyzed == 0:
        return "Öğrenme: Eşleşen tahmin bulunamadı (pred_store/FD id)."

    acc = 100.0 * correct / analyzed
    txt = []
    txt.append(f"Öğrenme Özeti: {analyzed} maç, doğruluk {acc:.1f}%")
    if sum_brier_model>0 and sum_brier_blend>0:
        txt.append(f"Brier(model)≈{(sum_brier_model/analyzed):.3f} | Brier(harman)≈{(sum_brier_blend/analyzed):.3f}")
    if sum_brier_market>0:
        txt.append(f"Brier(market)≈{(sum_brier_market/analyzed):.3f} | market iyileştirdi: {market_helped} maç")
    txt.append(f"w_mkt (yeni) = {get_w_mkt():.2f}")
    if goal_updates:
        ups = ", ".join([f"{k}:×{get_goal_scale(k):.3f}" for k in sorted(goal_updates)])
        txt.append(f"Gol tabanı ölçek güncellendi → {ups}")
    return " | ".join(txt)

# --- Derecelendirme ----------------------------------------------------------

def rate_fixture(fix, odds_info):
    area = fix["area"] or "Europe"
    tot  = base_total_goals(area)

    ah    = 1.12
    noise = (len((fix["home"] or "")) - len((fix["away"] or ""))) * 0.01
    lam_h = max(0.2, tot*0.5*ah        + noise)
    lam_a = max(0.2, tot*0.5*(2 - ah)  - noise)

    # Hava
    wx = fetch_weather_note(fix["home"])
    wx_adj = 1.0
    if wx:
        wind, precip = parse_weather(wx)
        try:
            if wind   is not None: wx_adj -= min(0.08, 0.003 * wind)
            if precip is not None: wx_adj -= min(0.08, 0.01  * precip)
            wx_adj = clamp(wx_adj, 0.8, 1.0)
            lam_h *= wx_adj; lam_a *= wx_adj
        except Exception:
            pass

    # Elo etkisi
    Eh = elo_get(area, fix["home"]); Ea = elo_get(area, fix["away"])
    elo_diff = (Eh + ELO_HOME_ADV) - Ea
    elo_adj  = clamp((elo_diff/400.0)*0.15, -0.20, 0.20)
    lam_h *= (1.0 + elo_adj); lam_a *= (1.0 - elo_adj)

    # Opp-adjusted form
    form_bits = []
    home_adj = away_adj = 0.0
    if fix.get("home_id"):
        home_adj, t = _form_adjust_from_matches(fix["home_id"], area, fix["home"])
        form_bits.append(t)
    if fix.get("away_id"):
        away_adj, t = _form_adjust_from_matches(fix["away_id"], area, fix["away"])
        form_bits.append(t)
    net_form = clamp(home_adj - away_adj, -0.18, 0.18)
    lam_h *= (1.0 + net_form); lam_a *= (1.0 - net_form)
    form_txt = (" | " + " / ".join(form_bits)) if form_bits else ""

    # TableAdj (FD varsa FD, yoksa API-Football)
    table_adj, table_txt = _table_adj_from_any(fix)
    lam_h *= (1.0 + table_adj); lam_a *= (1.0 - table_adj)

    # Streak (FD varsa FD, yoksa API-Football)
    net_streak, streak_txt = _streak_from_any(fix)
    lam_h *= (1.0 + net_streak); lam_a *= (1.0 - net_streak)

    # Model 1X2
    m_home, m_draw, m_away = poisson_prob(lam_h, lam_a)
    model_probs = (m_home, m_draw, m_away)

    # Market karışımı
    market_probs = None
    odds_txt = ""
    if odds_info:
        o1,oX,o2 = odds_info["odds"]
        p1,px,p2 = odds_info["probs"]
        market_probs = (p1,px,p2)
        odds_txt = f" | Odds(avg) 1/X/2: {o1:.2f}/{oX:.2f}/{o2:.2f}"

    p_home, p_draw, p_away = blend_model_market(model_probs, market_probs)
    blended_probs = (p_home, p_draw, p_away)

    picks = [("1", p_home), ("X", p_draw), ("2", p_away)]
    picks.sort(key=lambda x: x[1], reverse=True)
    pick, conf = picks[0]; conf_pct = int(round(conf*100))

    # Kart/Korner — API-Football ipucunu harmanla
    apihint = _apifoot_hint_cards_corners(fix.get("area"), fix.get("competition"),
                                          fix.get("home"), fix.get("away"))
    kk = model_cards_corners(area, lam_h, lam_a, wx, apifoot_hint=apihint)
    kk_txt = (f" | Korner μ≈{kk['mu_corners']:.1f} (Üst8.5 {int(kk['p_over_corners_8_5']*100)}% / "
              f"Üst9.5 {int(kk['p_over_corners_9_5']*100)}%)"
              f" | Kart μ≈{kk['mu_cards']:.1f} (Üst3.5 {int(kk['p_over_cards_3_5']*100)}%)")

    wx_txt = f" | {wx}" if wx else ""
    note = (f"Seçim: {pick} | Güven: {conf_pct}% | λ_h/λ_a: {lam_h:.2f}/{lam_a:.2f}"
            f"{wx_txt}{odds_txt}{kk_txt}{form_txt}{table_txt}{streak_txt}")

    return {
        "pick": pick, "confidence": conf_pct,
        "lambda_h": lam_h, "lambda_a": lam_a, "note": note,
        "probs_model": model_probs, "probs_market": market_probs, "probs_blend": blended_probs,
        "wx_adj": wx_adj, "elo_adj": elo_adj, "net_form": net_form
    }

# --- Sonuçlar (gece raporu) + Elo güncelle + Öğrenme -------------------------

def fetch_fd_results(date_str):
    if not FD_TOKEN:
        return []
    url = "https://api.football-data.org/v4/matches"
    headers = {"X-Auth-Token": FD_TOKEN, **HEADERS_JSON}
    data = http_get(url, headers=headers, params={"dateFrom": date_str, "dateTo": date_str})
    out = []
    if data and data.get("matches"):
        for m in data["matches"]:
            if m.get("status") != "FINISHED":
                continue
            comp = m.get("competition",{}) or {}
            area = comp.get("area",{}).get("name","") or "Europe"
            ht = (m.get("homeTeam",{}) or {})
            at = (m.get("awayTeam",{}) or {})
            score = (m.get("score",{}) or {}).get("fullTime",{}) or {}
            dtp = to_dt_utc(m.get("utcDate"))
            out.append({
                "id": m.get("id"),
                "utc_kickoff": dtp,
                "home": ht.get("name"), "away": at.get("name"),
                "home_id": ht.get("id"), "away_id": at.get("id"),
                "area": area, "competition": comp.get("name",""),
                "home_goals": score.get("home"), "away_goals": score.get("away")
            })
    return out

# --- Mail gönderimi ----------------------------------------------------------

def send_mail(subject, body):
    body = (body or "").strip()
    if not body:
        body = "(Bu e-postada içerik üretilemedi / maç bulunamadı.)"
    msg = EmailMessage()
    msg["From"] = GMAIL_USER; msg["To"] = GMAIL_TO; msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(GMAIL_USER, GMAIL_PASS); s.send_message(msg)
    log(f"Mail gönderildi: {subject}")

# --- Raporlar ----------------------------------------------------------------

def report_predictions(date_str):
    fixtures = fetch_fixtures(date_str)
    if not fixtures:
        send_mail(f"Günün Tahminleri | {date_str}", "Bugün için tahmin çıkarılacak maç bulunamadı.")
        return

    try:
        build_form_cache_for_date(fixtures)
    except Exception as e:
        log(f"form build warn: {e}")

    lines = [f"⚽ Günün Tahminleri — {date_str} (FD/OLD + Hava + Elo/Form + Table/Streak + OddsCache + API-Football* + Öğrenme)\n"]
    top = []; hi = []
    fixtures.sort(key=lambda x: x["utc_kickoff"] or datetime.now(timezone.utc))
    for fx in fixtures:
        odds = fetch_odds_avg(fx.get("area",""), fx.get("competition",""), fx["home"], fx["away"])
        rated = rate_fixture(fx, odds)

        # Öğrenme için tüm maçları kaydet (listeye eklemesek de)
        record_prediction(
            fx, rated,
            rated["probs_model"],
            rated["probs_market"],
            rated["probs_blend"],
            rated["wx_adj"], rated["elo_adj"], rated["net_form"]
        )

        if rated["confidence"] < MIN_CONF:
            continue

        ko_local = (fx["utc_kickoff"] or datetime.now(timezone.utc)).astimezone(TR_TZ).strftime("%H:%M")
        line = f"- {ko_local} | {fx.get('area','')} {fx.get('competition','')} | {fx['home']} vs {fx['away']} — {rated['note']}"
        lines.append(line)
        bucket = hi if rated["confidence"] >= HIGH_ALERT else top
        bucket.append((rated["confidence"], line))

    if len(lines) == 1:
        lines.append("Filtreler nedeniyle listelenecek maç kalmadı (MIN_CONF yüksek olabilir).")

    top.sort(reverse=True)
    best = [f"\n⭐ En Güçlü {TOP_N} Seçim:"] + ["  " + l.replace("- ","").strip() for c, l in top[:TOP_N]]

    hi_block = []
    if hi:
        hi.sort(reverse=True)
        hi_block.append("\n⚡ Yüksek Güven Seçimler:")
        for c, l in hi:
            hi_block.append("  " + l.replace("- ","").strip())

    body = "\n".join(lines + [""] + best + hi_block)
    send_mail(f"Günün Tahminleri | {date_str}", body)

    if SPLIT_HIGH and hi:
        body2 = "⚡ Yüksek Güven Eşik Üstü (≥{}%) — {}\n\n".format(HIGH_ALERT, date_str) + "\n".join([l for _, l in hi])
        send_mail(f"⚡ Yüksek Güven | {date_str}", body2)

def report_results(date_str):
    res = fetch_fd_results(date_str)
    if not res:
        send_mail(f"Günün Sonuçları | {date_str}", "Bugün için maç bulunamadı.")
        return

    lines = [f"📊 Günün Sonuçları — {date_str}", ""]
    for r in res:
        h,a = r["home_goals"], r["away_goals"]
        score = f"{r['home']} {h}-{a} {r['away']}"
        lines.append(f"- {score} | {r['area']} {r['competition']}")
        try:
            if h is not None and a is not None:
                if h > a:    res_hw = 1.0
                elif h == a: res_hw = 0.5
                else:        res_hw = 0.0
                elo_update(r["area"], r["home"], r["away"], res_hw)
        except Exception as e:
            log(f"Elo update warn: {e}")

    # Otomatik öğrenme (w_mkt ve goal_scale)
    learn_summary = ""
    try:
        learn_summary = analyze_and_learn(res)
        if learn_summary:
            lines += ["", "🔧 " + learn_summary]
    except Exception as e:
        log(f"learn err: {e}")

    # State kaydet
    try:
        _state_save(STATE)
    except Exception as e:
        log(f"state save warn: {e}")

    body = "\n".join(lines)
    send_mail(f"Günün Sonuçları | {date_str}", body)

# --- Çalıştırıcı -------------------------------------------------------------

def main():
    try:
        now_utc = datetime.now(timezone.utc)
        tr_now  = now_utc.astimezone(TR_TZ)
        date_str = tr_now.strftime("%Y-%m-%d")

        mode = MODE_ENV
        if mode == "AUTO":
            mode = "PREDICT" if now_utc.hour == 7 else "RESULTS"

        log(f"MODE={mode} | TR now={tr_now} | date={date_str} | w_mkt={get_w_mkt():.2f}")
        if mode == "PREDICT":
            report_predictions(date_str)
        elif mode == "RESULTS":
            report_results(date_str)
        else:
            send_mail("Tahmin Botu | Bilgi",
                      "AUTO modu dışı çalıştırma. MODE=PREDICT veya MODE=RESULTS bekleniyor.")
    except Exception:
        tb = traceback.format_exc(); log(tb)
        try: send_mail("Tahmin Botu | Hata", tb)
        except Exception: pass

if __name__ == "__main__":
    main()
