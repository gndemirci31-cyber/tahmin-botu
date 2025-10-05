# -*- coding: utf-8 -*-
"""
Tahmin Botu — tek parça mailer.py (GÜNCEL + Elo + OppAdj Form + State)
Ücretsiz kaynaklar:
- football-data.org (Fixtures/sonuçlar)  -> X-Auth-Token: FOOTBALL_DATA_TOKEN
- OpenLigaDB (fallback)                  -> anahtar gerekmez
- Open-Meteo (hava)                      -> anahtar gerekmez
- The Odds API (opsiyonel oranlar)       -> ODDS_API_KEY varsa kullanılır

Modlar:
- MODE=PREDICT  -> 10:00 TR “Günün Tahminleri”
- MODE=RESULTS  -> 23:59 TR “Günün Sonuçları”
- MODE=AUTO     -> Saat 07 UTC ise PREDICT, aksi ise RESULTS

Zorunlu Secrets: GMAIL_USER, GMAIL_PASS, GMAIL_TO
Önerilen Secrets: FOOTBALL_DATA_TOKEN
Opsiyonel Secrets: ODDS_API_KEY, APIFOOTBALL_KEY (şimdilik kullanılmıyor)

Opsiyonel ENV (varsayılanlar):
- TOP_N=5, MIN_CONF=0, HIGH_ALERT=90
- OLD_LEAGUES="bundesliga,bundesliga2"
- ODDS_TTL_MIN=15
- ELO_K=24, ELO_HOME_ADV=60
- FORM_LOOKBACK=10, FORM_DAYS=120
- ALLOW_STATE_FILE=1
"""

import os, math, time, json, smtplib, traceback
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage

import requests

# --- Ortak yardımcılar -------------------------------------------------------

TR_TZ = timezone(timedelta(hours=3))  # Türkiye
HEADERS_JSON = {"Accept": "application/json"}

def log(msg): print(f"[mailer] {msg}", flush=True)

def http_get(url, headers=None, params=None, timeout=20):
    try:
        r = requests.get(url, headers=headers or {}, params=params or {}, timeout=timeout)
        if r.status_code == 200:
            return r.json()
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

def safe_float(x, default=0.0):
    try: return float(x)
    except: return default

# --- Secrets / ortam ---------------------------------------------------------

GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_PASS")
GMAIL_TO   = os.getenv("GMAIL_TO")
FD_TOKEN   = os.getenv("FOOTBALL_DATA_TOKEN")
ODDS_KEY   = os.getenv("ODDS_API_KEY")
APIFOOT    = os.getenv("APIFOOTBALL_KEY", "").strip()  # ileride kullanacağız
MODE_ENV   = (os.getenv("MODE") or "AUTO").upper().strip()

TOP_N        = int(os.getenv("TOP_N", "5"))
MIN_CONF     = int(os.getenv("MIN_CONF", "0"))
HIGH_ALERT   = int(os.getenv("HIGH_ALERT", "90"))
OLD_LEAGUES  = [x.strip() for x in (os.getenv("OLD_LEAGUES", "bundesliga,bundesliga2").split(",")) if x.strip()]
ODDS_TTL_MIN = int(os.getenv("ODDS_TTL_MIN", "15"))

# Elo / Form ayarları
ELO_K          = float(os.getenv("ELO_K", "24"))
ELO_HOME_ADV   = float(os.getenv("ELO_HOME_ADV", "60"))   # puan
FORM_LOOKBACK  = int(os.getenv("FORM_LOOKBACK", "10"))    # son N maç
FORM_DAYS      = int(os.getenv("FORM_DAYS", "120"))       # geçmiş gün penceresi
ALLOW_STATE_FILE = os.getenv("ALLOW_STATE_FILE", "1") == "1"

if not (GMAIL_USER and GMAIL_PASS and GMAIL_TO):
    raise SystemExit("GMAIL_USER/GMAIL_PASS/GMAIL_TO secrets eksik.")

# --- State (kalibrasyon/Elo) -------------------------------------------------

STATE_PATH = "model_state.json"

def _state_load():
    try:
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log(f"state load err: {e}")
    return {"elo": {}, "last_saved": None}

def _state_save(st):
    if not ALLOW_STATE_FILE:
        log("STATE kaydı kapalı (ALLOW_STATE_FILE=0)")
        return
    try:
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=2)
        st["last_saved"] = datetime.utcnow().isoformat() + "Z"
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
    # 1/(1+10^(-diff/400))
    diff = (elo_a + home_adv) - elo_b
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

def elo_update(area, home_name, away_name, result_hw):
    # result_hw: 1=home win, 0.5=draw, 0=away win
    Eh = elo_get(area, home_name)
    Ea = elo_get(area, away_name)
    ph = elo_expect(Eh, Ea, ELO_HOME_ADV)
    pa = 1.0 - ph
    # home
    Eh_new = Eh + ELO_K * (result_hw - ph)
    # away
    Ea_new = Ea + ELO_K * ((1.0 - result_hw) - pa)
    elo_set(area, home_name, Eh_new)
    elo_set(area, away_name, Ea_new)

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
    geo = http_get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1, "language": "en"})
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

# --- Football-Data (ana kaynak) + OpenLigaDB (fallback) ----------------------

def fetch_fd_fixtures(date_str):
    """
    FD: UTC ±1 gün penceresiyle çek, TR gününe filtrele.
    """
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
            comp = m.get("competition", {})
            area = comp.get("area", {}).get("name", "")
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
                "id": f"old:{mm.get('MatchID')}",
            })
    log(f"OpenLigaDB fixtures (TR={date_str}) -> {len(out)}")
    return out

# --- 3. Fallback: The Odds API'den fikstür listesi --------------------------

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
        url = f"https://api.the-odds-api.com/v4/sports/{skey}/odds/"
        data = http_get(url, params={
            "regions": "eu,uk",
            "markets": "h2h",
            "oddsFormat": "decimal",
            "apiKey": ODDS_KEY,
        })
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

# --- Oranlar: cache'li -------------------------------------------------------

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
    for k, v in LEAGUE_GOAL_BASE.items():
        if k.lower() in (area or "").lower():
            return v
    return 2.60

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
    start = int(line_int) + 1
    s = 0.0
    m = start
    while m <= start + 40:
        s += (lam**m) * math.exp(-lam) / math.factorial(m)
        m += 1
    return min(max(s,0.0),1.0)

def blend_model_market(model_probs, market_probs):
    if not market_probs:
        return model_probs
    w_mkt = 0.35
    return tuple((1-w_mkt)*m + w_mkt*mk for m, mk in zip(model_probs, market_probs))

# --- Kart / Korner (model tabanlı) -------------------------------------------

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

def model_cards_corners(area, lam_h, lam_a, wx_text):
    cards_base  = base_from_area(area, LEAGUE_CARD_BASE, 4.6)
    corner_base = base_from_area(area, LEAGUE_CORNER_BASE, 9.2)

    wind, precip = parse_weather(wx_text)
    tempo_factor = (lam_h + lam_a) / 2.6
    tempo_factor = max(0.7, min(1.4, tempo_factor))

    cards   = cards_base
    corners = corner_base * tempo_factor

    if precip is not None:
        cards   *= 1.00 + min(0.10, 0.02  * precip)
        corners *= 1.00 - min(0.10, 0.015 * precip)
    if wind is not None:
        corners *= 1.00 + min(0.08, 0.003 * wind)
        cards   *= 1.00 + min(0.05, 0.002 * wind)

    p_corners_8_5 = poisson_over_prob(corners, 8.5)
    p_corners_9_5 = poisson_over_prob(corners, 9.5)
    p_cards_3_5   = poisson_over_prob(cards,   3.5)
    p_cards_4_5   = poisson_over_prob(cards,   4.5)

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
    # tarihe göre sırala (yeniden eskiye)
    ms = []
    for m in data["matches"]:
        dtp = to_dt_utc(m.get("utcDate"))
        if not dtp: 
            continue
        ms.append((dtp, m))
    ms.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in ms]

_FORM_CACHE = {}  # team_id -> form_adj,info_text

def _form_adjust_from_matches(team_id, area, team_name):
    # cache
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
        gh, ga = safe_float(score.get("home")), safe_float(score.get("away"))
        if gh is None or ga is None:
            continue
        is_home = (ht == team_name)
        gf = gh if is_home else ga
        ga_ = ga if is_home else gh
        opp_name = at if is_home else ht
        opp_elo = elo_get(area, opp_name)
        # rakip güçlü ise (Elo yüksek) pozitif gol farkına daha fazla ağırlık
        w = 1.0 + max(-0.2, min(0.2, (opp_elo - 1500.0)/1000.0))
        score_sum += (gf - ga_) * w
        n += 1
        if n >= FORM_LOOKBACK:
            break

    if n == 0:
        adj = 0.0
    else:
        avg = score_sum / n
        # yumuşak sıkıştırma
        adj = max(-0.15, min(0.15, math.tanh(avg / 3.0) * 0.12))

    txt = f"FormAdj {('+' if adj>=0 else '')}{int(adj*100)}%"
    _FORM_CACHE[team_id] = (adj, txt)
    return (adj, txt)

def build_form_cache_for_date(fixtures):
    # FD kaynaklı fikstürlerde team_id var; onlar için önceden cache oluştur.
    seen = set()
    for fx in fixtures:
        for tid in (fx.get("home_id"), fx.get("away_id")):
            if tid and tid not in seen:
                seen.add(tid)
                try:
                    _form_adjust_from_matches(tid, fx.get("area","Europe"), fx.get("home") if tid==fx.get("home_id") else fx.get("away"))
                    time.sleep(0.05)  # nazik ol
                except Exception as e:
                    log(f"form cache err team_id={tid}: {e}")

# --- Derecelendirme ----------------------------------------------------------

def rate_fixture(fix, odds_info):
    area = fix["area"] or "Europe"
    tot  = base_total_goals(area)

    # SPI proxy: ev etkisi (sabit) + minik isim-uzunluğu farkı
    ah    = 1.12
    noise = (len((fix["home"] or "")) - len((fix["away"] or ""))) * 0.01
    lam_h = max(0.2, tot*0.5*ah        + noise)
    lam_a = max(0.2, tot*0.5*(2 - ah)  - noise)

    # Hava
    wx = fetch_weather_note(fix["home"])
    if wx:
        wind, precip = parse_weather(wx)
        try:
            adj = 1.0
            if wind   is not None: adj -= min(0.08, 0.003 * wind)
            if precip is not None: adj -= min(0.08, 0.01  * precip)
            adj = max(0.8, adj)
            lam_h *= adj; lam_a *= adj
        except Exception:
            pass

    # Elo etkisi (lambda'ya küçük çarpan)
    Eh = elo_get(area, fix["home"]); Ea = elo_get(area, fix["away"])
    elo_diff = (Eh + ELO_HOME_ADV) - Ea
    elo_adj  = max(-0.20, min(0.20, (elo_diff/400.0)*0.15))  # ±%20 sınır
    lam_h *= (1.0 + elo_adj); lam_a *= (1.0 - elo_adj)

    # Opponent-adjusted form (yalnızca FD'de team_id varsa güçlü)
    form_txt = ""
    home_adj = away_adj = 0.0
    if fix.get("home_id"):
        home_adj, t = _form_adjust_from_matches(fix["home_id"], area, fix["home"]);  form_txt += " | " + t
    if fix.get("away_id"):
        away_adj, t = _form_adjust_from_matches(fix["away_id"], area, fix["away"]);  form_txt += " / " + t
    net_form = max(-0.18, min(0.18, home_adj - away_adj))
    lam_h *= (1.0 + net_form); lam_a *= (1.0 - net_form)

    # Model 1X2
    m_home, m_draw, m_away = poisson_prob(lam_h, lam_a)

    market_probs = None
    odds_txt = ""
    if odds_info:
        o1,oX,o2 = odds_info["odds"]
        p1,px,p2 = odds_info["probs"]
        market_probs = (p1,px,p2)
        odds_txt = f" | Odds(avg) 1/X/2: {o1:.2f}/{oX:.2f}/{o2:.2f}"

    p_home, p_draw, p_away = blend_model_market((m_home,m_draw,m_away), market_probs)

    picks = [("1", p_home), ("X", p_draw), ("2", p_away)]
    picks.sort(key=lambda x: x[1], reverse=True)
    pick, conf = picks[0]; conf_pct = int(round(conf*100))

    # Kart/Korner
    kk = model_cards_corners(area, lam_h, lam_a, wx)
    kk_txt = (f" | Korner μ≈{kk['mu_corners']:.1f} (Üst8.5 {int(kk['p_over_corners_8_5']*100)}% / "
              f"Üst9.5 {int(kk['p_over_corners_9_5']*100)}%)"
              f" | Kart μ≈{kk['mu_cards']:.1f} (Üst3.5 {int(kk['p_over_cards_3_5']*100)}%)")

    wx_txt = f" | {wx}" if wx else ""
    note = (f"Seçim: {pick} | Güven: {conf_pct}% | λ_h/λ_a: {lam_h:.2f}/{lam_a:.2f}"
            f"{wx_txt}{odds_txt}{kk_txt}{form_txt}")

    return {"pick": pick, "confidence": conf_pct, "lambda_h": lam_h, "lambda_a": lam_a, "note": note}

# --- Sonuçlar (gece raporu) + Elo güncelle ----------------------------------

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
            out.append({
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

    # Opp-adjusted form için öncache (FD olanlar)
    try:
        build_form_cache_for_date(fixtures)
    except Exception as e:
        log(f"form build warn: {e}")

    lines = [f"⚽ Günün Tahminleri — {date_str} (FD/OLD + Hava + Elo/Form + OddsCache)\n"]
    top = []
    fixtures.sort(key=lambda x: x["utc_kickoff"] or datetime.now(timezone.utc))
    for fx in fixtures:
        odds = fetch_odds_avg(fx.get("area",""), fx.get("competition",""), fx["home"], fx["away"])
        rated = rate_fixture(fx, odds)
        if rated["confidence"] < MIN_CONF:
            continue
        ko_local = (fx["utc_kickoff"] or datetime.now(timezone.utc)).astimezone(TR_TZ).strftime("%H:%M")
        line = f"- {ko_local} | {fx.get('area','')} {fx.get('competition','')} | {fx['home']} vs {fx['away']} — {rated['note']}"
        lines.append(line); top.append((rated["confidence"], line))

    if len(lines) == 1:
        lines.append("Filtreler nedeniyle listelenecek maç kalmadı (MIN_CONF yüksek olabilir).")

    top.sort(reverse=True)
    best = [f"\n⭐ En Güçlü {TOP_N} Seçim:"] + ["  " + l.replace("- ","").strip() for c, l in top[:TOP_N]]

    hi = [x for x in top if x[0] >= HIGH_ALERT]
    hi_block = []
    if hi:
        hi_block.append("\n⚡ Yüksek Güven Seçimler:")
        for c, l in hi:
            hi_block.append("  " + l.replace("- ","").strip())

    body = "\n".join(lines + [""] + best + hi_block)
    send_mail(f"Günün Tahminleri | {date_str}", body)

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

        # Elo güncelle
        try:
            if h is not None and a is not None:
                if h > a:    res_hw = 1.0
                elif h == a: res_hw = 0.5
                else:        res_hw = 0.0
                elo_update(r["area"], r["home"], r["away"], res_hw)
        except Exception as e:
            log(f"Elo update warn: {e}")

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

        log(f"MODE={mode} | TR now={tr_now} | date={date_str}")
        if mode == "PREDICT":
            report_predictions(date_str)
        elif mode == "RESULTS":
            report_results(date_str)
        else:
            send_mail("Tahmin Botu | Bilgi", "AUTO modu dışı çalıştırma. MODE=PREDICT veya MODE=RESULTS bekleniyor.")
    except Exception:
        tb = traceback.format_exc(); log(tb)
        try: send_mail("Tahmin Botu | Hata", tb)
        except Exception: pass

if __name__ == "__main__":
    main()
