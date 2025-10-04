# -*- coding: utf-8 -*-
"""
Tahmin Botu — tek parça mailer.py
Ücretsiz kaynaklar:
- football-data.org (Fixtures/sonuçlar)  -> X-Auth-Token: FOOTBALL_DATA_TOKEN
- OpenLigaDB (ek lig fallback'i)         -> anahtar gerekmez
- Open-Meteo (hava)                      -> anahtar gerekmez
- The Odds API (opsiyonel oranlar)       -> ODDS_API_KEY varsa kullanılır

Modlar:
- MODE=PREDICT  -> 10:00 TR “Günün Tahminleri”
- MODE=RESULTS  -> 23:59 TR “Günün Sonuçları”
- MODE=AUTO     -> Saat 07 UTC ise PREDICT, aksi ise RESULTS

Gerekli Secrets: GMAIL_USER, GMAIL_PASS, GMAIL_TO, FOOTBALL_DATA_TOKEN
Opsiyonel Secrets: ODDS_API_KEY
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

def today_str_tz(tz=TR_TZ):
    return datetime.now(tz).strftime("%Y-%m-%d")

def to_dt_utc(s):
    try:
        # football-data ISO 8601
        return datetime.fromisoformat(s.replace("Z", "+00:00")).astimezone(timezone.utc)
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
MODE_ENV   = (os.getenv("MODE") or "AUTO").upper().strip()

if not (GMAIL_USER and GMAIL_PASS and GMAIL_TO):
    raise SystemExit("GMAIL_USER/GMAIL_PASS/GMAIL_TO secrets eksik.")

# --- Odds sport-key haritalaması (geniş) ------------------------------------

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
    return team_name  # fallback: ismi doğrudan geocode dene

def fetch_weather_note(home_team):
    # Open-Meteo geocoding -> forecast summary
    city = guess_city_from_team(home_team)
    geo = http_get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1, "language": "en"})
    if not geo or not geo.get("results"): 
        return None
    lat = geo["results"][0]["latitude"]
    lon = geo["results"][0]["longitude"]
    wx = http_get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat, "longitude": lon,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "timezone": "auto"
    })
    if not wx: 
        return None
    # kaba özet (yakındaki saat ortalaması)
    try:
        temps = wx["hourly"]["temperature_2m"][:6]
        prec  = wx["hourly"]["precipitation"][:6]
        wind  = wx["hourly"]["wind_speed_10m"][:6]
        tavg  = sum(temps)/len(temps)
        pavg  = sum(prec)/len(prec)
        wavg  = sum(wind)/len(wind)
        return f"Hava: {tavg:.0f}°C, yağış {pavg:.1f}mm, rüzgâr {wavg:.0f} km/s"
    except Exception:
        return None

# --- Football-Data (ana kaynak) + OpenLigaDB (fallback) ----------------------

def fetch_fd_fixtures(date_str):
    """football-data.org: sadece bugünün planlı maçları"""
    if not FD_TOKEN:
        return []
    url = "https://api.football-data.org/v4/matches"
    headers = {"X-Auth-Token": FD_TOKEN, **HEADERS_JSON}
    data = http_get(url, headers=headers, params={"dateFrom": date_str, "dateTo": date_str})
    out = []
    if data and data.get("matches"):
        for m in data["matches"]:
            if m.get("status") not in ("SCHEDULED", "TIMED"):
                continue
            comp = m.get("competition", {})
            area = comp.get("area", {}).get("name", "")
            out.append({
                "source": "FD",
                "utc_kickoff": to_dt_utc(m.get("utcDate")),
                "home": m.get("homeTeam", {}).get("name"),
                "away": m.get("awayTeam", {}).get("name"),
                "area": area,
                "competition": comp.get("name", ""),
                "id": m.get("id"),
            })
    return out

def fetch_openligadb_day(date_str):
    """OpenLigaDB: bazı popüler ligler için günlük tarama (yaklaşık)."""
    y, m, d = date_str.split("-")
    leagues = [
        ("bundesliga",     int(y)),  # 1. Bundesliga
        ("bundesliga2",    int(y)),  # 2. Bundesliga
    ]
    out = []
    for lg, season in leagues:
        url = f"https://www.openligadb.de/api/getmatchdata/{lg}/{season}"
        data = http_get(url)
        if not data: 
            continue
        for m in data:
            dt = m.get("MatchDateTimeUTC") or m.get("MatchDateTime")
            dtp = to_dt_utc(dt) if dt else None
            if not dtp: 
                continue
            ds = dtp.astimezone(TR_TZ).strftime("%Y-%m-%d")
            if ds != date_str: 
                continue
            comp = m.get("LeagueName") or lg
            out.append({
                "source":"OLD",
                "utc_kickoff": dtp,
                "home": m.get("Team1",{}).get("TeamName"),
                "away": m.get("Team2",{}).get("TeamName"),
                "area": "Germany",
                "competition": comp,
                "id": f"old:{m.get('MatchID')}",
            })
    return out

def fetch_fixtures(date_str):
    fixtures = fetch_fd_fixtures(date_str)
    if not fixtures:
        fixtures = fetch_openligadb_day(date_str)
    return fixtures

# --- Oranlar (opsiyonel) -----------------------------------------------------

def fetch_odds_avg(area, comp, home, away):
    if not ODDS_KEY:
        return None
    skey = guess_sport_key(area, comp)
    if not skey:
        return None
    url = "https://api.the-odds-api.com/v4/sports/{}/odds/".format(skey)
    params = {
        "regions": "eu,uk",
        "markets": "h2h",
        "oddsFormat": "decimal",
        "apiKey": ODDS_KEY,
    }
    data = http_get(url, params=params)
    if not data or not isinstance(data, list):
        return None
    # Çok basit benzerlik eşleşmesi
    def norm(x): return (x or "").lower().replace(".", "").replace("-", " ").replace(" fc","").strip()
    h, a = norm(home), norm(away)
    best = None
    for ev in data:
        comps = ev.get("bookmakers", [])
        if not comps: 
            continue
        # takımlar
        t1 = norm(ev.get("home_team"))
        t2 = norm(ev.get("away_team"))
        if (t1.startswith(h) and t2.startswith(a)) or (t1 in h and t2 in a):
            # tüm book'larda ortalama
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
    # marjdan arındırılmış piyasa olasılıkları
    inv = (1/o1) + (1/ox) + (1/o2)
    p1, px, p2 = (1/o1)/inv, (1/ox)/inv, (1/o2)/inv
    return {"odds": (o1, ox, o2), "probs": (p1, px, p2)}

# --- Basit Poisson + "SPI proxy" --------------------------------------------

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
    # basit 0..10 olasılık matrisi
    def pois(m, lam): return (lam**m) * math.exp(-lam) / math.factorial(m)
    p_home = p_draw = p_away = 0.0
    for gh in range(0, 11):
        ph = pois(gh, lambda_h)
        for ga in range(0, 11):
            pa = pois(ga, lambda_a)
            if gh > ga:  p_home += ph*pa
            elif gh == ga: p_draw += ph*pa
            else:        p_away += ph*pa
    return p_home, p_draw, p_away

def blend_model_market(model_probs, market_probs):
    if not market_probs:
        return model_probs
    w_mkt = 0.35  # piyasa ağırlığı (ücretsiz katmanda hafif)
    return tuple((1-w_mkt)*m + w_mkt*mk for m, mk in zip(model_probs, market_probs))

def rate_fixture(fix, odds_info):
    area = fix["area"] or "Europe"
    tot = base_total_goals(area)
    # SPI proxy: ev sahibi için hafif ev-saha + isim uzunluğu gibi çok hafif fark (örüntüsel sabit)
    ah = 1.12
    noise = (len((fix["home"] or "")) - len((fix["away"] or ""))) * 0.01
    lam_h = max(0.2, tot*0.5*ah + noise)
    lam_a = max(0.2, tot*0.5*(2-ah) - noise)

    # Hava düzeltmesi (rüzgâr/yağış arttıkça toplam golü biraz düşür)
    wx = fetch_weather_note(fix["home"])
    if wx and "rüzgâr" in wx and "yağış" in wx:
        try:
            wind = safe_float(wx.split("rüzgâr")[1].split("km/s")[0].strip().split()[-1])
            precip = safe_float(wx.split("yağış")[1].split("mm")[0].strip().split()[-1])
            adj = 1.0 - min(0.15, 0.003*wind + 0.01*precip)
            lam_h *= adj; lam_a *= adj
        except Exception:
            pass

    # Model olasılıkları
    m_home, m_draw, m_away = poisson_prob(lam_h, lam_a)

    market_probs = None
    odds_txt = ""
    if odds_info:
        o1,oX,o2 = odds_info["odds"]
        p1,px,p2 = odds_info["probs"]
        market_probs = (p1,px,p2)
        odds_txt = f" | Odds(avg) 1/X/2: {o1:.2f}/{oX:.2f}/{o2:.2f}"

    p_home, p_draw, p_away = blend_model_market((m_home,m_draw,m_away), market_probs)

    # En yüksek olasılığı “seçim” yap
    picks = [("1", p_home), ("X", p_draw), ("2", p_away)]
    picks.sort(key=lambda x: x[1], reverse=True)
    pick, conf = picks[0]
    conf_pct = int(round(conf*100))

    wx_txt = f" | {wx}" if wx else ""
    note = f"Seçim: {pick} | Güven: {conf_pct}% | λ_h/λ_a: {lam_h:.2f}/{lam_a:.2f}{wx_txt}{odds_txt}"

    return {
        "pick": pick, "confidence": conf_pct,
        "lambda_h": lam_h, "lambda_a": lam_a,
        "note": note
    }

# --- Sonuçlar (gece raporu) --------------------------------------------------

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
            home = m.get("homeTeam",{}).get("name")
            away = m.get("awayTeam",{}).get("name")
            comp = m.get("competition",{}).get("name","")
            area = m.get("competition",{}).get("area",{}).get("name","")
            score = m.get("score",{})
            full = score.get("fullTime",{})
            out.append({
                "home":home,"away":away,
                "area":area,"competition":comp,
                "home_goals": full.get("home"),
                "away_goals": full.get("away")
            })
    return out

# --- Mail gönderimi ----------------------------------------------------------

def send_mail(subject, body):
    msg = EmailMessage()
    msg["From"] = GMAIL_USER
    msg["To"] = GMAIL_TO
    msg["Subject"] = subject
    msg.set_content(body)

    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(GMAIL_USER, GMAIL_PASS)
        s.send_message(msg)
    log(f"Mail gönderildi: {subject}")

# --- Raporlar ----------------------------------------------------------------

def report_predictions(date_str):
    fixtures = fetch_fixtures(date_str)
    if not fixtures:
        send_mail(f"Günün Tahminleri | {date_str}", "Bugün için tahmin çıkarılacak maç bulunamadı.")
        return

    lines = []
    top = []
    lines.append(f"⚽ Günün Tahminleri — {date_str} (OpenLigaDB + SPI + Hava + Odds)\n")
    fixtures.sort(key=lambda x: x["utc_kickoff"] or datetime.now(timezone.utc))
    for fx in fixtures:
        odds = fetch_odds_avg(fx.get("area",""), fx.get("competition",""), fx["home"], fx["away"])
        rated = rate_fixture(fx, odds)
        ko_local = (fx["utc_kickoff"] or datetime.now(timezone.utc)).astimezone(TR_TZ).strftime("%H:%M")
        line = f"- {ko_local} | {fx.get('area','')} {fx.get('competition','')} | {fx['home']} vs {fx['away']} — {rated['note']}"
        lines.append(line)
        top.append((rated["confidence"], line))

    # En güçlü 5 seçim
    top.sort(reverse=True)
    best = ["\n⭐ En Güçlü 5 Seçim:"]
    for c, l in top[:5]:
        best.append("  " + l.replace("- ","").strip())

    body = "\n".join(lines + [""] + best)
    send_mail(f"Günün Tahminleri | {date_str}", body)

def report_results(date_str):
    res = fetch_fd_results(date_str)
    if not res:
        send_mail(f"Günün Sonuçları | {date_str}", "Bugün için maç bulunamadı.")
        return
    lines = [f"📊 Günün Sonuçları — {date_str}", ""]
    ok = err = 0
    for r in res:
        h,a = r["home_goals"], r["away_goals"]
        score = f"{r['home']} {h}-{a} {r['away']}"
        lines.append(f"- {score} | {r['area']} {r['competition']}")
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
            # 07:xx UTC -> sabah tahmin, aksi gece sonuç
            if now_utc.hour == 7:
                mode = "PREDICT"
            else:
                mode = "RESULTS"

        log(f"MODE={mode} | TR now={tr_now} | date={date_str}")

        if mode == "PREDICT":
            report_predictions(date_str)
        elif mode == "RESULTS":
            report_results(date_str)
        else:
            send_mail("Tahmin Botu | Bilgi", "AUTO modu dışı çalıştırma. MODE=PREDICT veya MODE=RESULTS bekleniyor.")
    except Exception as e:
        tb = traceback.format_exc()
        log(tb)
        try:
            send_mail("Tahmin Botu | Hata", tb)
        except Exception:
            pass

if __name__ == "__main__":
    main()
