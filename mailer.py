# mailer.py — OpenLigaDB + SPI + Hava (Open-Meteo) + Odds (The Odds API)
# Tam entegre tek dosya. GitHub Actions:
#   python mailer.py predict  -> Sabah tahmin maili
#   python mailer.py results  -> Gece sonuç maili

import os, sys, json, math, time, io, csv, unicodedata
from datetime import datetime, timedelta, timezone
import smtplib, ssl
import requests

# =================== ZAMAN / GENEL ===========================================
TR_TZ = timezone(timedelta(hours=3))  # Europe/Istanbul
TODAY = datetime.now(TR_TZ).date()

def now_utc():
    return datetime.now(timezone.utc)

# =================== E-POSTA =================================================
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_PASS")
GMAIL_TO   = os.getenv("GMAIL_TO", GMAIL_USER)

def send_email(subject: str, body: str):
    assert GMAIL_USER and GMAIL_PASS and GMAIL_TO, "Gmail env eksik (GMAIL_USER/PASS/TO)."
    msg = f"From: {GMAIL_USER}\r\nTo: {GMAIL_TO}\r\nSubject: {subject}\r\n\r\n{body}"
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as s:
        s.login(GMAIL_USER, GMAIL_PASS)
        s.sendmail(GMAIL_USER, [GMAIL_TO], msg.encode("utf-8"))

# =================== YARDIMCI / MATH =========================================
def poisson_pmf(lmb, k): return math.exp(-lmb) * (lmb**k) / math.factorial(k)
def sigmoid(z): return 1/(1+math.exp(-z))
def logit(p):
    p = min(max(p,1e-6),1-1e-6)
    return math.log(p/(1-p))

def _norm_name(s: str) -> str:
    import re
    if not s: return ""
    s = unicodedata.normalize("NFKD", s).encode("ascii","ignore").decode("ascii")
    s = s.lower()
    toks = [t for t in re.split(r"[^a-z0-9]+", s) if t]
    drop = {"fc","cf","sc","ac","afc","jk","sk","fk","spor","kulubu","club","athletic",
            "atletico","calcio","deportivo","sporting","uniao","foot","football","futbol",
            "sv","sp","bk"}
    toks = [t for t in toks if t not in drop]
    return "".join(toks)

# =================== MODEL DURUMU (kalibrasyon) ==============================
STATE_PATH = "model_state.json"
def _default_state():
    return {
        # w1=model (Poisson), w2=market(odds), w3=history(rezerv), w4=SPI
        "blend": {"w1": 0.55, "w2": 0.20, "w3": 0.05, "w4": 0.20},
        "leagues": {}
    }
def load_state():
    try:
        if os.path.exists(STATE_PATH):
            return json.loads(open(STATE_PATH,"r",encoding="utf-8").read())
    except Exception: pass
    return _default_state()
def save_state(st):
    try:
        open(STATE_PATH,"w",encoding="utf-8").write(json.dumps(st,ensure_ascii=False,indent=2))
    except Exception:
        pass

# =================== SPI (FiveThirtyEight) ===================================
SPI_URL = "https://projects.fivethirtyeight.com/soccer-api/club/spi_global_rankings.csv"
_spi_cache = None
def fetch_spi_table() -> dict:
    global _spi_cache
    if _spi_cache is not None: return _spi_cache
    try:
        r = requests.get(SPI_URL, timeout=35)
        r.raise_for_status()
        d={}
        reader = csv.DictReader(io.StringIO(r.text))
        for row in reader:
            key = _norm_name(row.get("name") or row.get("team") or "")
            if not key: continue
            try:
                d[key]={
                    "spi": float(row.get("spi") or 0.0),
                    "off": float(row.get("off") or 1.0),
                    "def": float(row.get("def") or 1.0),
                    "league": row.get("league",""),
                }
            except: continue
        _spi_cache = d
    except Exception:
        d={}
    return _spi_cache

def spi_outcomes(home_name: str, away_name: str, k: float=10.0):
    tab = fetch_spi_table()
    if not tab: return None
    h, a = tab.get(_norm_name(home_name)), tab.get(_norm_name(away_name))
    if not h or not a: return None
    delta = (h["spi"] - a["spi"])
    p_home_raw = 1.0/(1.0+math.exp(-delta/k))
    closeness = math.exp(-abs(delta)/12.0)
    p_draw = max(0.18, min(0.35, 0.22 + 0.18*closeness))
    scale = 1.0 - p_draw
    return {"1": p_home_raw*scale, "X": p_draw, "2": (1-p_home_raw)*scale,
            "_delta": delta, "_h":h, "_a":a}

# =================== HAV A (Open-Meteo) ======================================
# Ücretsiz / API key gerekmez. Geocoding ile şehir→koordinat bulup
# maç saatindeki (UTC) temperature_2m, precipitation, wind_speed_10m çekiyoruz.
_geo_cache = {}  # bellekte
def geocode_city(name: str):
    key = _norm_name(name)
    if key in _geo_cache: return _geo_cache[key]
    try:
        r = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": name, "count": 1, "language": "en", "format": "json"},
            timeout=20
        )
        r.raise_for_status()
        j = r.json()
        if (j.get("results")):
            m = j["results"][0]
            lat, lon = m["latitude"], m["longitude"]
            _geo_cache[key] = (lat, lon, m.get("name",""))
            return _geo_cache[key]
    except Exception:
        pass
    _geo_cache[key] = None
    return None

def guess_city_from_team(team_name: str):
    # Basit yaklaşım: takım adındaki tekil şehir kelimesini dene, yoksa None
    # (örn. "Galatasaray" → "Istanbul", "Bayern München" → "Munich")
    # Burayı minimal bırakıyoruz; isim → şehir eşleşmesi çok garanti değil.
    # İlk aşamada direkt takım adını geocode etmeyi dene:
    return team_name

def fetch_weather_for_match(team_home: str, utc_iso: str):
    try:
        city = guess_city_from_team(team_home)
        coords = geocode_city(city)
        if not coords: 
            return None
        lat, lon, label = coords
        dt = datetime.fromisoformat(utc_iso.replace("Z","")).replace(tzinfo=timezone.utc)
        start = dt.date().isoformat()
        end   = start
        r = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "hourly": "temperature_2m,precipitation,wind_speed_10m",
                "start_date": start, "end_date": end,
                "timezone": "UTC"
            },
            timeout=30
        )
        r.raise_for_status()
        j = r.json()
        H = j.get("hourly", {})
        times = H.get("time", [])
        if not times: return None
        # En yakın saat indeksini bul
        target = dt.replace(minute=0, second=0, microsecond=0)
        idx = None
        for i,t in enumerate(times):
            if t.startswith(target.strftime("%Y-%m-%dT%H:00")):
                idx = i; break
        if idx is None:
            # fallback: ilk saat
            idx = 0
        return {
            "city": label or city,
            "temp": float(H.get("temperature_2m",[None])[idx]),
            "wind": float(H.get("wind_speed_10m",[None])[idx]),
            "prec": float(H.get("precipitation",[None])[idx])
        }
    except Exception:
        return None

def weather_adjustments(w):
    """
    Hava etkisini 3 metrikten üret:
      - Rüzgâr (m/s) yüksek → gol & korner ↓
      - Yağış (mm/h) yüksek → gol ↓, kart ↑ hafif
      - Aşırı sıcak/soğuk → tempo ↓
    Dönen: dict with goal_mu_factor, card_bias, corner_bias, note
    """
    if not w:
        return {"goal_mu_factor": 1.00, "card_bias": 1.00, "corner_bias": 1.00, "note": ""}
    wind = w["wind"] or 0.0
    prec = w["prec"] or 0.0
    temp = w["temp"] or 15.0

    mu = 1.00
    card = 1.00
    corner = 1.00
    reasons=[]

    # Rüzgâr
    if wind >= 12:   mu *= 0.90; corner *= 0.92; reasons.append(f"rüzgâr {wind:.0f} m/s")
    elif wind >= 8:  mu *= 0.95; corner *= 0.96; reasons.append(f"rüzgâr {wind:.0f} m/s")

    # Yağış
    if prec >= 4:    mu *= 0.92; card *= 1.05; reasons.append(f"yağış {prec:.1f} mm/h")
    elif prec >= 1:  mu *= 0.97; card *= 1.02; reasons.append(f"yağış {prec:.1f} mm/h")

    # Sıcaklık
    if temp <= 0:    mu *= 0.94; reasons.append(f"sıcaklık {temp:.0f}°C")
    elif temp >= 32: mu *= 0.95; reasons.append(f"sıcaklık {temp:.0f}°C")

    note = (" | Hava: "+", ".join(reasons)) if reasons else ""
    return {"goal_mu_factor": mu, "card_bias": card, "corner_bias": corner, "note": note}

# =================== ODDS (The Odds API, opsiyonel) ===========================
ODDS_API_KEY = os.getenv("ODDS_API_KEY")

def guess_sport_key(area: str, comp: str):
    a = (area or "").lower()
    c = (comp or "").lower()
    # Basit haritalama (gerekirse genişletiriz)
    if "germany" in a or "bl" in c or "bundes" in c:
        if "bl2" in c or "2." in c or "2 " in c:
            return "soccer_germany_bundesliga2"
        if "bl3" in c or "3." in c or "3 " in c:
            return "soccer_germany_bundesliga3"
        if "dfb" in c or "pokal" in c:
            return "soccer_germany_dfb_pokal"
        return "soccer_germany_bundesliga"
    if "turk" in a or "super" in c:
        return "soccer_turkey_super_league"
    # fallback: Avrupa büyük ligler için ekleme yapılabilir
    return None

def _remove_vig(p1, pX, p2):
    try:
        inv = (1/p1) + (1/pX) + (1/p2)
        q1, qX, q2 = (1/p1)/inv, (1/pX)/inv, (1/p2)/inv
        return q1, qX, q2
    except:
        return None

def fetch_market_probs(home, away, area, comp):
    if not ODDS_API_KEY: 
        return None, None  # (probs, odds)
    skey = guess_sport_key(area, comp)
    if not skey:
        return None, None
    try:
        url = f"https://api.the-odds-api.com/v4/sports/{skey}/odds"
        r = requests.get(url, params={
            "apiKey": ODDS_API_KEY,
            "regions": "eu",
            "markets": "h2h",
            "oddsFormat": "decimal",
            "dateFormat": "iso"
        }, timeout=35)
        r.raise_for_status()
        events = r.json()
        HN, AN = _norm_name(home), _norm_name(away)
        best = None
        for ev in events:
            h = _norm_name(ev.get("home_team",""))
            a = _norm_name(ev.get("away_team",""))
            if {h,a} == {HN,AN}:
                best = ev; break
        if not best:
            return None, None

        prices = []  # (o1,ox,o2)
        for bk in best.get("bookmakers", []):
            for mk in bk.get("markets", []):
                if mk.get("key")!="h2h": continue
                o1=ox=o2=None
                for outc in mk.get("outcomes", []):
                    name = (outc.get("name") or "").lower()
                    price = outc.get("price")
                    if price is None: continue
                    if name in ("home","1",home.lower()): o1 = float(price)
                    elif name in ("draw","x"): ox = float(price)
                    elif name in ("away","2",away.lower()): o2 = float(price)
                if o1 and ox and o2:
                    prices.append((o1,ox,o2))
        if not prices:
            return None, None

        # Konsensüs: her bookmaker için marjdan arındır → ortalama
        probs=[]
        for (o1,ox,o2) in prices:
            inv = (1/o1)+(1/ox)+(1/o2)
            probs.append(((1/o1)/inv, (1/ox)/inv, (1/o2)/inv))
        p1 = sum(p[0] for p in probs)/len(probs)
        pX = sum(p[1] for p in probs)/len(probs)
        p2 = sum(p[2] for p in probs)/len(probs)

        # Görsel için ortalama odds’u da dönelim:
        avg_o1 = sum(x[0] for x in prices)/len(prices)
        avg_ox = sum(x[1] for x in prices)/len(prices)
        avg_o2 = sum(x[2] for x in prices)/len(prices)

        return ({"1":p1,"X":pX,"2":p2}, {"1":avg_o1,"X":avg_ox,"2":avg_o2})
    except Exception:
        return None, None

# =================== FOOTBALL-DATA.ORG (opsiyonel) ============================
FD_TOKEN = os.getenv("FOOTBALL_DATA_TOKEN")
def fetch_fd_matches(day):
    if not FD_TOKEN: return []
    url = f"https://api.football-data.org/v4/matches?dateFrom={day}&dateTo={day}"
    try:
        r = requests.get(url, headers={"X-Auth-Token": FD_TOKEN}, timeout=35)
        if r.status_code==429:
            time.sleep(2); r = requests.get(url, headers={"X-Auth-Token": FD_TOKEN}, timeout=35)
        r.raise_for_status()
        out=[]
        for m in r.json().get("matches", []):
            comp = (m.get("competition") or {}).get("name","")
            area = (m.get("area") or {}).get("name","")
            ht = (m.get("homeTeam") or {}).get("name","")
            at = (m.get("awayTeam") or {}).get("name","")
            utc = m.get("utcDate")
            status = m.get("status","")
            score = (m.get("score") or {}).get("fullTime") or {}
            out.append({
                "source":"FD",
                "id": f"FD-{m.get('id')}",
                "utcDate": utc,
                "home": ht, "away": at,
                "comp": comp, "area": area,
                "status": status,
                "score": {"home": score.get("homeTeam"), "away": score.get("awayTeam")}
            })
        return out
    except Exception:
        return []

# =================== OPENLIGADB (ücretsiz/anahtarsız) =========================
OL_LEAGUES = ["bl1","bl2","bl3","dfb"]
def _ol_fetch_league(league: str, season: int):
    url = f"https://api.openligadb.de/getmatchdata/{league}/{season}"
    try:
        r = requests.get(url, timeout=35); r.raise_for_status(); return r.json()
    except Exception: return []

def fetch_openliga_matches(day):
    y = day.year
    raw=[]
    for sy in (y, y-1):
        for lg in OL_LEAGUES:
            data = _ol_fetch_league(lg, sy)
            if not data: continue
            raw.extend([("OL",lg,sy,m) for m in data])
    out=[]
    d0 = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    d1 = d0 + timedelta(days=1)
    for (_src, lg, sy, m) in raw:
        utc_str = m.get("MatchDateTimeUTC") or m.get("MatchDateTime")
        if not utc_str: continue
        try:
            dt = datetime.fromisoformat(utc_str.replace("Z","")).replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if not (d0 <= dt < d1): continue
        t1 = (m.get("Team1") or {}).get("TeamName","")
        t2 = (m.get("Team2") or {}).get("TeamName","")
        comp = lg.upper()
        status = "FINISHED" if m.get("MatchIsFinished") else "SCHEDULED"
        fh = fa = None
        for res in (m.get("MatchResults") or []):
            nm = (res.get("ResultName") or "").lower()
            if nm in {"endstand","fulltime"} or res.get("ResultTypeID")==2:
                fh = res.get("PointsTeam1"); fa = res.get("PointsTeam2")
        out.append({
            "source":"OL",
            "id": f"OL-{m.get('MatchID')}",
            "utcDate": dt.isoformat(),
            "home": t1, "away": t2,
            "comp": comp, "area": "Germany",
            "status": status,
            "score": {"home": fh, "away": fa}
        })
    return out

# =================== BİRLEŞTİRME / DEDUPE ====================================
def dedupe_merge(list_a, list_b):
    keyset=set(); out=[]
    def k(it): return (_norm_name(it["home"]), _norm_name(it["away"]), it["utcDate"][:16])
    for s in (list_a, list_b):
        for it in s:
            kk = k(it)
            if kk in keyset: continue
            keyset.add(kk); out.append(it)
    out.sort(key=lambda x: x["utcDate"])
    return out

# =================== MODEL (Poisson + SPI + Hava etkisi) =====================
def match_probs_poisson(home_name, away_name, utc_iso, league_mu=2.60, home_adv=1.12):
    spi = fetch_spi_table()
    h = spi.get(_norm_name(home_name))
    a = spi.get(_norm_name(away_name))
    hoff, hdef = (h["off"] if h else 1.0), (h["def"] if h else 1.0)
    aoff, adef = (a["off"] if a else 1.0), (a["def"] if a else 1.0)

    # Hava
    w = fetch_weather_for_match(home_name, utc_iso)
    wadd = weather_adjustments(w)
    mu = league_mu * wadd["goal_mu_factor"]

    lam_h = max(0.15, (mu/2.0) * (hoff / max(0.35, adef)) * home_adv)
    lam_a = max(0.15, (mu/2.0) * (aoff / max(0.35, hdef)))

    p1=pX=p2=0.0
    for hg in range(0,11):
        ph=poisson_pmf(lam_h,hg)
        for ag in range(0,11):
            pa=poisson_pmf(lam_a,ag)
            if   hg>ag: p1+=ph*pa
            elif hg==ag: pX+=ph*pa
            else: p2+=ph*pa
    return {"1":p1,"X":pX,"2":p2,"lam_h":lam_h,"lam_a":lam_a,"weather":wadd}

def blend_conf(p_model, p_market=None, hist_wr=None, p_spi=None, st=None):
    if st is None: st=_default_state()
    b = st.get("blend", {})
    w1, w2, w3, w4 = b.get("w1",0.55), b.get("w2",0.20), b.get("w3",0.05), b.get("w4",0.20)
    z = logit(p_model)*w1
    if p_market is not None: z += logit(p_market)*w2
    if hist_wr    is not None: z += logit(hist_wr)*w3
    if p_spi      is not None: z += logit(p_spi)*w4
    return sigmoid(z)

# =================== TAHMİN RAPORU ===========================================
def build_prediction_report(day):
    st = load_state()
    fd = fetch_fd_matches(day)
    ol = fetch_openliga_matches(day)
    matches = dedupe_merge(fd, ol)
    if not matches:
        return f"📭 Bugün ({day}) için maç bulunamadı."

    lines=[f"📅 Günün Tahminleri — {day} (OpenLigaDB + SPI + Hava + Odds)"]
    best=[]
    for m in matches:
        ht, at = m["home"], m["away"]
        when_local = datetime.fromisoformat(m["utcDate"].replace("Z","")).astimezone(TR_TZ).strftime("%H:%M")
        # SPI
        spi = spi_outcomes(ht, at)
        p_spi = {"1": spi["1"], "X": spi["X"], "2": spi["2"]} if spi else None
        # MODEL (Poisson + hava)
        pmdl = match_probs_poisson(ht, at, m["utcDate"])
        # ODDS (piyasa)
        pmkt, odds = fetch_market_probs(ht, at, m.get("area",""), m.get("comp",""))

        finals={}
        for side in ("1","X","2"):
            finals[side] = blend_conf(
                p_model=pmdl[side],
                p_market=None if not pmkt else pmkt.get(side),
                hist_wr=None,
                p_spi=None if not p_spi else p_spi.get(side),
                st=st
            )
        pick = max(finals, key=finals.get)
        conf = finals[pick]

        note = ""
        if spi: note += f" | SPI Δ={spi['_delta']:+.1f}"
        if pmdl["weather"]["note"]: note += pmdl["weather"]["note"]
        if odds: note += f" | Odds(avg) 1/X/2: {odds['1']:.2f}/{odds['X']:.2f}/{odds['2']:.2f}"

        line = (f"— {when_local} | {m['area']} {m['comp']} | {ht} vs {at} → "
                f"Seçim: {pick} | Güven: {conf:.0%} | λ_h/λ_a: {pmdl['lam_h']:.2f}/{pmdl['lam_a']:.2f}"
                f"{note}")
        lines.append(line)
        best.append((conf, line))

    lines.append("\n⭐ En Güçlü 5 Seçim:")
    for c,l in sorted(best, key=lambda x: x[0], reverse=True)[:5]:
        lines.append(f"  • {l}")

    return "\n".join(lines)

# =================== SONUÇ RAPORU ============================================
def build_results_report(day):
    fd = fetch_fd_matches(day)
    ol = fetch_openliga_matches(day)
    matches = dedupe_merge(fd, ol)
    finished = [m for m in matches if m.get("status")=="FINISHED" or
                (m.get("score",{}).get("home") is not None and m.get("score",{}).get("away") is not None)]
    if not finished:
        return f"🕗 {day} için final skoru bulunan maç yok (henüz tamamlanmamış olabilir)."

    lines=[f"📊 Günün Sonuçları — {day}"]
    for m in finished:
        ht, at = m["home"], m["away"]
        sc = m.get("score") or {}
        h, a = sc.get("home"), sc.get("away")
        when_local = datetime.fromisoformat(m["utcDate"].replace("Z","")).astimezone(TR_TZ).strftime("%H:%M")
        lines.append(f"— {when_local} | {m['area']} {m['comp']} | {ht} {h}-{a} {at}")
    return "\n".join(lines)

# =================== CLI ======================================================
def main():
    task = (sys.argv[1] if len(sys.argv)>1 else "predict").strip().lower()
    if task=="predict":
        body = build_prediction_report(TODAY)
        send_email(subject=f"Günün Tahminleri | {TODAY}", body=body)
        print("Predict mail gönderildi.")
    elif task=="results":
        body = build_results_report(TODAY)
        send_email(subject=f"Günün Sonuçları | {TODAY}", body=body)
        print("Results mail gönderildi.")
    else:
        print("Kullanım: python mailer.py [predict|results]")
        sys.exit(1)

if __name__=="__main__":
    main()
