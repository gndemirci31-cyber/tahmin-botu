# mailer.py  —  OpenLigaDB + SPI entegre, tek dosya
# -------------------------------------------------
# Gerekli env:
#   GMAIL_USER, GMAIL_PASS, GMAIL_TO          (zorunlu)
#   FOOTBALL_DATA_TOKEN                       (opsiyonel - football-data.org)
#   ODDS_API_KEY                              (opsiyonel - The Odds API)
#
# Çalıştırma modu (GitHub Actions 'python mailer.py <task>'):
#   python mailer.py predict   -> sabah tahmin maili
#   python mailer.py results   -> gece sonuç maili

import os, sys, json, math, time, io, csv, unicodedata
from datetime import datetime, timedelta, timezone
import smtplib, ssl
import requests

# ==== ZAMAN/LOKAL AYARLARI ===================================================
TR_TZ = timezone(timedelta(hours=3))  # Europe/Istanbul (UTC+3 sabit)
TODAY = datetime.now(TR_TZ).date()

# ==== E-POSTA AYARLARI =======================================================
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_PASS")
GMAIL_TO   = os.getenv("GMAIL_TO", GMAIL_USER)

def send_email(subject: str, body: str):
    assert GMAIL_USER and GMAIL_PASS and GMAIL_TO, "Gmail env eksik."
    msg = f"From: {GMAIL_USER}\r\nTo: {GMAIL_TO}\r\nSubject: {subject}\r\n\r\n{body}"
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
        server.login(GMAIL_USER, GMAIL_PASS)
        server.sendmail(GMAIL_USER, [GMAIL_TO], msg.encode("utf-8"))

# ==== MATEMATIK / YARDIMCILAR ================================================
def poisson_pmf(lmb, k):
    return math.exp(-lmb) * (lmb ** k) / math.factorial(k)

def sigmoid(z): return 1/(1+math.exp(-z))
def logit(p): 
    p=min(max(p,1e-6),1-1e-6)
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

# ==== MODEL DURUMU (opsiyonel küçük kalibrasyon) =============================
STATE_PATH = "model_state.json"
def _default_state():
    return {
        "blend": {"w1": 0.60, "w2": 0.10, "w3": 0.05, "w4": 0.25},  # w4=SPI ağırlığı
        "leagues": {},   # ileride mikro düzeltmeler için
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

# ==== SPI (FiveThirtyEight) ===================================================
SPI_URL = "https://projects.fivethirtyeight.com/soccer-api/club/spi_global_rankings.csv"
_spi_cache = None

def fetch_spi_table() -> dict:
    global _spi_cache
    if _spi_cache is not None: return _spi_cache
    try:
        r = requests.get(SPI_URL, timeout=30)
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
        return d
    except Exception:
        return {}

def spi_outcomes(home_name: str, away_name: str, k: float=10.0):
    tab = fetch_spi_table()
    h, a = tab.get(_norm_name(home_name)), tab.get(_norm_name(away_name))
    if not h or not a: return None
    delta = (h["spi"] - a["spi"])
    p_home_raw = 1.0/(1.0+math.exp(-delta/k))
    closeness = math.exp(-abs(delta)/12.0)
    p_draw = max(0.18, min(0.35, 0.22 + 0.18*closeness))
    scale = 1.0 - p_draw
    return {"1": p_home_raw*scale, "X": p_draw, "2": (1-p_home_raw)*scale, "_delta": delta, "_h":h, "_a":a}

# ==== ORANLAR (opsiyonel) ====================================================
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
def market_probs_stub(home, draw, away):
    """Fractional/decimal oddsı marjdan arındırıp p1,pX,p2 döndürür."""
    try:
        o1, ox, o2 = float(home), float(draw), float(away)
        inv = 1/o1 + 1/ox + 1/o2
        return {"1": (1/o1)/inv, "X": (1/ox)/inv, "2": (1/o2)/inv}
    except: 
        return None

# ==== HAVA (Open-Meteo - opsiyonel, koordinat yoksa atlar) ====================
def weather_penalty_stub():
    # Koordinat/şehir yoksa nötr bırak (0 etki)
    return 0.0  # 0..1 arası bir ceza katsayısı olabilirdi; şimdilik 0

# ==== FOOTBALL-DATA.ORG  (opsiyonel) =========================================
FD_TOKEN = os.getenv("FOOTBALL_DATA_TOKEN")
def fetch_fd_matches(day: datetime.date):
    if not FD_TOKEN: return []
    date_s = day.isoformat()
    url = f"https://api.football-data.org/v4/matches?dateFrom={date_s}&dateTo={date_s}"
    try:
        r = requests.get(url, headers={"X-Auth-Token": FD_TOKEN}, timeout=30)
        if r.status_code==429:  # rate limit
            time.sleep(2)
            r = requests.get(url, headers={"X-Auth-Token": FD_TOKEN}, timeout=30)
        r.raise_for_status()
        out=[]
        for m in r.json().get("matches",[]):
            comp = (m.get("competition") or {}).get("name","")
            area = (m.get("area") or {}).get("name","")
            ht = (m.get("homeTeam") or {}).get("name","")
            at = (m.get("awayTeam") or {}).get("name","")
            utc = m.get("utcDate")
            status = m.get("status","")
            score = ((m.get("score") or {}).get("fullTime") or {})
            out.append({
                "source":"FD",
                "id": f"FD-{m.get('id')}",
                "utcDate": utc,
                "home": ht,
                "away": at,
                "comp": comp,
                "area": area,
                "status": status,
                "score": {"home": score.get("homeTeam"), "away": score.get("awayTeam")}
            })
        return out
    except Exception:
        return []

# ==== OPENLIGADB (Almanya ağırlıklı - ücretsiz/anahtarsız) ===================
# BL1 (Bundesliga), BL2, BL3, DFB Pokal'ı kapsıyoruz.
OL_LEAGUES = ["bl1","bl2","bl3","dfb"]

def _ol_fetch_league(league: str, season: int):
    url = f"https://api.openligadb.de/getmatchdata/{league}/{season}"
    try:
        r = requests.get(url, timeout=35)
        r.raise_for_status()
        return r.json()
    except Exception:
        return []

def fetch_openliga_matches(day: datetime.date):
    # O günün (UTC) maçlarını filtrele
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
            # Örn: "2025-08-15T18:30:00"
            dt = datetime.fromisoformat(utc_str.replace("Z","")).replace(tzinfo=timezone.utc)
        except Exception:
            continue
        if not (d0 <= dt < d1): 
            continue
        t1 = (m.get("Team1") or {}).get("TeamName","")
        t2 = (m.get("Team2") or {}).get("TeamName","")
        comp = lg.upper()
        status = "FINISHED" if m.get("MatchIsFinished") else "SCHEDULED"
        # skor (varsa)
        fh = fa = None
        try:
            for res in (m.get("MatchResults") or []):
                if res.get("ResultName","").lower() in {"endstand","fulltime"} or res.get("ResultTypeID")==2:
                    fh = res.get("PointsTeam1"); fa = res.get("PointsTeam2")
        except Exception:
            pass
        out.append({
            "source":"OL",
            "id": f"OL-{m.get('MatchID')}",
            "utcDate": dt.isoformat(),
            "home": t1,
            "away": t2,
            "comp": comp,
            "area": "Germany",
            "status": status,
            "score": {"home": fh, "away": fa}
        })
    return out

# ==== BİRLEŞTİRME & DEDUPE ====================================================
def dedupe_merge(list_a, list_b):
    keyset=set(); out=[]
    def k(item):
        # aynı gün içinde aynı eşleşme & saate yakın olanları birleştir
        return (_norm_name(item["home"]), _norm_name(item["away"]), item["utcDate"][:16])
    for s in (list_a, list_b):
        for it in s:
            kk = k(it)
            if kk in keyset: 
                continue
            keyset.add(kk); out.append(it)
    # saat sırasına göre
    out.sort(key=lambda x: x["utcDate"])
    return out

# ==== OLASILIK / MODEL ========================================================
def match_probs_poisson(home_name, away_name, league_mu=2.6, home_adv=1.10):
    """
    Poisson tabanlı 1/X/2 çıkarımı. SPI of/def varsa λ'ları SPI ile şekillendiririz.
    """
    spi = fetch_spi_table()
    h = spi.get(_norm_name(home_name))
    a = spi.get(_norm_name(away_name))
    # of/def SPI metrikleri yoksa 1.0 al
    hoff, hdef = (h["off"] if h else 1.0), (h["def"] if h else 1.0)
    aoff, adef = (a["off"] if a else 1.0), (a["def"] if a else 1.0)

    # Hava etkisi (opsiyonel proxy)
    wpen = weather_penalty_stub()  # 0.0 => etki yok
    mu = league_mu * (1.0 - 0.15*wpen)

    lam_home = max(0.15, (mu/2.0) * (hoff / max(0.35, adef)) * home_adv)
    lam_away = max(0.15, (mu/2.0) * (aoff / max(0.35, hdef)))

    p1=pX=p2=0.0
    for hgo in range(0,11):
        ph=poisson_pmf(lam_home,hgo)
        for ago in range(0,11):
            pa=poisson_pmf(lam_away,ago)
            if hgo>ago: p1+=ph*pa
            elif hgo==ago: pX+=ph*pa
            else: p2+=ph*pa
    return {"1":p1,"X":pX,"2":p2,"lam_h":lam_home,"lam_a":lam_away}

def blend_conf(p_model, p_market=None, hist_wr=None, p_spi=None, st=None):
    if st is None: st=_default_state()
    b = st.get("blend", {})
    w1 = b.get("w1",0.60); w2=b.get("w2",0.10); w3=b.get("w3",0.05); w4=b.get("w4",0.25)
    z = logit(p_model)*w1
    if p_market is not None: z += logit(p_market)*w2
    if hist_wr    is not None: z += logit(hist_wr)*w3
    if p_spi      is not None: z += logit(p_spi)*w4
    return sigmoid(z)

# ==== TAHMİN RAPORU ===========================================================
def build_prediction_report(day: datetime.date):
    st = load_state()
    fd = fetch_fd_matches(day)
    ol = fetch_openliga_matches(day)
    matches = dedupe_merge(fd, ol)

    if not matches:
        return f"📭 Bugün ({day.isoformat()}) için eşleşme bulunamadı."

    lines=[f"📅 Günün Tahminleri — {day.strftime('%Y-%m-%d')} (OpenLigaDB + SPI)"]
    best=[]
    for m in matches:
        ht, at = m["home"], m["away"]
        spi = spi_outcomes(ht, at)  # None olabilir
        p_spi_1 = spi["1"] if spi else None
        p_spi_X = spi["X"] if spi else None
        p_spi_2 = spi["2"] if spi else None
        delta   = spi["_delta"] if spi else 0.0

        # Model (Poisson) – SPI of/def ile λ şekilleniyor
        probs = match_probs_poisson(ht, at, league_mu=2.60, home_adv=1.12)
        p_model = probs  # outcome bazlı

        # Piyasa (opsiyonel) — basit placeholder (gerçek odds bağlamadıysan None kalır)
        p_mkt = None

        # Outcome’lara göre nihai skorlar
        finals={}
        for side in ("1","X","2"):
            finals[side] = blend_conf(
                p_model=p_model[side],
                p_market=None if p_mkt is None else p_mkt.get(side),
                hist_wr=None,
                p_spi=({"1":p_spi_1,"X":p_spi_X,"2":p_spi_2}.get(side) if spi else None),
                st=st
            )

        pick = max(finals, key=finals.get)
        conf = finals[pick]
        note_spi = f" | SPI Δ={delta:+.1f}" if spi else ""
        when_local = datetime.fromisoformat(m["utcDate"].replace("Z","")).astimezone(TR_TZ).strftime("%H:%M")
        line = (f"— {when_local} | {m['area']} {m['comp']} | {ht} vs {at} → "
                f"Seçim: {pick} | Güven: {conf:.0%}{note_spi} "
                f"| λ_h/λ_a: {probs['lam_h']:.2f}/{probs['lam_a']:.2f}")
        lines.append(line)
        best.append((conf, line))

    # En iyi 5
    lines.append("\n⭐ En Güçlü 5 Seçim:")
    for c,l in sorted(best, key=lambda x: x[0], reverse=True)[:5]:
        lines.append(f"  • {l}")

    return "\n".join(lines)

# ==== SONUÇ RAPORU ============================================================
def build_results_report(day: datetime.date):
    # Aynı kaynaklardan biten maçları bul → skorla yaz
    fd = fetch_fd_matches(day)
    ol = fetch_openliga_matches(day)
    matches = dedupe_merge(fd, ol)
    finished = [m for m in matches if m.get("status")=="FINISHED" or
                (m.get('score',{}).get('home') is not None and m.get('score',{}).get('away') is not None)]

    if not finished:
        return f"🕗 {day.isoformat()} için final skoru bulunan maç yok (henüz tamamlanmamış olabilir)."

    lines=[f"📊 Günün Sonuçları — {day.strftime('%Y-%m-%d')}"]
    for m in finished:
        ht, at = m["home"], m["away"]
        sc = m.get("score") or {}
        h, a = sc.get("home"), sc.get("away")
        when_local = datetime.fromisoformat(m["utcDate"].replace("Z","")).astimezone(TR_TZ).strftime("%H:%M")
        lines.append(f"— {when_local} | {m['area']} {m['comp']} | {ht} {h}-{a} {at}")

    return "\n".join(lines)

# ==== CLI =====================================================================
def main():
    task = (sys.argv[1] if len(sys.argv)>1 else "predict").strip().lower()
    if task=="predict":
        body = build_prediction_report(TODAY)
        send_email(subject=f"Günün Tahminleri | {TODAY.isoformat()}", body=body)
        print("Predict mail gönderildi.")
    elif task=="results":
        body = build_results_report(TODAY)
        send_email(subject=f"Günün Sonuçları | {TODAY.isoformat()}", body=body)
        print("Results mail gönderildi.")
    else:
        print("Kullanım: python mailer.py [predict|results]")
        sys.exit(1)

if __name__=="__main__":
    main()
