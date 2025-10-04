# mailer.py — Ücretsiz veri + hava + risk filtresi + oto-öğrenme
import os, json, math, smtplib, ssl, datetime as dt
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests
from pathlib import Path

# === Ayarlar ================================================================
ONLY_EUROPE_TR = True
ALLOWED_AREAS = {
    "Turkey", "England", "Spain", "Italy", "Germany", "France",
    "Portugal", "Netherlands", "Belgium", "Scotland", "Austria",
    "Switzerland", "Denmark", "Norway", "Sweden", "Poland",
    "Czech Republic", "Croatia", "Greece", "Serbia", "Romania",
    "Hungary", "Ukraine", "Russia", "Ireland", "Wales", "Northern Ireland",
    "Iceland", "Finland", "Slovenia", "Slovakia", "Bosnia and Herzegovina",
    "Bulgaria"
}
TOP_N = 12                  # Günlük en fazla öneri (risk filtresi sonrası)
MAX_PER_LEAGUE = 2          # Bir ligden en fazla kaç öneri
FORM_WINDOW = 8             # Son kaç maç forma bakılsın

# === Secrets ================================================================
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_PASS")
GMAIL_TO   = os.getenv("GMAIL_TO") or GMAIL_USER
FD_TOKEN   = os.getenv("FOOTBALL_DATA_TOKEN")
MODE       = os.getenv("MODE", "PREDICT").upper()  # PREDICT / RESULTS

# === Oto-öğrenme dosyası ====================================================
STATE_PATH = Path("model_state.json")

def _default_state():
    return {"blend":{"w1":0.65,"w2":0.25,"w3":0.10}, "leagues":{}}

def load_state():
    if STATE_PATH.exists():
        try:
            return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except Exception:
            pass
    return _default_state()

def save_state(st):
    STATE_PATH.write_text(json.dumps(st, ensure_ascii=False, indent=2), encoding="utf-8")

def league_key(m):
    comp = (m.get("competition") or {})
    area = ((comp.get("area") or {}).get("name") or "")
    name = (comp.get("name") or "")
    return f"{area}:{name}".strip(":")

def league_params(st, key):
    d = st["leagues"].setdefault(key, {"mu_offset":0.0, "home_adv":1.10, "tau":1.00})
    return d

# === Yardımcılar ============================================================
def send_mail(subject, body):
    msg = MIMEMultipart()
    msg["From"], msg["To"], msg["Subject"] = GMAIL_USER, GMAIL_TO, subject
    msg.attach(MIMEText(body, "plain", "utf-8"))
    ctx = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=ctx) as server:
        server.login(GMAIL_USER, GMAIL_PASS)
        server.sendmail(GMAIL_USER, GMAIL_TO, msg.as_string())

def fd_get(path, params=None):
    url = "https://api.football-data.org/v4" + path
    headers = {"X-Auth-Token": FD_TOKEN} if FD_TOKEN else {}
    r = requests.get(url, headers=headers, params=params, timeout=30)
    if r.status_code in (200, 201):
        return r.json()
    return {}

# Open-Meteo Geocoding + Forecast (anahtarsız)
def geocode_city(q):
    url = "https://geocoding-api.open-meteo.com/v1/search"
    r = requests.get(url, params={"name":q, "count":1, "language":"en", "format":"json"}, timeout=20)
    j = r.json() if r.status_code==200 else {}
    if (j.get("results") or []):
        x = j["results"][0]
        return x["latitude"], x["longitude"]
    return None

def weather_factor(name, area, kick_dt):
    # şehir/klüp adıyla yaklaşıkla
    coords = geocode_city(f"{name} {area}") or geocode_city(area) or None
    if not coords:
        return 1.0, "hava: veri yok"
    lat, lon = coords
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude":lat, "longitude":lon,
        "hourly":"rain,wind_speed_10m",
        "start_date":kick_dt.date().isoformat(),
        "end_date":kick_dt.date().isoformat(),
        "timezone":"UTC"
    }
    r = requests.get(url, params=params, timeout=20)
    if r.status_code!=200: return 1.0, "hava: erişilemedi"
    j = r.json()
    hours = j.get("hourly",{})
    ts = hours.get("time",[])
    rains = hours.get("rain",[])
    winds = hours.get("wind_speed_10m",[])
    # Kickoff saatine en yakın saati bul
    target = kick_dt.strftime("%Y-%m-%dT%H:00")
    try:
        idx = ts.index(target)
    except ValueError:
        # en yakın saat
        diffs = [abs((dt.datetime.fromisoformat(t)-kick_dt).total_seconds()) for t in ts] if ts else []
        idx = int(diffs.index(min(diffs))) if diffs else None
    if idx is None: return 1.0, "hava: saat eşleşmedi"
    rain = float(rains[idx]) if idx<len(rains) else 0.0     # mm
    wind = float(winds[idx]) if idx<len(winds) else 0.0     # m/s değil km/h gelebilir; open-meteo m/s -> basit ölçek
    # Basit çarpan: yağmur & rüzgar gole zarar
    f = 1.0 - min(0.18, rain*0.03 + (wind/40.0)*0.10)
    f = max(0.75, min(1.05, f))
    note = f"hava: yağış {rain:.1f}, rüzgâr {wind:.1f} → x{f:.2f}"
    return f, note

def poisson_pmf(lam,k):
    return math.exp(-lam)*lam**k/math.factorial(k)

def probs_from_lambdas(lh, la):
    p1=pX=p2=0.0
    for h in range(0,11):
        ph=poisson_pmf(lh,h)
        for a in range(0,11):
            pa=poisson_pmf(la,a)
            if h>a: p1+=ph*pa
            elif h==a: pX+=ph*pa
            else: p2+=ph*pa
    return {"1":p1,"X":pX,"2":p2}

def sigmoid(z): return 1/(1+math.exp(-z))
def logit(p): p=min(max(p,1e-6),1-1e-6); return math.log(p/(1-p))

# Form & lig temposu (yaklaşık)
def team_recent(team_id, to_date, limit=FORM_WINDOW):
    frm = fd_get(f"/teams/{team_id}/matches",
                 params={"status":"FINISHED","dateTo":to_date.isoformat(),"limit":50}) or {}
    ms = frm.get("matches",[])
    ms = sorted(ms, key=lambda x: x.get("utcDate",""))[-limit:]
    gf=ga=n=0
    for m in ms:
        ht, at = m["homeTeam"]["id"], m["awayTeam"]["id"]
        ft = (m.get("score") or {}).get("fullTime",{})
        h, a = (ft.get("home",0) or 0), (ft.get("away",0) or 0)
        if team_id==ht: gf+=h; ga+=a
        else: gf+=a; ga+=h
        n+=1
    return {"gf":(gf/max(1,n)), "ga":(ga/max(1,n)), "n":n}

def league_goal_tempo(matches):
    # Lig gol temposunu o ligdeki yakın geçmişten yaklaşıkla (maç listesi veriliyorsa oradan)
    if not matches: return 2.40
    s=0; n=0
    for m in matches:
        ft = (m.get("score") or {}).get("fullTime",{})
        s += (ft.get("home",0) or 0) + (ft.get("away",0) or 0)
        n+=1
    return (s/max(1,n)) if n else 2.40

# === Tahmin (günün maçları) ================================================
def fetch_matches_for_date(d):
    data = fd_get("/matches", params={"dateFrom":d.isoformat(),"dateTo":d.isoformat()}) or {}
    arr = data.get("matches",[])
    out=[]
    for m in arr:
        comp = (m.get("competition") or {})
        area = ((comp.get("area") or {}).get("name") or "")
        if ONLY_EUROPE_TR and area and area not in ALLOWED_AREAS:
            continue
        out.append(m)
    return out

def pick_for_match(m, st):
    kick = dt.datetime.fromisoformat(m["utcDate"].replace("Z","+00:00")).replace(tzinfo=None)
    ht, at = m["homeTeam"], m["awayTeam"]
    area = ((m.get("competition",{}).get("area") or {}).get("name") or "")
    lk = league_key(m)
    lp = league_params(st, lk)

    # form
    hs = team_recent(ht["id"], kick.date())
    as_ = team_recent(at["id"], kick.date())

    # lig temposu yaklaşık
    mu = 2.40 * (1.0 + lp["mu_offset"])

    # hava etkisi
    wf, wnote = weather_factor(ht["name"], area, kick)

    # lambdalar
    lam_h = max(0.25, (mu/2.0) * (hs["gf"]/max(0.6, as_["ga"])) * lp["home_adv"] * wf)
    lam_a = max(0.25, (mu/2.0) * (as_["gf"]/max(0.6, hs["ga"])) * wf)

    pr = probs_from_lambdas(lam_h, lam_a)
    # seçim
    side = max(pr.items(), key=lambda x:x[1])[0]
    p_model = pr[side]

    # sıcaklık (temkinlilik)
    z = logit(p_model)/max(0.5, min(1.5, lp["tau"]))
    p_cal = sigmoid(z)

    reason = f"{ht['name']} vs {at['name']} | form(H:{hs['gf']:.2f}/{hs['ga']:.2f},A:{as_['gf']:.2f}/{as_['ga']:.2f}) | {wnote}"
    return {
        "match": m, "league": lk, "kick": kick, "side": side,
        "conf": p_cal, "reason": reason, "lam_h": lam_h, "lam_a": lam_a
    }

def risk_filter(picks):
    # Lig başına en fazla X seçim, tümünde TOP_N
    picks = sorted(picks, key=lambda x:x["conf"], reverse=True)
    kept, per_league = [], {}
    for p in picks:
        k = p["league"]
        if per_league.get(k,0) >= MAX_PER_LEAGUE: continue
        kept.append(p); per_league[k]=per_league.get(k,0)+1
        if len(kept)>=TOP_N: break
    return kept

def build_prediction_report(d):
    st = load_state()
    ms = fetch_matches_for_date(d)
    if not ms:
        return "Bugün için tahmin çıkarılacak maç bulunamadı."
    picks=[]
    for m in ms:
        # Başlamamış maç
        if m.get("status") not in ("TIMED","SCHEDULED"): continue
        try:
            picks.append(pick_for_match(m, st))
        except Exception:
            continue
    if not picks:
        return "Bugün için tahmin çıkarılacak maç bulunamadı."
    picks = risk_filter(picks)
    lines=[f"📆 Tarih: {d.isoformat()} — Başlamamış maçlar\n"]
    for p in picks:
        m, ht, at = p["match"], p["match"]["homeTeam"]["name"], p["match"]["awayTeam"]["name"]
        comp = (m.get("competition") or {}).get("name","")
        lines.append(f"• {p['kick'].strftime('%H:%M')} | {ht} vs {at} — Tahmin: {p['side']} (güven %{int(round(p['conf']*100))}) | {comp}\n  ↳ {p['reason']}")
    return "\n".join(lines)

# === Sonuç + oto-öğrenme ====================================================
def why_wrong(pick, lh, la, hg, ag):
    tot = lh+la; real = hg+ag
    if real < tot-0.8: return "tempo beklenenden düşük"
    if pick=="1" and ag>hg: return "denge/ev-saha hatası"
    if pick=="2" and hg>ag: return "denge/ev-saha hatası"
    return "aşırı güven"

def learn_from_match(st, m, pick, won, lam_h, lam_a):
    lk = league_key(m); lp = league_params(st, lk)
    ft = (m.get("score") or {}).get("fullTime",{})
    hg, ag = (ft.get("home",0) or 0), (ft.get("away",0) or 0)
    total = hg+ag; exp = lam_h+lam_a

    lr_mu, lr_home, lr_tau = 0.010, 0.020, 0.020
    if not won and total < exp-0.8:
        lp["mu_offset"] = max(-0.20, lp["mu_offset"] - lr_mu)
    if not won and pick=="1" and ag>hg:
        lp["home_adv"] = max(1.00, lp["home_adv"] - lr_home)
    if not won and pick=="2" and hg>ag:
        lp["home_adv"] = min(1.25, lp["home_adv"] + lr_home)
    if not won: lp["tau"] = min(1.40, lp["tau"] + lr_tau)
    else:       lp["tau"] = max(0.80, lp["tau"] - lr_tau/2)

def build_results_and_learn(d):
    st = load_state()
    # Basitçe bugün oynanan tüm maçlar
    ms = fetch_matches_for_date(d)
    if not ms:
        return "Bugün için maç bulunamadı."
    lines=[f"📊 Günün Sonuçları — {d.isoformat()}"]
    any_finished=False
    for m in sorted(ms, key=lambda x:x.get("utcDate","")):
        if m.get("status")!="FINISHED": continue
        any_finished=True
        ht, at = m["homeTeam"]["name"], m["awayTeam"]["name"]
        ft = (m.get("score") or {}).get("fullTime",{})
        hg, ag = (ft.get("home",0) or 0), (ft.get("away",0) or 0)

        # Tahmin anını yaklaşıkla: tekrar pick üret (yaklaşık)
        try:
            p = pick_for_match(m, st)
        except Exception:
            lines.append(f"— {ht} {hg}-{ag} {at} → veri eksik")
            continue

        won = (p["side"]=="1" and hg>ag) or (p["side"]=="2" and ag>hg) or (p["side"]=="X" and hg==ag)
        learn_from_match(st, m, p["side"], won, p["lam_h"], p["lam_a"])
        status = "✅ tuttu" if won else "❌ tutmadı"
        why = "" if won else f" — neden: {why_wrong(p['side'], p['lam_h'], p['lam_a'], hg, ag)}"
        lines.append(f"— {ht} {hg}-{ag} {at} → seçim: {p['side']} → {status}{why}")

    if not any_finished:
        return "Bugün için maç bulunamadı."
    save_state(st)
    return "\n".join(lines)

# === Çalıştır ===============================================================
def main():
    today = dt.date.today()
    if MODE=="PREDICT":
        body = build_prediction_report(today)
        send_mail(f"Günün Tahminleri | {today.isoformat()}", body)
    elif MODE=="RESULTS":
        body = build_results_and_learn(today)
        send_mail(f"Günün Sonuçları | {today.isoformat()}", body)
    else:
        body = "AUTO modu dışı çalıştırma. MODE=PREDICT veya MODE=RESULTS bekleniyor."
        send_mail(f"Tahmin Botu | Bilgi", body)

if __name__ == "__main__":
    main()
