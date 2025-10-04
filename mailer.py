# mailer.py  —  Ücretsiz tahmin + sonuç & oto-kalibrasyon (football-data.org + Open-Meteo)
# Gerekli Secrets: GMAIL_USER, GMAIL_PASS, GMAIL_TO, FOOTBALL_DATA_TOKEN
# MODE: AUTO (varsayılan) | PREDICT | RESULTS

import os, json, math, time, smtplib, sys
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText

import requests

# =============== AYARLAR ===============

# Sadece Avrupa + Türkiye (True yaparsan Brezilya vb. dışarıda kalır)
ONLY_EUROPE_TR = True
ALLOWED_AREAS = {
    "Turkey","England","Spain","Germany","Italy","France","Netherlands",
    "Portugal","Scotland","Belgium","Greece","Austria","Switzerland",
    "Denmark","Norway","Sweden","Poland","Czech Republic","Romania",
    "Serbia","Croatia","Hungary","Ukraine","Bulgaria","Slovenia",
    "Slovakia","Bosnia and Herzegovina","Ireland","Wales","Europe"
}

# Risk filtresi
MAX_PICKS_PER_LEAGUE = 4
TOP_N_PICKS = 12

# Varsayılan lig gol temposu (proxy, Poisson toplam λ ≈ 2.6 civarı)
DEFAULT_LEAGUE_MU = 2.60
DEFAULT_HOME_ADV   = 1.10

# Tarih: Türkiye saati
TR = timezone(timedelta(hours=3))
TODAY = datetime.now(TR).date()

# Dosyalar
STATE_PATH = "model_state.json"
MEM_PATH   = "mem.json"

# =============== ORTAM ===============

GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_PASS = os.environ.get("GMAIL_PASS")
GMAIL_TO   = os.environ.get("GMAIL_TO", GMAIL_USER)
FD_TOKEN   = os.environ.get("FOOTBALL_DATA_TOKEN")
MODE       = os.environ.get("MODE", "AUTO").upper().strip()

if not (GMAIL_USER and GMAIL_PASS and GMAIL_TO and FD_TOKEN):
    print("Missing secrets (GMAIL_USER/PASS/TO or FOOTBALL_DATA_TOKEN).")
    # yine de koşup sessizce çıkmasın:
    # sys.exit(0)

# =============== YARDIMCILAR ===============

def send_email(subject: str, body: str):
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = GMAIL_TO
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(GMAIL_USER, GMAIL_PASS)
        s.sendmail(GMAIL_USER, [GMAIL_TO], msg.as_string())

def load_json(path, default):
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return default

def save_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def logit(p): 
    p = max(1e-6, min(1-1e-6, p))
    return math.log(p/(1-p))

def sigmoid(z):
    return 1/(1+math.exp(-z))

def poisson_pmf(lam, k):
    lam = max(0.05, lam)
    return (lam**k)*math.exp(-lam)/math.factorial(k)

# =============== DURUM / ÖĞRENME ===============

def default_state():
    return {
        "blend": {"w1": 0.70, "w2": 0.00, "w3": 0.30},  # model / market / history (market yoksa 0)
        "leagues": {}  # key: "Area:Competition"
    }

def load_state():
    return load_json(STATE_PATH, default_state())

def save_state(st):
    save_json(STATE_PATH, st)

def league_key(m):
    area = (m.get("area") or {}).get("name","")
    comp = (m.get("competition") or {}).get("name","")
    return f"{area}:{comp}" if comp else area

def league_params(st, key):
    d = st["leagues"].setdefault(key, {
        "mu_offset": 0.0,    # toplam gol temposu düzeltmesi (×(1+ofset))
        "home_adv": DEFAULT_HOME_ADV,
        "tau": 1.00          # güven sıcaklığı (↑ = daha temkinli)
    })
    return d

def apply_temperature(p, tau):
    p = max(1e-3, min(1-1e-3, p))
    z = math.log(p/(1-p))
    return 1.0/(1.0+math.exp(-z / max(0.5, min(1.5, tau))))

# =============== FOOTBALL-DATA.ORG ===============

FD_BASE = "https://api.football-data.org/v4"

def fd_get(path, params=None):
    url = f"{FD_BASE}{path}"
    headers = {"X-Auth-Token": FD_TOKEN}
    for _ in range(3):
        r = requests.get(url, headers=headers, params=params, timeout=20)
        if r.status_code == 200:
            return r.json()
        # basit geri çekilme
        time.sleep(0.8)
    return {}

def fetch_matches_for_date(d):
    params = {"dateFrom": d.isoformat(), "dateTo": d.isoformat()}
    data = fd_get("/matches", params=params) or {}
    matches = data.get("matches", [])
    cleaned = []
    for m in matches:
        area = (m.get("area") or {}).get("name","")
        if ONLY_EUROPE_TR and area not in ALLOWED_AREAS:
            continue
        cleaned.append(m)
    return cleaned

def fetch_team_last5(team_id):
    # Son 5 bitmiş maç (form)
    params = {
        "status": "FINISHED",
        "limit": 5,
        "dateFrom": (TODAY - timedelta(days=120)).isoformat(),
        "dateTo": TODAY.isoformat()
    }
    data = fd_get(f"/teams/{team_id}/matches", params=params) or {}
    return data.get("matches", [])[:5]

# =============== OPEN-METEO (isteğe bağlı) ===============

def open_meteo_adjustment(lat=None, lon=None, kickoff_utc=None):
    # Basit: rüzgar > 7 m/s → tempo -4% ; yoğun yağış işareti varsa -6%
    try:
        if lat is None or lon is None or kickoff_utc is None:
            return 1.00
        ts = kickoff_utc.strftime("%Y-%m-%dT%H:00")
        url = "https://api.open-meteo.com/v1/forecast"
        params = {
            "latitude": lat, "longitude": lon,
            "hourly": "windspeed_10m,precipitation",
            "start_hour": ts, "end_hour": ts
        }
        r = requests.get(url, params=params, timeout=12)
        if r.status_code != 200:
            return 1.00
        h = (r.json().get("hourly") or {})
        ws = (h.get("windspeed_10m") or [None])[0]
        pr = (h.get("precipitation") or [None])[0]
        mult = 1.00
        if ws is not None and ws >= 7:   # rüzgar yüksek
            mult *= 0.96
        if pr is not None and pr >= 2:   # yağış belirgin
            mult *= 0.94
        return mult
    except Exception:
        return 1.00

# =============== MODEL ===============

def team_form_rating(last_matches, team_id):
    # Son 5 maçtan basit form
    gf = ga = w = d = l = 0
    for m in last_matches:
        ft = (m.get("score") or {}).get("fullTime", {})
        hg, ag = ft.get("home",0) or 0, ft.get("away",0) or 0
        th = (m.get("homeTeam") or {}).get("id")
        ta = (m.get("awayTeam") or {}).get("id")
        if th == team_id:
            gf += hg; ga += ag
            if hg>ag: w+=1
            elif hg==ag: d+=1
            else: l+=1
        elif ta == team_id:
            gf += ag; ga += hg
            if ag>hg: w+=1
            elif ag==hg: d+=1
            else: l+=1
    n = max(1, len(last_matches))
    atk = gf/max(1, n)      # g/m
    defn = ga/max(1, n)     # y/m
    pts = (3*w + d)/max(1, n)  # 0..3
    # Basit normalize
    return {
        "gf": max(0.3, atk),
        "ga": max(0.3, defn),
        "pts": pts
    }

def match_probs_with_params(hs, as_, league_mu, home_adv):
    # Form → λ (çok basit, güvenli)
    # Ev λ ~ (lig_mu/2) × (hs_atk / as_def) × home_adv
    lam_h = max(0.2, (league_mu/2.0) * (hs["gf"]/max(0.5, as_["ga"])) * home_adv)
    lam_a = max(0.2, (league_mu/2.0) * (as_["gf"]/max(0.5, hs["ga"])))
    p1=pX=p2=0.0
    for h in range(0,11):
        ph=poisson_pmf(lam_h,h)
        for a in range(0,11):
            pa=poisson_pmf(lam_a,a)
            if   h>a: p1+=ph*pa
            elif h==a: pX+=ph*pa
            else: p2+=ph*pa
    return {"1":p1,"X":pX,"2":p2,"lam_home":lam_h,"lam_away":lam_a}

def blend_conf(p_model, p_market=None, hist_wr=None, st=None):
    if st is None: w1,w2,w3 = 0.7,0.0,0.3
    else:
        w1 = st["blend"]["w1"]; w2 = st["blend"]["w2"]; w3 = st["blend"]["w3"]
    if p_market is None:  # odds yoksa ağırlığı kapat
        w2 = 0.0
        # w1'i biraz artır, w3 sabit
        w1 = min(0.9, max(0.6, w1+0.1))
    z = logit(p_model)*w1
    if p_market is not None: z += logit(p_market)*w2
    if hist_wr    is not None: z += logit(hist_wr)*w3
    return sigmoid(z)

# =============== HATIRLATICI / HAFIZA ===============

def load_mem():
    return load_json(MEM_PATH, {"pending":{}})

def save_mem(mem):
    save_json(MEM_PATH, mem)

# =============== TAHMİN ===============

def build_prediction_email():
    st = load_state()
    mem = load_mem()
    ms = fetch_matches_for_date(TODAY)

    scheduled = [m for m in ms if (m.get("status") in {"TIMED","SCHEDULED"})]
    if not scheduled:
        return f"Günün Tahminleri | {TODAY}", "Bugün için tahmin çıkarılacak maç bulunamadı."

    lines = [f"📅 Tarih: {TODAY} — Başlamamış maçlar", ""]

    picks = []
    by_league = {}

    for m in sorted(scheduled, key=lambda x: x.get("utcDate","")):
        area = (m.get("area") or {}).get("name","")
        comp = (m.get("competition") or {}).get("name","")
        lk = league_key(m)
        lp = league_params(st, lk)

        ht = m.get("homeTeam") or {}
        at = m.get("awayTeam") or {}
        hid, aid = ht.get("id"), at.get("id")

        # Form
        last_h = fetch_team_last5(hid)
        last_a = fetch_team_last5(aid)
        hs = team_form_rating(last_h, hid)
        as_ = team_form_rating(last_a, aid)

        # Lig temposu
        base_mu = DEFAULT_LEAGUE_MU
        mu_adj = base_mu*(1.0 + max(-0.25, min(0.25, lp["mu_offset"])))

        # (isteğe bağlı) hava - stad konumu bilinmediği için skip; varsayılan 1.00
        tempo_mult = 1.00
        mu_final = mu_adj * tempo_mult

        probs = match_probs_with_params(hs, as_, mu_final, lp["home_adv"])

        # En olası seçim + model p
        pick = max(("1","X","2"), key=lambda k: probs[k])
        p_model = probs[pick]

        # Sıcaklık kalibrasyonu (temkin/keskin)
        p_model_adj = apply_temperature(p_model, lp["tau"])

        # Takım/fiyat geçmişi yok → hist_wr None (ücretsiz mod)
        p_final = blend_conf(p_model_adj, p_market=None, hist_wr=None, st=st)

        # Lig bazlı limit
        by_league[lk] = by_league.get(lk, 0) + 1
        picks.append({
            "league": lk, "kick": m.get("utcDate","")[:16].replace("T"," "),
            "home": ht.get("name",""), "away": at.get("name",""),
            "pick": pick, "p_final": p_final, "why": explain_short(hs, as_, lp, mu_final, probs)
        })

    # Risk filtresi: lig başına limit + en iyi N
    filtered = []
    per_league_count = {}
    for p in sorted(picks, key=lambda x: x["p_final"], reverse=True):
        lg = p["league"]
        c = per_league_count.get(lg, 0)
        if c >= MAX_PICKS_PER_LEAGUE:
            continue
        filtered.append(p); per_league_count[lg]=c+1
        if len(filtered) >= TOP_N_PICKS:
            break

    # Pending hafızası (sonuçta bakmak için)
    mem["pending"] = {}
    for p in filtered:
        mid = f"{p['league']}|{p['home']}|{p['away']}|{p['kick']}"
        mem["pending"][mid] = {"side": p["pick"]}
    save_mem(mem)

    # Mail metni
    for p in filtered:
        conf = int(round(p["p_final"]*100))
        lines.append(f"⏰ {p['kick']} | {p['home']} vs {p['away']} — Tahmin: {p['pick']} (güven %{conf}) | {p['league']}")
        lines.append(f"   ↳ {p['why']}")
    if not filtered:
        lines.append("⚠️ Risk filtresi sonrası yayınlanacak güçlü seçim kalmadı.")

    subject = f"Günün Tahminleri | {TODAY}"
    body = "\n".join(lines)
    return subject, body

def explain_short(hs, as_, lp, mu, probs):
    best = max(("1","X","2"), key=lambda k: probs[k])
    tip = {"1":"Ev","X":"Berab.","2":"Dep"}[best]
    # nedenler: ev-saha, form (gf/ga), lig tempo
    note = []
    if lp["home_adv"] > 1.08: note.append("ev-saha +" )
    if hs["gf"] > as_["gf"]+0.3: note.append("ev form atk +" )
    if as_["ga"] > hs["ga"]+0.3: note.append("dep def zayıf")
    if mu < DEFAULT_LEAGUE_MU*0.97: note.append("tempo düşük")
    if mu > DEFAULT_LEAGUE_MU*1.03: note.append("tempo yüksek")
    if not note: note = ["denge avantajı"]
    return f"{tip} tarafında {', '.join(note)}"

# =============== SONUÇ + ÖĞRENME ===============

def build_results_email_and_learn():
    st = load_state()
    mem = load_mem()
    ms = fetch_matches_for_date(TODAY)
    finished = [m for m in ms if m.get("status") == "FINISHED"]

    if not finished:
        return f"Günün Sonuçları | {TODAY}", "Bugün için maç bulunamadı."

    lines = [f"📊 Günün Sonuçları — {TODAY}", ""]
    correct = wrong = 0

    for m in sorted(finished, key=lambda x: x.get("utcDate","")):
        area = (m.get("area") or {}).get("name","")
        comp = (m.get("competition") or {}).get("name","")
        lk = league_key(m)
        lp = league_params(st, lk)

        ht = m.get("homeTeam") or {}
        at = m.get("awayTeam") or {}
        ft = (m.get("score") or {}).get("fullTime",{})
        hg, ag = (ft.get("home",0) or 0), (ft.get("away",0) or 0)

        mid = f"{lk}|{ht.get('name','')}|{at.get('name','')}|{(m.get('utcDate','')[:16]).replace('T',' ')}"
        if mid not in mem.get("pending", {}):
            continue  # bugün tahmin edilmemiş
        pick = mem["pending"][mid]["side"]

        # Tahmin anına yakın lam tahmini (yaklaşık)
        last_h = fetch_team_last5(ht.get("id"))
        last_a = fetch_team_last5(at.get("id"))
        hs = team_form_rating(last_h, ht.get("id"))
        as_ = team_form_rating(last_a, at.get("id"))
        mu_adj = DEFAULT_LEAGUE_MU*(1.0 + lp["mu_offset"])
        probs = match_probs_with_params(hs, as_, mu_adj, lp["home_adv"])

        won = (pick=="1" and hg>ag) or (pick=="2" and ag>hg) or (pick=="X" and hg==ag)
        if won: correct+=1
        else: wrong+=1

        reason = "" if won else (" — Neden: "+ why_wrong(pick, probs["lam_home"], probs["lam_away"], hg, ag))
        lines.append(f"— {ht.get('name','')} {hg}-{ag} {at.get('name','')} → Seçim: {pick} → {'✅ tuttu' if won else '❌ tutmadı'}{reason}")

        # Mikro öğrenme
        micro_learn_from_match(st, m, pick, won, probs["lam_home"], probs["lam_away"])

        # pending'den düş
        mem["pending"].pop(mid, None)

    save_mem(mem)
    save_state(st)

    lines.append(f"\nÖzet: {correct} doğru / {wrong} yanlış")
    subject = f"Günün Sonuçları | {TODAY}"
    body = "\n".join(lines)
    return subject, body

def why_wrong(pick, lam_h, lam_a, hg, ag):
    total = hg+ag
    exp_tot = lam_h+lam_a
    if total < exp_tot - 0.8: return "tempo beklenenden düşük"
    if total > exp_tot + 0.8: return "tempo beklenenden yüksek"
    if pick=="1" and ag>hg:  return "denge sapması (deplasman)"
    if pick=="2" and hg>ag:  return "denge sapması (ev)"
    return "varyans"

def micro_learn_from_match(st, m, pick, won, lam_h, lam_a):
    lk = league_key(m)
    lp = league_params(st, lk)
    lr_mu, lr_home, lr_tau = 0.010, 0.020, 0.020

    ft = (m.get("score") or {}).get("fullTime",{})
    hg, ag = (ft.get("home",0) or 0), (ft.get("away",0) or 0)
    total = hg+ag
    exp_tot = lam_h+lam_a

    if not won and (total < exp_tot - 0.8):
        lp["mu_offset"] = max(-0.25, lp["mu_offset"] - lr_mu)
    if not won and (total > exp_tot + 0.8):
        lp["mu_offset"] = min( 0.25, lp["mu_offset"] + lr_mu)

    if not won and pick=="1" and ag>hg:
        lp["home_adv"] = max(1.00, lp["home_adv"] - lr_home)
    if not won and pick=="2" and hg>ag:
        lp["home_adv"] = min(1.25, lp["home_adv"] + lr_home)

    if not won:
        lp["tau"] = min(1.40, lp["tau"] + lr_tau)
    else:
        lp["tau"] = max(0.80, lp["tau"] - lr_tau/2)

# =============== ÇALIŞTIR ===============

def main():
    # AUTO → gündüz PREDICT, gece RESULTS
    mode = MODE
    if mode == "AUTO":
        hour_tr = datetime.now(TR).hour
        mode = "PREDICT" if hour_tr < 20 else "RESULTS"

    if mode == "PREDICT":
        subject, body = build_prediction_email()
        send_email(subject, body)
        print(subject)
    elif mode == "RESULTS":
        subject, body = build_results_email_and_learn()
        send_email(subject, body)
        print(subject)
    else:
        print(f"Unknown MODE: {MODE}")

if __name__ == "__main__":
    main()
