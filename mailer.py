# mailer.py  —  Full auto: tahmin + (opsiyonel) piyasa + hava + hakem-proxy + kart&korner + gece sonuç & öğrenme
# Python 3.11 / yalnızca standart kütüphane + requests
import os, json, math, time, random, statistics
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText
import smtplib
import requests

# =========================
# Env / Secrets
# =========================
GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_PASS = os.environ.get("GMAIL_PASS")
GMAIL_TO   = os.environ.get("GMAIL_TO", GMAIL_USER)
FD_TOKEN   = os.environ.get("FOOTBALL_DATA_TOKEN")   # football-data.org
ODDS_KEY   = os.environ.get("ODDS_API_KEY")          # TheOddsAPI (opsiyonel)
MODE       = os.environ.get("MODE", "").strip().upper()  # PREDICT / RESULTS / AUTO
TZ_TR      = timezone(timedelta(hours=3))

# =========================
# Mail helper
# =========================
def send_mail(subject: str, body: str):
    if not (GMAIL_USER and GMAIL_PASS and GMAIL_TO):
        print("E-posta secret'ları eksik.")
        return
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = GMAIL_TO
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(GMAIL_USER, GMAIL_PASS)
        s.sendmail(GMAIL_USER, [GMAIL_TO], msg.as_string())

# =========================
# Utils
# =========================
def utc_now():
    return datetime.now(timezone.utc)

def today_tr():
    return datetime.now(TZ_TR).date()

def iso_date(d):  # YYYY-MM-DD
    return d.strftime("%Y-%m-%d")

def clamp(x, lo, hi): 
    return max(lo, min(hi, x))

def sigmoid(z): 
    return 1.0/(1.0+math.exp(-z))

def logit(p):
    p = clamp(p, 1e-4, 1-1e-4)
    return math.log(p/(1-p))

def poisson_pmf(lam, k):
    try:
        return math.exp(-lam) * lam**k / math.factorial(k)
    except OverflowError:
        return 0.0

def safe_get(d, *keys, default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict): return default
        cur = cur.get(k)
    return cur if cur is not None else default

# =========================
# football-data.org —— FREE
# =========================
FD_BASE = "https://api.football-data.org/v4"
FD_HEADERS = {"X-Auth-Token": FD_TOKEN} if FD_TOKEN else {}

def fd_get(path, params=None):
    if not FD_TOKEN:
        return None
    url = FD_BASE + path
    for _ in range(2):
        r = requests.get(url, headers=FD_HEADERS, params=params, timeout=20)
        if r.status_code == 200:
            return r.json()
        time.sleep(0.5)
    return None

def get_scheduled_matches(date_utc: datetime):
    """Bugün oynanacak planlı maçlar (SCHEDULED/ TIMED)."""
    d0 = iso_date(date_utc.date())
    d1 = iso_date(date_utc.date())
    data = fd_get("/matches", {"dateFrom": d0, "dateTo": d1})
    ms = []
    if data and "matches" in data:
        for m in data["matches"]:
            if m.get("status") in ("SCHEDULED","TIMED","POSTPONED"):
                ms.append(m)
    return ms

def get_finished_matches(date_utc: datetime):
    """Bugün biten maçlar."""
    d0 = iso_date(date_utc.date())
    d1 = iso_date(date_utc.date())
    data = fd_get("/matches", {"dateFrom": d0, "dateTo": d1})
    ms = []
    if data and "matches" in data:
        for m in data["matches"]:
            if m.get("status") == "FINISHED":
                ms.append(m)
    return ms

def team_recent_form(team_id: int, days=60, limit=10):
    """Son maçlardan form (GF/GA) ve ev/dış ayrımı basit metrikler."""
    end = utc_now().date()
    start = end - timedelta(days=days)
    data = fd_get(f"/teams/{team_id}/matches", {
        "dateFrom": iso_date(start), "dateTo": iso_date(end),
        "status": "FINISHED", "limit": limit
    })
    gf = ga = home_gf = away_gf = 0
    home_ga = away_ga = 0
    n = nh = na = 0
    if data and "matches" in data:
        for m in data["matches"]:
            sc = m.get("score",{}).get("fullTime",{})
            hg, ag = sc.get("home",0) or 0, sc.get("away",0) or 0
            home_id = safe_get(m,"homeTeam","id",default=None)
            away_id = safe_get(m,"awayTeam","id",default=None)
            if team_id == home_id:
                gf += hg; ga += ag; home_gf += hg; home_ga += ag; nh += 1
            elif team_id == away_id:
                gf += ag; ga += hg; away_gf += ag; away_ga += hg; na += 1
            n += 1
    av = lambda x,c: (x/c) if c>0 else 1.0
    return {
        "n": n,
        "gf": av(gf, n) if n else 1.1,
        "ga": av(ga, n) if n else 1.1,
        "home_gf": av(home_gf, nh) if nh else 1.1,
        "home_ga": av(home_ga, nh) if nh else 1.1,
        "away_gf": av(away_gf, na) if na else 1.1,
        "away_ga": av(away_ga, na) if na else 1.1,
    }

def competition_goal_tempo(comp_id: int, days=60):
    """Lig gol temposu (son ~60 gün) — veri yoksa 2.5."""
    end = utc_now().date()
    start = end - timedelta(days=days)
    data = fd_get(f"/competitions/{comp_id}/matches", {
        "dateFrom": iso_date(start), "dateTo": iso_date(end), "status":"FINISHED"
    })
    gsum = cnt = 0
    if data and "matches" in data:
        for m in data["matches"]:
            sc = m.get("score",{}).get("fullTime",{})
            hg, ag = sc.get("home",0) or 0, sc.get("away",0) or 0
            gsum += (hg+ag); cnt += 1
    return (gsum/cnt) if cnt>4 else 2.5

# =========================
# Open-Meteo (free) – Weather
# =========================
def geocode_city(name: str):
    try:
        r = requests.get("https://geocoding-api.open-meteo.com/v1/search",
                         params={"name":name, "count":1, "language":"en", "format":"json"}, timeout=15)
        if r.status_code==200:
            js = r.json()
            if js.get("results"):
                it = js["results"][0]
                return it["latitude"], it["longitude"]
    except: pass
    return None

def weather_effect_for_match(city_name: str, when_utc: str):
    """
    Basit hava düzeltmesi (tempo faktörü):
    Rüzgâr >7 m/s => -6% ; Yağış olasılığı >60% => -6% ; ikisi birden => ~-12%.
    """
    if not city_name:
        return 1.0, None
    latlon = geocode_city(city_name)
    if not latlon:
        return 1.0, None
    lat, lon = latlon
    try:
        # maç saati civarında saatlik veriyi çek
        r = requests.get("https://api.open-meteo.com/v1/forecast",
                         params={
                             "latitude": lat, "longitude": lon,
                             "hourly":"windspeed_10m,precipitation_probability",
                             "timezone":"UTC"
                         }, timeout=20)
        if r.status_code != 200:
            return 1.0, None
        js = r.json()
        times = js.get("hourly",{}).get("time",[])
        winds = js.get("hourly",{}).get("windspeed_10m",[])
        precs = js.get("hourly",{}).get("precipitation_probability",[])
        # en yakın saat için indeks:
        if not times: return 1.0, None
        # when_utc: ISO string
        if when_utc in times:
            idx = times.index(when_utc)
        else:
            # yakına bul
            diffs = [(abs(datetime.fromisoformat(t+"+00:00") - datetime.fromisoformat(when_utc+"+00:00")).total_seconds(),i)
                     for i,t in enumerate(times)]
            idx = min(diffs)[1]
        wind = winds[idx] if idx < len(winds) else 0
        pprec = precs[idx] if idx < len(precs) else 0
        factor = 1.0
        note = []
        if wind is not None and wind > 7: 
            factor *= 0.94; note.append(f"rüzgar {wind} m/s")
        if pprec is not None and pprec > 60:
            factor *= 0.94; note.append(f"yağış olası %{pprec}")
        return factor, (", ".join(note) if note else None)
    except:
        return 1.0, None

# =========================
# TheOddsAPI (opsiyonel) – Market
# =========================
# Keys ör.: soccer_epl, soccer_spain_la_liga, soccer_italy_serie_a, soccer_germany_bundesliga, soccer_france_ligue_one, soccer_turkey_super_ligi, vb.
ODDS_KEYS = [
    "soccer_epl","soccer_spain_la_liga","soccer_italy_serie_a","soccer_germany_bundesliga",
    "soccer_france_ligue_one","soccer_turkey_super_ligi","soccer_portugal_primeira_liga",
    "soccer_netherlands_eredivisie","soccer_uefa_champs_league","soccer_uefa_europa_league",
    "soccer_brazil_campeonato_a","soccer_usa_mls","soccer_argentina_primera_division",
]

def odds_search_match(home_name, away_name, iso_when):
    """Piyasa 1X2 oranları — varsa döner, yoksa None (free tier limitli, robust yazıldı)."""
    if not ODDS_KEY:
        return None
    home_l = home_name.lower(); away_l = away_name.lower()
    best = None
    for key in ODDS_KEYS:
        try:
            url = f"https://api.the-odds-api.com/v4/sports/{key}/odds"
            params = {"apiKey": ODDS_KEY, "regions":"eu,uk,us", "markets":"h2h", "oddsFormat":"decimal"}
            r = requests.get(url, params=params, timeout=25)
            if r.status_code != 200: 
                continue
            for ev in r.json():
                teams = [t.lower() for t in ev.get("teams",[])]
                if not teams or len(teams)<2: 
                    continue
                # basit eşleşme: isim parçaları
                if home_l[:6] in teams[0] or home_l[:6] in teams[1]:
                    if away_l[:6] in teams[0] or away_l[:6] in teams[1]:
                        # market h2h
                        for bk in ev.get("bookmakers",[]):
                            for mk in bk.get("markets",[]):
                                if mk.get("key")=="h2h":
                                    odds = mk.get("outcomes",[])
                                    # outcomes: {name: team name or "Draw", price: 1.85}
                                    o = {"1":None,"X":None,"2":None}
                                    for x in odds:
                                        nm = x.get("name","").lower()
                                        pr = x.get("price",None)
                                        if pr is None: continue
                                        if "draw" in nm: o["X"]=pr
                                        elif home_l[:6] in nm: o["1"]=pr
                                        elif away_l[:6] in nm: o["2"]=pr
                                    # keep best (yüksek fiyat = düşük marj)
                                    if all(o.values()):
                                        best = o
                                        break
                        if best: break
            if best: break
        except: 
            continue
    if not best:
        return None
    # Marjdan arındır
    inv = {k: 1.0/float(v) for k,v in best.items()}
    s = sum(inv.values())
    p = {k: inv[k]/s for k in inv}
    return p  # {"1":p1, "X":pX, "2":p2}

# =========================
# Hakem & Kart/Korner proxy
# =========================
def referee_proxy(match):
    """
    Hakem adı yoksa lig-temposuna göre nötr olsun.
    Basit proxy: önemli/derbi (isim kesişimi) + yarışma safhası (final/playoff?) küçük artış.
    """
    hname = safe_get(match,"homeTeam","name",default="")
    aname = safe_get(match,"awayTeam","name",default="")
    comp_name = safe_get(match,"competition","name",default="")

    derby = 0.0
    s1, s2 = hname.split(" ")[0].lower(), aname.split(" ")[0].lower()
    if s1 == s2 or s1 in aname.lower() or s2 in hname.lower():
        derby += 0.1  # aynı şehir/isim benzerliği -> gerginlik

    comp_boost = 0.0
    if any(w in comp_name.lower() for w in ["final","play-off","cup","kupa"]):
        comp_boost += 0.05

    return clamp(1.0 + derby + comp_boost, 0.9, 1.2)  # kart eğilimini ölçekler

def cards_and_corners_lambda(total_goal_mu, strength_ratio, ref_factor):
    """
    Basit, güvenli proxy:
    - Kart λ ~ 4.2 * ref_factor * (1 + 0.07*(denge maçları)) 
    - Korner λ ~ 8.8 + 0.5*total_goal_mu + 2.0*(strength_ratio-1)
    """
    balance = 1.0 - clamp(abs(strength_ratio-1.0), 0.0, 0.5)  # ne kadar dengeli
    lam_cards = 4.2 * ref_factor * (1.0 + 0.07*balance)
    lam_corners = 8.8 + 0.5*total_goal_mu + 2.0*max(0.0, strength_ratio-1.0)
    return max(2.5, lam_cards), max(6.0, lam_corners)

def poisson_over_prob(lam, line):  # örn line=3.5
    # P(X > line) = 1 - P(X <= floor(line))
    k = int(math.floor(line))
    c = sum(poisson_pmf(lam, i) for i in range(0, k+1))
    return clamp(1.0 - c, 0.0, 1.0)

# =========================
# Model State (öğrenme)
# =========================
STATE_PATH = "model_state.json"

def load_state():
    if os.path.exists(STATE_PATH):
        try:
            return json.loads(open(STATE_PATH,"r",encoding="utf-8").read())
        except: pass
    return {"blend":{"w1":0.60,"w2":0.30,"w3":0.10},
            "leagues":{}}

def save_state(st):
    open(STATE_PATH,"w",encoding="utf-8").write(json.dumps(st,ensure_ascii=False,indent=2))

def lg_key(m):
    area = safe_get(m,"area","name",default=""); comp = safe_get(m,"competition","name",default="")
    return f"{area}:{comp}" if comp else area

def lp_for(st, key):
    d = st["leagues"].setdefault(key, {"mu_offset":0.0, "home_adv":1.10, "tau":1.00})
    return d

def apply_temperature(p, tau):
    p = clamp(p, 1e-3, 1-1e-3)
    z = math.log(p/(1-p))
    return 1.0/(1.0+math.exp(-z / clamp(tau,0.6,1.4)))

# =========================
# Core model (Poisson + blend)
# =========================
def match_probs_with_params(hs, as_, league_mu, home_adv):
    # basit gollü model
    lam_h = clamp(league_mu * hs["gf"]/max(0.6, as_["ga"]) * home_adv, 0.2, 4.5)
    lam_a = clamp(league_mu * as_["gf"]/max(0.6, hs["ga"]), 0.2, 4.5)
    p1=pX=p2=0.0
    for h in range(0,11):
        ph = poisson_pmf(lam_h,h)
        for a in range(0,11):
            pa = poisson_pmf(lam_a,a)
            if h>a: p1+=ph*pa
            elif h==a: pX+=ph*pa
            else: p2+=ph*pa
    return {"1":p1,"X":pX,"2":p2,"lam_home":lam_h,"lam_away":lam_a}

def blend_conf(p_model, p_market=None, st=None):
    if st is None: w1,w2 = 0.60,0.30
    else: w1, w2 = st["blend"]["w1"], st["blend"]["w2"]
    if p_market is None:
        w1, w2 = clamp(w1+0.15,0.6,0.9), 0.0
    z = w1*logit(p_model)
    if p_market is not None:
        z += w2*logit(p_market)
    return sigmoid(z)

# =========================
# PREDICT
# =========================
MEM_PATH = "mem_pending.json"
def load_mem():
    if os.path.exists(MEM_PATH):
        try: return json.loads(open(MEM_PATH,"r",encoding="utf-8").read())
        except: pass
    return {"pending":{}}

def save_mem(m): 
    open(MEM_PATH,"w",encoding="utf-8").write(json.dumps(m,ensure_ascii=False,indent=2))

def predict_today():
    d_tr = today_tr()
    subject = f"Günün Tahminleri | {d_tr.isoformat()}"
    ms = get_scheduled_matches(utc_now())
    if not ms:
        send_mail(subject, "Bugün için tahmin çıkarılacak maç bulunamadı.")
        return

    st = load_state()
    lines = [f"🗓 Tarih: {d_tr.isoformat()} — Başlamamış maçlar", ""]
    mem = load_mem()

    for m in sorted(ms, key=lambda x: x.get("utcDate","")):
        comp = safe_get(m,"competition","name",default="?")
        comp_id = safe_get(m,"competition","id",default=None)
        ht = safe_get(m,"homeTeam","name",default="?")
        at = safe_get(m,"awayTeam","name",default="?")
        hid = safe_get(m,"homeTeam","id",default=None)
        aid = safe_get(m,"awayTeam","id",default=None)
        when = m.get("utcDate")  # "2025-10-05T03:00:00Z"
        when_iso = when.replace("Z","")  # "YYYY-MM-DDTHH:MM:SS"
        # Lig parametreleri
        lk = lg_key(m)
        lp = lp_for(st, lk)
        base_mu = competition_goal_tempo(comp_id) if comp_id else 2.5

        # Hava etkisi (şehir tahmini: home team area/venue yoksa area adını kullan)
        city = safe_get(m,"area","name",default=None)
        weather_factor, weather_note = weather_effect_for_match(city, when_iso)

        mu_adj = base_mu * (1.0 + clamp(lp["mu_offset"], -0.2, 0.2)) * weather_factor

        # Formlar
        hs = team_recent_form(hid) if hid else {"gf":1.1,"ga":1.1}
        as_ = team_recent_form(aid) if aid else {"gf":1.1,"ga":1.1}
        probs = match_probs_with_params(hs, as_, mu_adj, lp["home_adv"])

        # Model pick
        pick = max(("1","X","2"), key=lambda k: probs[k])
        p_model = probs[pick]
        p_model_cal = apply_temperature(p_model, lp["tau"])

        # Market (opsiyonel)
        p_market_pick = None
        if ODDS_KEY:
            market = odds_search_match(ht, at, when_iso)
            if market:
                p_market_pick = market.get(pick)

        p_final = blend_conf(p_model_cal, p_market_pick, st)

        # Güç oranı (dominance)
        strength_ratio = max(probs["1"], probs["2"]) / max(1e-6, min(probs["1"], probs["2"]))
        # Hakem proxy -> kart/korner
        ref_factor = referee_proxy(m)
        lam_cards, lam_corners = cards_and_corners_lambda(probs["lam_home"]+probs["lam_away"], strength_ratio, ref_factor)
        card_o35 = poisson_over_prob(lam_cards, 3.5)
        card_o45 = poisson_over_prob(lam_cards, 4.5)
        cor_o85  = poisson_over_prob(lam_corners, 8.5)
        cor_o95  = poisson_over_prob(lam_corners, 9.5)

        # Satır
        conf_pct = int(round(100*p_final))
        why = []
        why.append(f"lig μ≈{base_mu:.2f}→{mu_adj:.2f}")
        if weather_note: why.append(f"hava({weather_note})")
        why.append(f"form(H:{hs['gf']:.2f}/{hs['ga']:.2f},A:{as_['gf']:.2f}/{as_['ga']:.2f})")
        if p_market_pick is not None: why.append("piyasa karışımı")
        why.append(f"kartλ≈{lam_cards:.1f}, kornerλ≈{lam_corners:.1f}")

        line = f"• {when[11:16]} | {ht} vs {at} — Tahmin: {pick} (güven %{conf_pct}) | {comp}"
        lines.append(line)
        lines.append("   ↳ Nedenler: " + "; ".join(why))
        lines.append(f"   ↳ Kart O3.5 %{int(card_o35*100)}, O4.5 %{int(card_o45*100)} | Korner O8.5 %{int(cor_o85*100)}, O9.5 %{int(cor_o95*100)}")
        lines.append("")

        # pending hafıza
        mid = str(m.get("id"))
        mem["pending"][mid] = {
            "utcDate": when, "home": ht, "away": at,
            "side": pick, "p_model": p_model, "p_final": p_final,
            "lam_h": probs["lam_home"], "lam_a": probs["lam_away"]
        }

    save_mem(mem)
    body = "\n".join(lines).strip()
    send_mail(subject, body)

# =========================
# RESULTS + micro-learn
# =========================
def why_wrong(pick, lam_h, lam_a, hg, ag):
    total = lam_h + lam_a
    t_act = hg + ag
    parts = []
    if t_act < total - 0.8: parts.append("tempo beklenenin altı")
    if (pick=="1" and ag>hg) or (pick=="2" and hg>ag): parts.append("denge/ev-saha yanılgısı")
    if not parts: parts.append("varyans/şans")
    return ", ".join(parts)

def micro_learn(st, match, pick, won, lam_h, lam_a):
    lk = lg_key(match)
    lp = lp_for(st, lk)
    lr_mu, lr_home, lr_tau = 0.01, 0.02, 0.02
    ft = safe_get(match,"score","fullTime",default={})
    hg, ag = (ft.get("home",0) or 0), (ft.get("away",0) or 0)
    total = hg + ag; exp_tot = lam_h + lam_a

    # tempo hatası
    if not won and (total < exp_tot - 0.8):
        lp["mu_offset"] = clamp(lp["mu_offset"] - lr_mu, -0.2, 0.2)
    # ev-saha/dengesizlik
    if not won and pick=="1" and ag>hg:
        lp["home_adv"] = clamp(lp["home_adv"] - lr_home, 1.00, 1.25)
    if not won and pick=="2" and hg>ag:
        lp["home_adv"] = clamp(lp["home_adv"] + lr_home, 1.00, 1.25)
    # aşırı güven → yumuşat
    if not won: lp["tau"] = clamp(lp["tau"] + lr_tau, 0.80, 1.40)
    else:       lp["tau"] = clamp(lp["tau"] - lr_tau/2, 0.80, 1.40)

def results_today():
    d_tr = today_tr()
    subject = f"Günün Sonuçları | {d_tr.isoformat()}"
    ms = get_finished_matches(utc_now())
    if not ms:
        send_mail(subject, "Bugün için maç bulunamadı.")
        return
    mem = load_mem()
    st = load_state()

    lines = [f"📊 Günün Sonuçları — {d_tr.isoformat()}", ""]
    correct=wrong=0

    for m in sorted(ms, key=lambda x: x.get("utcDate","")):
        mid = str(m.get("id"))
        if mid not in mem["pending"]:
            # sistem dışı maç (sorun değil)
            continue
        entry = mem["pending"].pop(mid)
        ht, at = entry["home"], entry["away"]
        pick = entry["side"]
        lam_h, lam_a = entry["lam_h"], entry["lam_a"]

        ft = safe_get(m,"score","fullTime",default={})
        hg, ag = (ft.get("home",0) or 0), (ft.get("away",0) or 0)
        won = (pick=="1" and hg>ag) or (pick=="2" and ag>hg) or (pick=="X" and hg==ag)
        if won: correct+=1
        else: wrong+=1

        # öğrenme
        micro_learn(st, m, pick, won, lam_h, lam_a)

        status = "✅ tuttu" if won else "❌ tutmadı"
        reason = "" if won else (" — Neden: " + why_wrong(pick, lam_h, lam_a, hg, ag))
        lines.append(f"— {ht} {hg}-{ag} {at} → Seçim: {pick} → {status}{reason}")

    save_mem(mem)
    save_state(st)
    lines.append("")
    lines.append(f"Özet: {correct} doğru / {wrong} yanlış")
    send_mail(subject, "\n".join(lines).strip())

# =========================
# AUTO / ENTRY
# =========================
def main():
    if MODE == "PREDICT":
        predict_today()
        return
    if MODE == "RESULTS":
        results_today()
        return
    # AUTO: saat kontrolü (UTC 07:00 ≈ TR 10:00; UTC 20:59 ≈ TR 23:59)
    hour_utc = int(datetime.utcnow().strftime("%H"))
    if hour_utc == 7:
        predict_today()
    elif hour_utc == 20:
        results_today()
    else:
        send_mail("Tahmin Botu | Bilgi", "AUTO modu dışı çalıştırma. MODE=PREDICT veya MODE=RESULTS bekleniyor.")

if __name__ == "__main__":
    main()
