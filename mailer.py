import os, json, math, time, statistics as stats
from datetime import datetime, timedelta, timezone
import smtplib, ssl, subprocess
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import requests

# ========= ENV / SECRETS =========
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_PASS")
GMAIL_TO   = os.getenv("GMAIL_TO")
FD_TOKEN   = os.getenv("FOOTBALL_DATA_TOKEN")  # football-data.org
ODDS_KEY   = os.getenv("ODDS_API_KEY", "")     # The Odds API (opsiyonel)
MODE       = os.getenv("MODE", "AUTO").upper() # AUTO / PREDICT / RESULTS

# ========= SABİTLER =========
TR_TZ   = timezone(timedelta(hours=3))  # Türkiye
TODAY_TR= datetime.now(TR_TZ).date()
TODAY_UTC = datetime.utcnow().date()

STATE_FILE = "model_state.json"  # repo kökünde
TOP_N_PICKS = 12                 # Risk filtresinden sonra en iyi N
FORM_N = 5                       # Son N maç form penceresi
HOME_ADV = 1.10                  # İç saha çarpanı
BLEND_W_MARKET = 0.25            # Pazar (odds) karışım ağırlığı
MIN_CONF_TO_LIST = 0.60          # %60 üstü aday göster
HI_CONF_FLAG = 0.90              # %90 üstü ⚡
ODDS_SPORT_KEYS = [
    "soccer_epl","soccer_spain_la_liga","soccer_italy_serie_a",
    "soccer_france_ligue_one","soccer_germany_bundesliga",
    "soccer_turkey_super_league","soccer_brazil_campeonato",
    "soccer_netherlands_eredivisie","soccer_portugal_primeira_liga",
    "soccer_uefa_champs_league","soccer_uefa_europa_league",
    "soccer_usa_mls"
]

# Bazı ülke başkent koordinatları (Open-Meteo için basit proxy)
COUNTRY_COORDS = {
    "England": (51.5074, -0.1278),
    "United Kingdom": (51.5074, -0.1278),
    "Spain": (40.4168, -3.7038),
    "Italy": (41.9028, 12.4964),
    "France": (48.8566, 2.3522),
    "Germany": (52.52, 13.4050),
    "Türkiye": (39.9208, 32.8541),
    "Turkey": (39.9208, 32.8541),
    "Brazil": ( -15.793889, -47.882778),
    "Netherlands": (52.3676, 4.9041),
    "Portugal": (38.7223, -9.1393),
    "USA": (38.9072, -77.0369)
}

# ========= YARDIMCILAR =========
def send_mail(subject, body):
    assert GMAIL_USER and GMAIL_PASS and GMAIL_TO, "GMAIL_* secrets eksik."
    msg = MIMEMultipart()
    msg["From"] = GMAIL_USER
    msg["To"] = GMAIL_TO
    msg["Subject"] = subject
    msg.attach(MIMEText(body, "plain", "utf-8"))

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
        server.login(GMAIL_USER, GMAIL_PASS)
        server.sendmail(GMAIL_USER, [GMAIL_TO], msg.as_string())

def tr_today_str():
    return datetime.now(TR_TZ).strftime("%Y-%m-%d")

def date_str(d):
    return d.strftime("%Y-%m-%d")

def load_state():
    try:
        with open(STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

def save_state(state, commit_message="Update model_state.json"):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    # Actions içinde commit/push
    try:
        subprocess.run(["git","config","user.email","bot@github.actions"], check=True)
        subprocess.run(["git","config","user.name","Actions Bot"], check=True)
        subprocess.run(["git","add", STATE_FILE], check=True)
        subprocess.run(["git","commit","-m", commit_message], check=True)
        subprocess.run(["git","push"], check=True)
    except Exception as e:
        print("Commit/push atlanıyor:", e)

# ========= DATA KATMANI =========
def fd_headers():
    assert FD_TOKEN, "FOOTBALL_DATA_TOKEN eksik."
    return {"X-Auth-Token": FD_TOKEN}

def fd_get(url, params=None):
    r = requests.get(url, headers=fd_headers(), params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()

def get_day_matches(date_utc:datetime.date):
    """ Günün tüm maçları (UTC günü) """
    url = "https://api.football-data.org/v4/matches"
    params = {"dateFrom": date_str(date_utc), "dateTo": date_str(date_utc)}
    js = fd_get(url, params)
    return js.get("matches", [])

def recent_matches_for_team(team_id:int, days_back=65, limit=30):
    """ Takımın son bitmiş maçları (form için) """
    since = datetime.utcnow().date() - timedelta(days=days_back)
    url = f"https://api.football-data.org/v4/teams/{team_id}/matches"
    params = {"dateFrom": date_str(since), "dateTo": date_str(datetime.utcnow().date())}
    try:
        js = fd_get(url, params)
        allm = js.get("matches", [])
    except:
        return []
    # FINISHED filtrele, tarihe göre sırala
    fins = [m for m in allm if m.get("status")=="FINISHED"]
    fins.sort(key=lambda m: m.get("utcDate",""))
    return fins[-limit:]

def league_goal_mean_recent(area_name:str, days_back=35):
    """ Lig/ülke bazında basit gol ort. (aynı ülkedeki maçlardan proxy) """
    # football-data org doğrudan area filtresi sunmuyor → günün tüm maçları proxy yaklaşımı
    tot_goals, cnt = 0, 0
    for d in range(days_back):
        day = datetime.utcnow().date()-timedelta(days=d+1)
        matches = get_day_matches(day)
        for m in matches:
            comp_area = (m.get("area") or {}).get("name","")
            if comp_area == area_name and m.get("status")=="FINISHED":
                score = m.get("score",{})
                ft = (score.get("fullTime") or {})
                if ft.get("home") is not None and ft.get("away") is not None:
                    tot_goals += ft["home"] + ft["away"]
                    cnt += 1
        # API rate limitini üzmemek adına mini uyku:
        time.sleep(0.3)
    return (tot_goals / cnt) if cnt>5 else 2.6  # default

def form_goals_lambda(team_id:int, is_home:bool, league_mean:float):
    """ Son FORM_N maçtan GF/GA → basit λ tahmini (Poisson) """
    fins = recent_matches_for_team(team_id)
    if not fins:
        # veri yoksa lig ort. ve iç saha bonusu
        base = league_mean * (1.05 if is_home else 0.95)
        return max(0.3, base/2.0)
    last = fins[-FORM_N:]
    gf, ga = 0, 0
    for m in last:
        ht = m["homeTeam"]["id"]
        score = m.get("score",{}).get("fullTime",{})
        h, a = score.get("home"), score.get("away")
        if h is None or a is None: 
            continue
        if team_id == ht: 
            gf += h; ga += a
        else:
            gf += a; ga += h
    n = max(1, len(last))
    att = gf/n
    defn = ga/n
    # iç saha bonusu + lig uyumu
    lam = max(0.2, att * (HOME_ADV if is_home else 1.0) * (league_mean/2.6))
    return lam

def poisson_prob_home_draw_away(lh, la, max_goals=10):
    """ Poisson skorlardan 1-X-2 olasılıklarını yaklaşıkla """
    from math import exp, factorial
    def P(k, lam): 
        return (lam**k) * math.exp(-lam) / math.factorial(k)
    p_home=p_draw=p_away=0.0
    for i in range(max_goals+1):
        pi = P(i, lh)
        for j in range(max_goals+1):
            pj = P(j, la)
            if i>j: p_home += pi*pj
            elif i==j: p_draw += pi*pj
            else: p_away += pi*pj
    return p_home, p_draw, p_away

def open_meteo_adjust(area_name:str, utc_iso:str):
    """ Ülke merkezine göre rüzgar/yağış → tempo/gol düzeltmesi (-10% .. +5%) """
    latlon = COUNTRY_COORDS.get(area_name)
    if not latlon:
        return 1.0, {}  # düzeltme yok
    lat, lon = latlon
    # maça en yakın saat (UTC)
    t = datetime.fromisoformat(utc_iso.replace("Z","+00:00"))
    hour = int(t.strftime("%H"))
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "hourly":"precipitation,windspeed_10m",
        "start_hour":0, "forecast_days":2, "timezone":"UTC"
    }
    try:
        js = requests.get(url, params=params, timeout=20).json()
        times = js.get("hourly",{}).get("time",[])
        precs = js.get("hourly",{}).get("precipitation",[])
        winds = js.get("hourly",{}).get("windspeed_10m",[])
        adj = 1.0
        meta = {}
        if times:
            # en yakın saat index’i
            # (liste bugün+yarın olabilir; basitçe saat eşleşmesi)
            candidates = []
            for idx, ts in enumerate(times):
                try:
                    tt = datetime.fromisoformat(ts.replace("Z","+00:00"))
                    candidates.append((abs((tt - t).total_seconds()), idx))
                except: pass
            if candidates:
                _, k = min(candidates)
                p = precs[k] if k<len(precs) else 0.0
                w = winds[k] if k<len(winds) else 0.0
                # rüzgar/yağış etkisi
                if w >= 8:   adj *= 0.92
                if w >= 12:  adj *= 0.88
                if p >= 1.5: adj *= 0.94
                if p >= 3.0: adj *= 0.90
                meta = {"wind":w, "rain":p}
        return adj, meta
    except:
        return 1.0, {}

def fetch_odds_h2h(home, away, utc_iso):
    """ The Odds API: 1X2 “h2h” market (EU bölgesi). Sport key eşleşirse döner. """
    if not ODDS_KEY:
        return None
    ts = int(datetime.fromisoformat(utc_iso.replace("Z","+00:00")).timestamp())
    for sk in ODDS_SPORT_KEYS:
        try:
            url = f"https://api.the-odds-api.com/v4/sports/{sk}/odds"
            params = {"apiKey":ODDS_KEY,"markets":"h2h","regions":"eu","oddsFormat":"decimal"}
            js = requests.get(url, params=params, timeout=30).json()
            if not isinstance(js,list): 
                continue
            # Eşleşen etkinlik bul
            for ev in js:
                # basit string eşlemesi (casefold)
                hn = (ev.get("home_team") or "").casefold()
                an = (ev.get("away_team") or "").casefold()
                if home.casefold() in hn and away.casefold() in an:
                    # outcome to pazar olasılığı
                    best = None
                    for bm in ev.get("bookmakers",[]):
                        for mk in bm.get("markets",[]):
                            if mk.get("key")=="h2h":
                                outs = mk.get("outcomes",[])
                                # 1X2 sırala
                                odds_map = {}
                                for o in outs:
                                    name = o.get("name","").lower()
                                    price = float(o.get("price",0))
                                    if price>1.01:
                                        odds_map[name]=price
                                if not odds_map: 
                                    continue
                                # piyasa olasılıkları (marjlı)
                                inv = {}
                                for k,v in odds_map.items():
                                    inv[k] = 1.0/v
                                s = sum(inv.values())
                                market = {k:inv[k]/s for k in inv}
                                # standardize isimler
                                ph = market.get(home.lower(), None) or market.get("home", None)
                                pd = market.get("draw", None) or market.get("tie", None)
                                pa = market.get(away.lower(), None) or market.get("away", None)
                                if ph and pd and pa:
                                    best = {"pH":ph,"pD":pd,"pA":pa}
                    if best:
                        return best
        except:
            continue
    return None

# Basit hakem/kart/korner proxy λ:
def tempo_proxies(league_mean, lam_h, lam_a):
    """ 
    Korner λ ve Kart λ için ücretsiz veriden basit yaklaşım.
    - Korner: gol beklentisi yükseldikçe artar
    - Kart: düşük lig gol temposu + yakın güç dengesi -> kart olasılığı artar
    """
    exp_goals = lam_h + lam_a
    corner_lambda = 7.5 * (exp_goals/2.6)  # ~ 8-9 korner normu
    # denge katsayısı: yakın güçte takımlar → kart ihtimali artar
    balance = 1.0 - min(0.9, abs(lam_h - lam_a)/max(0.4, lam_h+lam_a))
    card_lambda = 4.2 * ( (2.6/league_mean) * 0.6 + balance * 0.4 )
    # mantıklı sınırlar
    return max(5.0, min(12.0, corner_lambda)), max(3.0, min(6.5, card_lambda))

# ========= PREDICT =========
def run_predict():
    matches = get_day_matches(TODAY_UTC)
    upcoming = [m for m in matches if m.get("status") in ("TIMED","SCHEDULED")]
    if not upcoming:
        send_mail(f"Günün Tahminleri | {tr_today_str()}", "Bugün için tahmin çıkarılacak maç bulunamadı.")
        return

    picks = []
    league_cache = {}

    for m in sorted(upcoming, key=lambda x: x.get("utcDate","")):
        home = m["homeTeam"]["name"]
        away = m["awayTeam"]["name"]
        hid  = m["homeTeam"]["id"]
        aid  = m["awayTeam"]["id"]
        area = (m.get("area") or {}).get("name","")
        comp = (m.get("competition") or {}).get("name","")
        utc_iso = m.get("utcDate")

        # Lig gol ort.
        if area not in league_cache:
            league_cache[area] = league_goal_mean_recent(area)
        lmean = league_cache[area]

        # Form λ
        lam_h = form_goals_lambda(hid, True,  lmean)
        lam_a = form_goals_lambda(aid, False, lmean)

        # Hava düzeltmesi
        w_adj, w_meta = open_meteo_adjust(area, utc_iso)
        lam_h *= w_adj
        lam_a *= w_adj

        # Poisson 1X2
        pH, pD, pA = poisson_prob_home_draw_away(lam_h, lam_a)

        # Pazar/odds (varsa) → karışım
        market = fetch_odds_h2h(home, away, utc_iso)
        if market:
            pH = (1-BLEND_W_MARKET)*pH + BLEND_W_MARKET*market["pH"]
            pD = (1-BLEND_W_MARKET)*pD + BLEND_W_MARKET*market["pD"]
            pA = (1-BLEND_W_MARKET)*pA + BLEND_W_MARKET*market["pA"]

        # Nihai seçim + güven
        probs = {"1":pH, "X":pD, "2":pA}
        pick = max(probs, key=probs.get)
        conf = probs[pick]

        # Korner/Kart öneri λ
        corner_lam, card_lam = tempo_proxies(lmean, lam_h, lam_a)
        corner_sug = "Korner Üst 8.5" if corner_lam >= 9.0 else ("Korner Alt 9.5" if corner_lam <= 7.5 else None)
        card_sug   = "Kart Üst 4.5"   if card_lam   >= 4.7 else ("Kart Alt 4.5"   if card_lam   <= 3.8 else None)

        # Kısa açıklama
        reasons = []
        reasons.append(f"Form λ: {lam_h:.2f}-{lam_a:.2f} (Lig {lmean:.2f})")
        if w_meta:
            wtxt=[]
            if "wind" in w_meta: wtxt.append(f"rüzgâr {w_meta['wind']:.1f} m/s")
            if "rain" in w_meta: wtxt.append(f"yağış {w_meta['rain']:.1f} mm/h")
            reasons.append("Hava: " + ", ".join(wtxt) + (f" → x{w_adj:.2f}" if abs(w_adj-1)>0.01 else "")) 
        if market:
            reasons.append("Piyasa blend: ev/ber/depl = " +
                           f"{market['pH']:.2f}/{market['pD']:.2f}/{market['pA']:.2f}")
        if corner_sug: reasons.append(corner_sug)
        if card_sug:   reasons.append(card_sug)

        picks.append({
            "matchId": m["id"],
            "utc": utc_iso,
            "timeTR": datetime.fromisoformat(utc_iso.replace("Z","+00:00")).astimezone(TR_TZ).strftime("%H:%M"),
            "area": area, "comp": comp,
            "home": home, "away": away,
            "pick": pick, "conf": conf,
            "reasons": reasons[:4],
            "corner": corner_sug, "card": card_sug
        })

        # Rate limit nazik kal
        time.sleep(0.3)

    # Risk filtresi: lig/maç bazlı çoklu korele seçimleri kırp
    # (aynı maçtan sadece en yüksek güvenli 1 pick)
    by_match = {}
    for p in picks:
        key = p["matchId"]
        if key not in by_match or p["conf"] > by_match[key]["conf"]:
            by_match[key] = p
    dedup = list(by_match.values())
    # En iyi N
    dedup.sort(key=lambda x: x["conf"], reverse=True)
    final = dedup[:TOP_N_PICKS]

    # Mail gövdesi
    lines = []
    lines.append(f"📅 Tarih: {tr_today_str()} — Başlamamış maçlar")
    hi_lines = []
    for p in final:
        tag = " ⚡" if p["conf"]>=HI_CONF_FLAG else ""
        line = f"• {p['timeTR']} | {p['home']} vs {p['away']} — Tahmin: {p['pick']} (güven %{int(p['conf']*100)}){tag}"
        line += f" | {p['comp']} | lig: {p['area']}"
        lines.append(line)
        if p["reasons"]:
            lines.append("   - " + " | ".join(p["reasons"]))
        if p["corner"] or p["card"]:
            side = "   - Yan pazar: " + " / ".join([x for x in [p["corner"], p["card"]] if x])
            lines.append(side)

        if tag:
            hi_lines.append(line)

    body = "\n".join(lines) if final else "Bugün için tahmin çıkarılacak maç bulunamadı."

    # state’e yaz (gece sonuçta ölçmek için)
    state = load_state()
    state.setdefault(date_str(TODAY_UTC), {})
    # sade bir kayıt
    state[date_str(TODAY_UTC)] = {
        "generated_at": datetime.utcnow().isoformat()+"Z",
        "picks": {str(p["matchId"]): {"pick":p["pick"], "conf":p["conf"],
                                      "home":p["home"], "away":p["away"]} for p in final}
    }
    save_state(state, f"Predictions for {date_str(TODAY_UTC)}")

    sub = f"Günün Tahminleri | {tr_today_str()}"
    send_mail(sub, body)

    if hi_lines:
        send_mail(f"⚡ Yüksek Güven | {tr_today_str()}", "\n".join(hi_lines))

# ========= RESULTS =========
def run_results():
    matches = get_day_matches(TODAY_UTC)
    finished = [m for m in matches if m.get("status")=="FINISHED"]

    if not finished:
        send_mail(f"Günün Sonuçları | {tr_today_str()}", "Bugün için maç bulunamadı.")
        return

    state = load_state().get(date_str(TODAY_UTC), {}).get("picks", {})
    hits=miss=0
    lines = [f"📅 Tarih: {tr_today_str()} — Bitmiş maçlar"]

    for m in sorted(finished, key=lambda x: x.get("utcDate","")):
        mid = str(m["id"])
        score = (m.get("score") or {}).get("fullTime") or {}
        h, a = score.get("home"), score.get("away")
        home = m["homeTeam"]["name"]; away = m["awayTeam"]["name"]
        if h is None or a is None:
            continue
        # Gerçek sonuç 1X2
        real = "1" if h>a else ("2" if a>h else "X")
        pick = state.get(mid,{}).get("pick")
        ok = (pick == real)
        if pick:
            hits += 1 if ok else 0
            miss += 0 if ok else 1
        lines.append(f"• {home} {h}-{a} {away} — {('✅ TUTTU' if ok else '❌ KAÇTI') if pick else '– (tahmin yok)'}")

        # nazik rate control
        time.sleep(0.1)

    summary = f"\nÖzet: Doğru = {hits}, Yanlış = {miss}"
    body = "\n".join(lines) + summary

    # mini öğrenme: başarı oranına göre market ağırlığı mikro ayar
    global BLEND_W_MARKET
    total = hits+miss
    if total>=6:
        acc = hits/total
        if acc>=0.62 and BLEND_W_MARKET>0.15:
            BLEND_W_MARKET = round(BLEND_W_MARKET - 0.02, 2)
        elif acc<=0.48 and BLEND_W_MARKET<0.40:
            BLEND_W_MARKET = round(BLEND_W_MARKET + 0.02, 2)
        # state’e yaz
        st = load_state()
        st.setdefault("meta",{})["blend_w_market"] = BLEND_W_MARKET
        save_state(st, f"Learn blend after results {date_str(TODAY_UTC)}")

    send_mail(f"Günün Sonuçları | {tr_today_str()}", body)

# ========= ANA AKIŞ =========
def main():
    if MODE == "AUTO":
        # UTC 07:00 ~ TR 10:00 (tahmin), UTC 20:59 ~ TR 23:59 (sonuç)
        hour = datetime.utcnow().hour
        if hour == 7:
            run_predict()
        elif hour == 20:
            run_results()
        else:
            # Güvenli tarafta ol: Bot dışı çağrıldıysa bilgi geç
            send_mail(f"Tahmin Botu | Bilgi", "AUTO modu dışı çalıştırma. MODE=PREDICT veya MODE=RESULTS bekleniyor.")
    elif MODE == "PREDICT":
        run_predict()
    elif MODE == "RESULTS":
        run_results()
    else:
        send_mail("Tahmin Botu | Bilgi", f"Bilinmeyen MODE: {MODE}")

if __name__ == "__main__":
    main()
