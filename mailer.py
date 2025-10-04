import os, sys, math, json, datetime as dt, requests, pathlib, smtplib
from email.mime.text import MIMEText
from pathlib import Path

# --------- Secrets / Config ----------
GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_PASS = os.environ.get("GMAIL_PASS")
GMAIL_TO   = os.environ.get("GMAIL_TO")
FD_TOKEN   = os.environ.get("FOOTBALL_DATA_TOKEN")     # football-data.org (ücretsiz)
ODDS_API_KEY = os.environ.get("ODDS_API_KEY")          # opsiyonel (free tier varsa)

TZ = dt.timezone(dt.timedelta(hours=3))  # TR
TODAY = dt.datetime.now(TZ).date()

EU_COUNTRIES = {
    "Turkey","England","Spain","Germany","Italy","France","Portugal",
    "Netherlands","Belgium","Scotland","Denmark","Austria","Switzerland",
    "Norway","Sweden","Poland","Czech Republic","Greece","Croatia",
    "Serbia","Romania","Ukraine","Hungary"
}

FD_HEADERS = {"X-Auth-Token": FD_TOKEN} if FD_TOKEN else {}

# --------- IO helpers ----------
def send_mail(subject, body):
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = GMAIL_TO
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(GMAIL_USER, GMAIL_PASS)
        s.sendmail(GMAIL_USER, [GMAIL_TO], msg.as_string())

def fd(url, params=None):
    r = requests.get("https://api.football-data.org/v4"+url, params=params, headers=FD_HEADERS, timeout=30)
    r.raise_for_status()
    return r.json()

def get_matches(date_from, date_to):
    data = fd("/matches", {"dateFrom": str(date_from), "dateTo": str(date_to)})
    out = []
    for m in data.get("matches", []):
        area = (m.get("area") or {}).get("name","")
        if area in EU_COUNTRIES:
            out.append(m)
    return out

def finished_today():
    ms = get_matches(TODAY, TODAY)
    return [m for m in ms if m.get("status")=="FINISHED"]

def todays_scheduled():
    ms = get_matches(TODAY, TODAY)
    return [m for m in ms if m.get("status") in ("SCHEDULED","TIMED")]

# --------- Team form / league tempo ----------
def team_form_rating(team_id, last_n=10):
    data = fd("/matches", {"limit":100, "dateTo": str(TODAY), "dateFrom": str(TODAY - dt.timedelta(days=120))})
    tms = []
    for m in data.get("matches", []):
        if m.get("status") not in ("FINISHED","AWARDED"): continue
        if m["homeTeam"]["id"] == team_id or m["awayTeam"]["id"] == team_id:
            tms.append(m)
    tms = sorted(tms, key=lambda x: x["utcDate"], reverse=True)[:last_n]
    if not tms: return {"pts_per_game":1.2, "gf":1.1, "ga":1.1}
    pts=gf=ga=0
    for m in tms:
        ft = (m.get("score") or {}).get("fullTime",{})
        hs, as_ = ft.get("home",0) or 0, ft.get("away",0) or 0
        is_home = (m["homeTeam"]["id"] == team_id)
        g_for  = hs if is_home else as_
        g_ag   = as_ if is_home else hs
        gf += g_for; ga += g_ag
        if hs==as_: pts+=1
        elif (is_home and hs>as_) or ((not is_home) and as_>hs): pts+=3
    n = max(1,len(tms))
    return {"pts_per_game": pts/n, "gf": gf/n, "ga": ga/n}

def league_goal_tempo():
    data = fd("/matches", {"limit":100, "dateTo": str(TODAY), "dateFrom": str(TODAY - dt.timedelta(days=60))})
    tot=n=0
    for m in data.get("matches", []):
        if m.get("status") not in ("FINISHED","AWARDED"): continue
        ft = (m.get("score") or {}).get("fullTime",{})
        if ft:
            g=(ft.get("home",0) or 0)+(ft.get("away",0) or 0)
            tot+=g; n+=1
    return (tot/n) if n else 2.6

# --------- Poisson core ----------
def poisson_pmf(lmbd,k): return math.exp(-lmbd)*(lmbd**k)/math.factorial(k)
def poisson_cdf(lmbd,k): return sum(poisson_pmf(lmbd,i) for i in range(0,k+1))

def match_probs_with_params(hs, as_, league_mu, home_adv):
    lam_h = max(0.2, league_mu * hs["gf"]/max(0.6, as_["ga"]) * home_adv)
    lam_a = max(0.2, league_mu * as_["gf"]/max(0.6, hs["ga"]))
    p1=pX=p2=0.0
    for h in range(0,11):
        ph=poisson_pmf(lam_h,h)
        for a in range(0,11):
            pa=poisson_pmf(lam_a,a)
            if   h>a: p1+=ph*pa
            elif h==a: pX+=ph*pa
            else: p2+=ph*pa
    return {"1":p1,"X":pX,"2":p2,"lam_home":lam_h,"lam_away":lam_a}

# --------- Cards & Corners (heuristic, free) ----------
def league_cards_mu():   return 4.2
def league_corners_mu(): return 9.6

def intensity_factor(hs,as_):
    gap=abs(hs["pts_per_game"]-as_["pts_per_game"])
    return 1.15 if gap<0.3 else (1.05 if gap<0.6 else 0.95)

def est_cards_lambda(hs,as_):
    base=league_cards_mu(); f=intensity_factor(hs,as_)
    lam_goals=(hs["gf"]+as_["gf"])/max(1.0,(hs["ga"]+as_["ga"])/2.0)
    tempo=1.0+min(0.15,max(-0.1,(lam_goals-2.5)*0.08))
    return max(2.8, base*f*tempo)

def est_corners_lambda(lh,la,hs,as_):
    base=league_corners_mu()
    g_tot=lh+la
    tempo=1.0+min(0.25,max(-0.2,(g_tot-2.6)*0.25))
    gap=hs["pts_per_game"]-as_["pts_per_game"]
    press=1.0+max(-0.1,min(0.15,gap*0.12))
    return max(6.5, base*tempo*press)

def over_prob(lmbd,line): return 1.0 - poisson_cdf(lmbd,int(math.floor(line)))

# --------- Odds → market prob + memory (free) ----------
MEM_PATH = pathlib.Path("market_memory.json")
STATE_PATH = Path("model_state.json")

def load_mem():
    if MEM_PATH.exists():
        try: return json.loads(MEM_PATH.read_text(encoding="utf-8"))
        except: pass
    return {"price_buckets": {}, "prob_buckets": {}, "pending": {}}

def save_mem(mem): MEM_PATH.write_text(json.dumps(mem,ensure_ascii=False,indent=2),encoding="utf-8")

def _default_state():
    return {"blend": {"w1": 0.55, "w2": 0.35, "w3": 0.10},"leagues": {}}

def load_state():
    if STATE_PATH.exists():
        try: return json.loads(STATE_PATH.read_text(encoding="utf-8"))
        except: pass
    return _default_state()

def save_state(st): STATE_PATH.write_text(json.dumps(st,ensure_ascii=False,indent=2),encoding="utf-8")

def league_key(match):
    area = (match.get("area") or {}).get("name", "")
    comp = (match.get("competition") or {}).get("name", "")
    return f"{area}:{comp}" if comp else area

def league_params(st, key):
    d = st["leagues"].setdefault(key, {"mu_offset": 0.0,"home_adv": 1.10,"tau": 1.00})
    return d

def apply_temperature(p, tau):
    p = max(1e-3, min(1-1e-3, p))
    z = math.log(p/(1-p))
    return 1.0/(1.0+math.exp(-z / max(0.5, min(1.5, tau))))

def normalize_market_probs(o1,ox,o2):
    inv=[1.0/o1,1.0/ox,1.0/o2]; s=sum(inv); return [x/s for x in inv]  # [p1,pX,p2]

def fetch_current_odds(home, away):
    if not ODDS_API_KEY: return None
    try:
        # Ücretsiz odds endpointini ekleyince burayı doldur
        return None
    except Exception:
        return None

def logit(p): p=min(0.999,max(0.001,p)); return math.log(p/(1-p))
def sigmoid(z): return 1/(1+math.exp(-z))

def blend_conf(p_model, p_market=None, hist_wr=None, hist_n=0, st=None):
    if st is None:
        w1,w2,w3=0.55,0.35,0.10
    else:
        w1,w2,w3 = st["blend"]["w1"], st["blend"]["w2"], st["blend"]["w3"]
    if p_market is None:
        w1,w2,w3 = max(0.7,w1), 0.0, min(0.3, w3+0.1)
    if hist_n>=20: w3 += 0.05
    if hist_n>=50: w3 += 0.10
    z = logit(p_model)*w1
    if p_market is not None: z += logit(p_market)*w2
    if hist_wr    is not None: z += logit(hist_wr)*w3
    return sigmoid(z)

def explain_pick(best, probs, hs, as_, league_mu):
    why=[]
    if hs["pts_per_game"]>as_["pts_per_game"]+0.4: why.append("form avantajı")
    if hs["gf"]>as_["ga"]*1.1 and best=="1": why.append("hücum üstünlüğü")
    if league_mu>2.8: why.append("lig gol temposu yüksek")
    if best=="1": why.append("iç saha etkisi")
    return ", ".join(why) or "genel üstünlük"

def why_wrong(best, lam_h, lam_a, hg, ag):
    reasons=[]
    total = (hg or 0) + (ag or 0)
    if best in ("1","2"):
        fav_lam = lam_h if best=="1" else lam_a
        dog_lam = lam_a if best=="1" else lam_h
        if total <= 2: reasons.append("beklenen tempodan düşük skor")
        if dog_lam >= fav_lam*0.9: reasons.append("maç beklenenden dengeli")
        if (best=="1" and ag>hg) or (best=="2" and hg>ag): reasons.append("verimlilik tersine döndü")
    elif best=="X":
        if total >=3: reasons.append("yüksek tempo → beraberlik olasılığı düştü")
    return ", ".join(reasons) or "varyans / maç içi faktörler"

# --------- Reports ----------
def build_prediction_report(matches):
    if not matches: return "Bugün için maç bulunamadı."
    mem = load_mem()
    st  = load_state()
    league_mu_base = league_goal_tempo()
    lines=[f"Günün Tahminleri — {TODAY.isoformat()}\n"]

    for m in sorted(matches, key=lambda x: x.get("utcDate")):
        ht,at = m["homeTeam"], m["awayTeam"]
        hs,as_ = team_form_rating(ht["id"]), team_form_rating(at["id"])
        lk = league_key(m)
        lp = league_params(st, lk)

        mu_adj = league_mu_base * (1.0 + max(-0.2, min(0.2, lp["mu_offset"])))
        probs = match_probs_with_params(hs,as_,mu_adj, lp["home_adv"])

        pick = max(("1","X","2"), key=lambda k: probs[k])
        p_model_raw = probs[pick]
        p_model = apply_temperature(p_model_raw, lp["tau"])

        odds = fetch_current_odds(ht["name"], at["name"])
        p_market = None
        if odds:
            p1,pX,p2 = normalize_market_probs(odds["1"], odds["X"], odds["2"])
            p_market = {"1":p1,"X":pX,"2":p2}[pick]

        hist_wr=None; hist_n=0; hist_line=""; key=""
        if odds and pick in ("1","2"):
            chosen = odds[pick]; b=f"{round(round(chosen/0.05)*0.05,2):.2f}"
            key=f"{(ht['id'] if pick=='1' else at['id'])}@{b}@{pick}"
            stats = mem["price_buckets"].get(key, {"w":0,"l":0})
            hist_n = stats["w"]+stats["l"]
            if hist_n>=3:
                hist_wr = stats["w"]/hist_n
                hist_line = f" | Geçmiş @{b}: {stats['w']}/{hist_n} (%{int(100*hist_wr)})"
        else:
            # prob kovası
            def bucket_prob(p, step=0.05):
                low = math.floor(p/step)*step; high=low+step
                return f"{int(low*100)}–{int(high*100)}%"
            bprob = bucket_prob(p_model)
            key=f"{(ht['id'] if pick=='1' else at['id'])}@{bprob}@{pick}"
            stats = mem["prob_buckets"].get(key, {"w":0,"l":0})
            hist_n = stats["w"]+stats["l"]
            if hist_n>=3:
                hist_wr = stats["w"]/hist_n
                hist_line = f" | Geçmiş (prob {bprob}): {stats['w']}/{hist_n} (%{int(100*hist_wr)})"

        p_final = blend_conf(p_model, p_market, hist_wr, hist_n, st=st)
        tag = " (YÜKSEK GÜVEN)" if p_final>=0.90 else ""
        why = explain_pick(pick, probs, hs, as_, mu_adj)

        lam_cards = est_cards_lambda(hs,as_)
        p_c_o45 = over_prob(lam_cards,4.5); p_c_o35 = over_prob(lam_cards,3.5)
        lam_h, lam_a = probs["lam_home"], probs["lam_away"]
        lam_corners = est_corners_lambda(lam_h,lam_a,hs,as_)
        p_k_o95 = over_prob(lam_corners,9.5); p_k_o85 = over_prob(lam_corners,8.5)
        alts=[]
        if p_c_o45>=0.62: alts.append(f"Kartlar Over 4.5 (%{int(100*p_c_o45)})")
        elif p_c_o35>=0.62: alts.append(f"Kartlar Over 3.5 (%{int(100*p_c_o35)})")
        if p_k_o95>=0.62: alts.append(f"Korner Over 9.5 (%{int(100*p_k_o95)})")
        elif p_k_o85>=0.62: alts.append(f"Korner Over 8.5 (%{int(100*p_k_o85)})")
        alt_line = (" | Alternatif: " + "; ".join(alts)) if alts else ""

        odds_txt = f" | Oran: 1={odds['1']:.2f}/X={odds['X']:.2f}/2={odds['2']:.2f}" if odds else ""
        lines.append(
            f"• {ht['name']} vs {at['name']} → 1X2: **{pick}** ({p_final*100:.1f}%)"
            f"{tag}{odds_txt} — Nedenler: {why}{hist_line}\n"
            f"  Kart λ≈{lam_cards:.1f}, Korner λ≈{lam_corners:.1f}{alt_line}"
        )

        mem["pending"][str(m["id"])] = {"key": key, "side": pick, "has_odds": bool(odds)}
        save_mem(mem)

    lines.append("\nNot: Oran verisi opsiyoneldir; yoksa model olasılık kovası kullanılır.")
    return "\n".join(lines)

def micro_learn_from_match(st, match, pick, won, lam_h, lam_a):
    lk = league_key(match)
    lp = league_params(st, lk)
    lr_mu   = 0.01
    lr_home = 0.02
    lr_tau  = 0.02

    ft = (match.get("score") or {}).get("fullTime",{})
    hg, ag = (ft.get("home",0) or 0), (ft.get("away",0) or 0)
    total  = hg + ag
    exp_tot = lam_h + lam_a

    if not won and (total < exp_tot - 0.8):
        lp["mu_offset"] = max(-0.2, lp["mu_offset"] - lr_mu)
    elif won and (total > exp_tot + 0.8):
        lp["mu_offset"] = min(0.2, lp["mu_offset"] + lr_mu/2)

    if not won and pick == "1" and ag > hg:
        lp["home_adv"] = max(1.00, lp["home_adv"] - lr_home)
    if not won and pick == "2" and hg > ag:
        lp["home_adv"] = min(1.25, lp["home_adv"] + lr_home)

    if not won: 
        lp["tau"] = min(1.40, lp["tau"] + lr_tau)
    else:
        lp["tau"] = max(0.80, lp["tau"] - lr_tau/2)

def build_results_and_learn():
    ms = get_matches(TODAY, TODAY)
    st = load_state()
    mem = load_mem()
    lines=[f"📊 Günün Sonuçları — {TODAY.isoformat()}"]
    correct=wrong=0
    league_mu_base = league_goal_tempo()

    for m in sorted(ms, key=lambda x: x.get("utcDate")):
        if m.get("status")!="FINISHED": continue
        ht,at = m["homeTeam"], m["awayTeam"]
        ft = (m.get("score") or {}).get("fullTime",{})
        hg, ag = (ft.get("home",0) or 0), (ft.get("away",0) or 0)

        mid=str(m["id"])
        if mid not in mem["pending"]: 
            continue
        entry = mem["pending"].pop(mid); save_mem(mem)
        pick = entry["side"]

        hs,as_ = team_form_rating(ht["id"]), team_form_rating(at["id"])
        lp = league_params(st, league_key(m))
        mu_adj = league_mu_base * (1.0 + lp["mu_offset"])
        probs = match_probs_with_params(hs,as_,mu_adj, lp["home_adv"])
        lam_h, lam_a = probs["lam_home"], probs["lam_away"]

        won = (pick=="1" and hg>ag) or (pick=="2" and ag>hg) or (pick=="X" and hg==ag)
        if won: correct += 1
        else:   wrong   += 1

        bucket_map = mem["price_buckets"] if entry.get("has_odds") else mem["prob_buckets"]
        b = bucket_map.get(entry["key"], {"w":0,"l":0})
        if won: b["w"]+=1
        else:   b["l"]+=1
        bucket_map[entry["key"]]=b
        save_mem(mem)

        micro_learn_from_match(st, m, pick, won, lam_h, lam_a)

        status = "✅ tuttu" if won else "❌ tutmadı"
        reason = "" if won else (" — Neden (kısa): "+ why_wrong(pick, lam_h, lam_a, hg, ag))
        lines.append(f"— {ht['name']} {hg}-{ag} {at['name']} → Seçim: {pick} → {status}{reason}")

    save_state(st)
    lines.append(f"\nÖzet: {correct} doğru / {wrong} yanlış")
    return "\n".join(lines)

def run_predict():
    try:
        ms = todays_scheduled()
        report = build_prediction_report(ms)
        send_mail(f"Günün Tahminleri | {TODAY.isoformat()}", report)
    except Exception as e:
        send_mail(f"[HATA] Tahmin Botu | {TODAY.isoformat()}", f"Hata: {e}")

def run_results():
    try:
        report = build_results_and_learn()
        send_mail(f"Günün Sonuçları | {TODAY.isoformat()}", report)
    except Exception as e:
        send_mail(f"[HATA] Sonuç Botu | {TODAY.isoformat()}", f"Hata: {e}")

if __name__ == "__main__":
    mode = None
    if len(sys.argv)>=3 and sys.argv[1]=="--mode":
        mode = sys.argv[2]
    if mode=="results":
        run_results()
    else:
        run_predict()
