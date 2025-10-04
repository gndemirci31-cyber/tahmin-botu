import os, sys, json, math, smtplib, requests
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText

# ------------ Ayarlar / Secret'lar ------------
GMAIL_USER = os.environ["GMAIL_USER"]
GMAIL_PASS = os.environ["GMAIL_PASS"]
GMAIL_TO   = os.environ.get("GMAIL_TO", GMAIL_USER)
FD_TOKEN   = os.environ.get("FOOTBALL_DATA_TOKEN", "").strip()
MODE_ENV   = os.environ.get("MODE", "AUTO").upper()

UTC = timezone.utc
TODAY = datetime.now(UTC).date()

# ------------ Mail ------------
def send_mail(subject, body):
    msg = MIMEText(body, "plain", "utf-8")
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = GMAIL_TO
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(GMAIL_USER, GMAIL_PASS)
        s.sendmail(GMAIL_USER, [GMAIL_TO], msg.as_string())

# ------------ football-data.org basit istemci ------------
BASE = "https://api.football-data.org/v4"

def fd_get(path, params=None):
    if not FD_TOKEN:
        raise RuntimeError("FOOTBALL_DATA_TOKEN yok. Settings → Secrets → Actions ekleyin.")
    headers = {"X-Auth-Token": FD_TOKEN}
    r = requests.get(f"{BASE}{path}", params=params or {}, headers=headers, timeout=25)
    if r.status_code == 426:
        # plan kısıtına takıldıysa daha küçük set döner
        pass
    r.raise_for_status()
    return r.json()

def get_matches(date_from, date_to, status=None):
    params = {"dateFrom": str(date_from), "dateTo": str(date_to)}
    js = fd_get("/matches", params)
    ms = js.get("matches", [])
    if status:
        ms = [m for m in ms if m.get("status")==status]
    return ms

def team_recent_form(team_id, n=5):
    try:
        js = fd_get(f"/teams/{team_id}/matches", {"status":"FINISHED","limit":n})
        matches = js.get("matches", [])[:n]
    except Exception:
        matches = []
    gf=ga=0
    for m in matches:
        home_id = m["homeTeam"]["id"]
        ft = (m.get("score") or {}).get("fullTime",{})
        hg, ag = ft.get("home",0) or 0, ft.get("away",0) or 0
        if home_id==team_id:
            gf += hg; ga += ag
        else:
            gf += ag; ga += hg
    if len(matches)==0:
        return {"gfpm":1.2, "gapm":1.2}  # varsayılan
    return {"gfpm": gf/len(matches), "gapm": ga/len(matches)}

# ------------ Basit Poisson tahmini ------------
def poisson_pmf(lmb, k):
    return math.exp(-lmb) * (lmb**k) / math.factorial(k)

def match_probs(h_form, a_form, mu=2.6, home_adv=1.10):
    lam_h = max(0.2, mu * (h_form["gfpm"]/max(0.6, a_form["gapm"])) * home_adv)
    lam_a = max(0.2, mu * (a_form["gfpm"]/max(0.6, h_form["gapm"])))
    p1=pX=p2=0.0
    for h in range(0,8):
        ph=poisson_pmf(lam_h,h)
        for a in range(0,8):
            pa=poisson_pmf(lam_a,a)
            if h>a:   p1+=ph*pa
            elif h==a: pX+=ph*pa
            else:     p2+=ph*pa
    return {"1":p1, "X":pX, "2":p2, "lam_h":lam_h, "lam_a":lam_a}

def pick_from_probs(p):
    side = max(p, key=lambda k: p[k])  # "1","X","2"
    conf = p[side]
    return side, conf

# ------------ Raporlar ------------
def build_predict_report():
    # Bugünün programı (SCHEDULED)
    ms = get_matches(TODAY, TODAY)  # tüm maçlar
    ms = [m for m in ms if m.get("status") in ("TIMED","SCHEDULED")]
    if not ms:
        return "Bugün için tahmin çıkarılacak maç bulunamadı.", False

    lines = [f"📅 Günün Tahminleri — {TODAY.isoformat()}"]
    cnt=0
    for m in sorted(ms, key=lambda x: x.get("utcDate")):
        ht, at = m["homeTeam"]["name"], m["awayTeam"]["name"]
        hid, aid = m["homeTeam"]["id"], m["awayTeam"]["id"]
        try:
            h_form = team_recent_form(hid)
            a_form = team_recent_form(aid)
            probs = match_probs(h_form, a_form)
            side, conf = pick_from_probs(probs)
            conf_pc = int(round(conf*100))
            note = f"(λH={probs['lam_h']:.2f}, λA={probs['lam_a']:.2f} | form H:{h_form['gfpm']:.2f}/{h_form['gapm']:.2f} A:{a_form['gfpm']:.2f}/{a_form['gapm']:.2f})"
            lines.append(f"— {ht} vs {at} → Tahmin: {side} | Güven %{conf_pc} {note}")
            cnt+=1
        except Exception as e:
            lines.append(f"— {ht} vs {at} → (tahmin üretilemedi: {e})")
    if cnt==0:
        lines.append("\nNot: Ücretsiz plandaki lig kapsamı sınırlı olabilir.")
    return "\n".join(lines), True

def build_results_report():
    ms = get_matches(TODAY, TODAY, status="FINISHED")
    if not ms:
        return "Bugün için tamamlanmış maç bulunamadı.", False
    lines = [f"📊 Günün Sonuçları — {TODAY.isoformat()}"]
    for m in sorted(ms, key=lambda x: x.get("utcDate")):
        ht, at = m["homeTeam"]["name"], m["awayTeam"]["name"]
        ft = (m.get("score") or {}).get("fullTime",{})
        hg, ag = ft.get("home",0) or 0, ft.get("away",0) or 0
        lines.append(f"— {ht} {hg}-{ag} {at}")
    return "\n".join(lines), True

# ------------ Koş ve maille ------------
def main():
    # AUTO modu: UTC 07'de PREDICT, UTC 20:59 sonrası RESULTS
    if MODE_ENV not in ("AUTO","PREDICT","RESULTS"):
        mode = "AUTO"
    else:
        mode = MODE_ENV
    now_h = int(datetime.now(UTC).strftime("%H"))
    if mode=="AUTO":
        mode = "PREDICT" if now_h<21 else "RESULTS"

    if mode=="PREDICT":
        body, ok = build_predict_report()
        subject = f"Günün Tahminleri | {TODAY.isoformat()}"
    else:
        body, ok = build_results_report()
        subject = f"Günün Sonuçları | {TODAY.isoformat()}"

    send_mail(subject, body)

if __name__=="__main__":
    main()
