import os, smtplib, math, json, time
from datetime import datetime, timedelta, timezone
from email.mime.text import MIMEText

import requests
from dateutil.relativedelta import relativedelta

# ---------------- CFG ----------------
FD_TOKEN = os.environ.get("FOOTBALL_DATA_TOKEN", "").strip()
GMAIL_USER = os.environ.get("GMAIL_USER", "").strip()
GMAIL_PASS = os.environ.get("GMAIL_PASS", "").strip()
GMAIL_TO   = os.environ.get("GMAIL_TO", "").strip()
MODE       = os.environ.get("MODE", "PREDICT").upper()  # PREDICT | RESULTS

API_BASE = "https://api.football-data.org/v4"

TR_TZ = timezone(timedelta(hours=3))

# ---------------- UTIL ----------------
def today_tr():
    # TR tarihini (YYYY-MM-DD) üret
    return datetime.now(TR_TZ).date()

def send_mail(subject: str, body: str):
    msg = MIMEText(body, _charset="utf-8")
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = GMAIL_TO
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(GMAIL_USER, GMAIL_PASS)
        s.sendmail(GMAIL_USER, [GMAIL_TO], msg.as_string())

def http_get(url, params=None):
    headers = {"X-Auth-Token": FD_TOKEN} if FD_TOKEN else {}
    r = requests.get(url, params=params or {}, headers=headers, timeout=30)
    r.raise_for_status()
    return r.json()

def poisson_pmf(lmb, k):
    return math.exp(-lmb) * (lmb**k) / math.factorial(k)

def probs_poisson(lh, la):
    p1 = pX = p2 = 0.0
    for h in range(0, 11):
        ph = poisson_pmf(lh, h)
        for a in range(0, 11):
            pa = poisson_pmf(la, a)
            if h > a: p1 += ph * pa
            elif h == a: pX += ph * pa
            else: p2 += ph * pa
    return {"1": p1, "X": pX, "2": p2}

# ---------------- DATA ----------------
def fetch_matches_for_date(day):
    """Günün tüm maçları."""
    if not FD_TOKEN:
        return []
    d = day.strftime("%Y-%m-%d")
    url = f"{API_BASE}/matches"
    data = http_get(url, {"dateFrom": d, "dateTo": d})
    return data.get("matches", [])

def fetch_team_recent(team_id, days=90):
    """Takımın son 10 bitmiş maçı (GF/GA formu)."""
    if not FD_TOKEN:
        return {"gf": 1.2, "ga": 1.2}  # güvenli varsayılan
    to = today_tr()
    fr = to - relativedelta(days=days)
    url = f"{API_BASE}/teams/{team_id}/matches"
    params = {
        "status": "FINISHED",
        "dateFrom": fr.strftime("%Y-%m-%d"),
        "dateTo": to.strftime("%Y-%m-%d"),
    }
    data = http_get(url, params)
    ms = data.get("matches", [])[:10]  # ilk 10
    gf = ga = n = 0
    for m in ms:
        ht, at = m["homeTeam"]["id"], m["awayTeam"]["id"]
        sc = (m.get("score") or {}).get("fullTime", {})
        hg, ag = (sc.get("home") or 0), (sc.get("away") or 0)
        if team_id == ht:
            gf += hg; ga += ag
        else:
            gf += ag; ga += hg
        n += 1
    if n == 0:  # veri yoksa nötr
        return {"gf": 1.2, "ga": 1.2}
    return {"gf": max(0.2, gf / n), "ga": max(0.2, ga / n)}

def league_mu_proxy():
    """Lig gol temposu için basit proxy (global ortalama)."""
    # Basit güvenli sabit: 2.6 toplam gol ~ Avrupa ortalaması
    return 2.6

# ---------------- MODEL ----------------
def match_prediction(m):
    """Tek maç için 1X2 tahmini + nedenler."""
    ht = m["homeTeam"]; at = m["awayTeam"]
    hid, aid = ht["id"], at["id"]
    hname, aname = ht["name"], at["name"]

    # Form
    hs = fetch_team_recent(hid)
    as_ = fetch_team_recent(aid)

    mu = league_mu_proxy()
    # basit ev-saha etkisi
    home_adv = 1.10

    # lambda'lar: lig temposu × form oranı
    lam_h = max(0.2, mu * (hs["gf"] / max(0.5, as_["ga"])) * home_adv)
    lam_a = max(0.2, mu * (as_["gf"] / max(0.5, hs["ga"])))

    pr = probs_poisson(lam_h, lam_a)
    pick = max(pr, key=pr.get)
    conf = round(pr[pick] * 100, 1)

    # 1 satır açıklama
    why = []
    if hs["gf"] > as_["ga"]:  why.append("Ev formu↑")
    if as_["gf"] > hs["ga"]:  why.append("Dep. üretken")
    if lam_h + lam_a >= 2.8:  why.append("Tempo↑")
    if lam_h + lam_a <= 2.2:  why.append("Tempo↓")
    why_s = ", ".join(why) if why else "Temel form/tempo"

    return {
        "id": m["id"],
        "home": hname,
        "away": aname,
        "lam_h": lam_h, "lam_a": lam_a,
        "p": pr, "pick": pick, "conf": conf,
        "why": why_s
    }

def format_pred_line(p):
    side_map = {"1": p["home"], "X": "Beraberlik", "2": p["away"]}
    return f"• {p['home']} – {p['away']} → Seçim: {side_map[p['pick']]} ({p['conf']}%)  —  {p['why']}"

# ---------------- FLOWS ----------------
def run_predict():
    day = today_tr()
    ms = fetch_matches_for_date(day)
    # çok alakasız ligleri temizlemek istersen burada filtreleyebilirsin
    if not ms:
        send_mail(f"Günün Tahminleri | {day}", "Bugün için maç bulunamadı.")
        return

    lines = [f"📅 Günün Tahminleri — {day}", ""]
    preds = []
    for m in ms:
        if m.get("status") not in (None, "SCHEDULED", "TIMED", "POSTPONED"):
            continue
        try:
            p = match_prediction(m)
            preds.append(p)
        except Exception as e:
            lines.append(f"• {m['homeTeam']['name']} – {m['awayTeam']['name']} → hesaplanamadı ({e})")

        # rate limitten kaçınmak için mini bekleme
        time.sleep(0.2)

    if not preds:
        send_mail(f"Günün Tahminleri | {day}", "Bugün için uygun maç bulunamadı.")
        return

    # güvene göre sırala
    preds.sort(key=lambda x: x["conf"], reverse=True)
    for p in preds:
        lines.append(format_pred_line(p))

    # yüksek güven sepeti
    hi = [p for p in preds if p["conf"] >= 90]
    if hi:
        lines += ["", "⚡ Yüksek Güven (≥%90):"]
        for p in hi:
            lines.append(f"  - {p['home']} – {p['away']} ({p['conf']}%)")

    body = "\n".join(lines)
    send_mail(f"Günün Tahminleri | {day}", body)

def run_results():
    day = today_tr()
    ms = fetch_matches_for_date(day)
    if not ms:
        send_mail(f"Günün Sonuçları | {day}", "Bugün için maç bulunamadı.")
        return

    lines = [f"📊 Günün Sonuçları — {day}", ""]
    correct = wrong = 0

    for m in ms:
        if m.get("status") != "FINISHED":
            continue
        sc = (m.get("score") or {}).get("fullTime", {})
        hg, ag = (sc.get("home") or 0), (sc.get("away") or 0)

        # aynı günkü metoda göre pick’i tekrar üret (deterministik)
        try:
            p = match_prediction(m)
        except Exception as e:
            lines.append(f"• {m['homeTeam']['name']} {hg}-{ag} {m['awayTeam']['name']} → analiz edilemedi ({e})")
            continue

        if (p["pick"] == "1" and hg > ag) or (p["pick"] == "2" and ag > hg) or (p["pick"] == "X" and hg == ag):
            correct += 1
            res = "✅ tuttu"
        else:
            wrong += 1
            res = "❌ tutmadı"

        lines.append(f"• {m['homeTeam']['name']} {hg}-{ag} {m['awayTeam']['name']} → Seçim: {p['pick']} ({p['conf']}%) → {res}")

        time.sleep(0.2)

    lines += ["", f"Özet: {correct} doğru / {wrong} yanlış"]
    send_mail(f"Günün Sonuçları | {day}", "\n".join(lines))

# ---------------- MAIN ----------------
if __name__ == "__main__":
    if MODE == "RESULTS":
        run_results()
    else:
        run_predict()
