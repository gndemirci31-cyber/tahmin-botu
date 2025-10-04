# mailer.py  —  Tam Otomatik (AUTO/PREDICT/RESULTS) + TR saati + yarına kaydırma
# Gereken Secrets: GMAIL_USER, GMAIL_PASS, GMAIL_TO, FOOTBALL_DATA_TOKEN (football-data.org)

import os, smtplib, math, time
from email.mime.text import MIMEText
from datetime import datetime, timedelta, timezone
import requests

# ---- Ayarlar ----
TR = timezone(timedelta(hours=3))
FD_API = "https://api.football-data.org/v4"

GMAIL_USER = os.environ.get("GMAIL_USER")
GMAIL_PASS = os.environ.get("GMAIL_PASS")
GMAIL_TO   = os.environ.get("GMAIL_TO", GMAIL_USER)
FD_TOKEN   = os.environ.get("FOOTBALL_DATA_TOKEN", "").strip()
MODE_ENV   = os.environ.get("MODE", "AUTO").upper()  # AUTO | PREDICT | RESULTS

# ---- Email ----
def send_mail(subject: str, body: str):
    if not (GMAIL_USER and GMAIL_PASS and GMAIL_TO):
        print("E-posta bilgileri eksik; mail gönderilemedi.")
        print("Konu:", subject)
        print("Gövde:\n", body)
        return
    msg = MIMEText(body, _charset="utf-8")
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = GMAIL_TO
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(GMAIL_USER, GMAIL_PASS)
        s.sendmail(GMAIL_USER, [GMAIL_TO], msg.as_string())

# ---- Football-Data yardımcıları ----
def _fd_get(path, params=None):
    """football-data çağrısı (basit retry)."""
    if not FD_TOKEN:
        return None
    headers = {"X-Auth-Token": FD_TOKEN}
    url = f"{FD_API}{path}"
    for _ in range(2):
        r = requests.get(url, headers=headers, params=params, timeout=20)
        if r.status_code == 200:
            return r.json()
        # free planda rate limit olabilir
        if r.status_code in (429, 503):
            time.sleep(2)
            continue
        # Diğer hatalar
        print("FD error:", r.status_code, r.text[:200])
        return None
    return None

def get_matches(date_from, date_to, status=None):
    """Belirli gün aralığındaki maçları getirir."""
    if not FD_TOKEN:
        return []
    params = {
        "dateFrom": date_from.isoformat(),
        "dateTo":   date_to.isoformat(),
        # free planda “competitions” filtrelemiyoruz; olanı alıyoruz
    }
    data = _fd_get("/matches", params=params)
    if not data or "matches" not in data:
        return []
    ms = data["matches"]
    if status:
        ms = [m for m in ms if (m.get("status") == status)]
    return ms

# Standings cache: competition_id -> team_id -> position
_STANDINGS = {}
def get_positions_for_comp(competition_id: int):
    if competition_id in _STANDINGS:
        return _STANDINGS[competition_id]
    data = _fd_get(f"/competitions/{competition_id}/standings")
    mapping = {}
    if data and "standings" in data:
        # genelde "TOTAL" tablosu ilk oluyor
        for table in data["standings"]:
            if table.get("type") != "TOTAL": 
                continue
            for row in table.get("table", []):
                team = row.get("team", {})
                tid = team.get("id")
                pos = row.get("position")
                if tid and pos:
                    mapping[tid] = pos
    _STANDINGS[competition_id] = mapping
    return mapping

# ---- Basit tahmin: lig pozisyonu + ev-saha ----
def predict_for_match(m):
    """1/X/2 ve güven yüzdesi döndürür. 
       Basit kural: (daha iyi lig sırası) + ev-saha = 1, yoksa 2; çok yakınsa X."""
    comp = m.get("competition", {})
    comp_id = comp.get("id")
    comp_name = comp.get("name", "")

    ht = m.get("homeTeam", {})
    at = m.get("awayTeam", {})
    hid, aid = ht.get("id"), at.get("id")
    hname, aname = ht.get("name","?"), at.get("name","?")

    # default değerler
    pick = "1"
    conf = 0.58
    reason = "ev-saha"

    # lig pozisyonlarına bak
    pos_map = get_positions_for_comp(comp_id) if comp_id else {}
    ph = pos_map.get(hid)
    pa = pos_map.get(aid)

    if ph and pa:
        # pozisyon düşük olan daha iyi (1 = lider)
        diff = pa - ph  # pozitifse ev daha iyi
        # ev-saha küçük sabit avantaj
        adv = 0.6
        score = diff/10.0 + (adv - 0.5)  # yaklaşık skala

        if score > 0.1:
            pick = "1"
            reason = f"lig sırası (H:{ph}–A:{pa}) + ev-saha"
            conf = 0.58 + min(0.25, abs(diff)/20.0)
        elif score < -0.1:
            pick = "2"
            reason = f"lig sırası (H:{ph}–A:{pa})"
            conf = 0.56 + min(0.25, abs(diff)/20.0)
        else:
            pick = "X"
            reason = f"dengeli (H:{ph}–A:{pa})"
            conf = 0.52

    # yüzdeye çevir, sınırla
    conf = max(0.50, min(0.85, conf))
    return pick, int(round(conf*100)), reason, comp_name, hname, aname

# ---- Raporlar ----
def build_predict_report(target_date):
    ms = get_matches(target_date, target_date)
    # başlamamış olanları al
    ms = [m for m in ms if m.get("status") in ("TIMED","SCHEDULED")]
    if not ms:
        return f"Bugün için tahmin çıkarılacak maç bulunamadı.", False

    lines = [f"📅 Tarih: {target_date.isoformat()} — Başlamamış maçlar"]
    # Zamanı TR'ye çevir
    for m in sorted(ms, key=lambda x: x.get("utcDate","")):
        utc_str = m.get("utcDate")
        try:
            dt_utc = datetime.fromisoformat(utc_str.replace("Z","+00:00"))
            dt_tr  = dt_utc.astimezone(TR)
            saat   = dt_tr.strftime("%H:%M")
        except Exception:
            saat = "—"

        pick, conf, reason, comp_name, hname, aname = predict_for_match(m)
        lines.append(f"• {saat} | {hname} vs {aname} — Tahmin: {pick} (güven %{conf}) | {comp_name} | {reason}")
    return "\n".join(lines), True

def build_results_report(target_date):
    ms = get_matches(target_date, target_date, status="FINISHED")
    if not ms:
        return "Bugün için sonuç bulunamadı.", False
    lines = [f"📊 {target_date.isoformat()} — Bitmiş maçlar"]
    for m in sorted(ms, key=lambda x: x.get("utcDate","")):
        ht, at = m.get("homeTeam",{}), m.get("awayTeam",{})
        hname, aname = ht.get("name","?"), at.get("name","?")
        ft = (m.get("score") or {}).get("fullTime", {})
        hg, ag = ft.get("home",0), ft.get("away",0)
        lines.append(f"• {hname} {hg}-{ag} {aname}")
    return "\n".join(lines), True

# ---- Ana akış ----
def main():
    mode = MODE_ENV
    now_tr = datetime.now(TR)

    if mode not in ("AUTO","PREDICT","RESULTS"):
        mode = "AUTO"

    if mode == "AUTO":
        # TR'de 21:00'dan önce PREDICT, sonra RESULTS
        mode = "PREDICT" if now_tr.hour < 21 else "RESULTS"

    # PREDICT'i gece çalıştırırsak yarına kaydır
    target_date = now_tr.date()
    if mode == "PREDICT" and now_tr.hour >= 21:
        target_date = (now_tr + timedelta(days=1)).date()

    if mode == "PREDICT":
        body, ok = build_predict_report(target_date)
        subject = f"Günün Tahminleri | {target_date.isoformat()}"
    else:
        body, ok = build_results_report(now_tr.date())
        subject = f"Günün Sonuçları | {now_tr.date().isoformat()}"

    send_mail(subject, body)
    print(subject)
    print(body)

if __name__ == "__main__":
    main()
