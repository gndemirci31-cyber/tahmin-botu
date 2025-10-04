import os, smtplib, datetime, io, csv, urllib.request
from email.mime.text import MIMEText

GMAIL_USER = os.environ["GMAIL_USER"]
GMAIL_PASS = os.environ["GMAIL_PASS"]
GMAIL_TO   = os.environ.get("GMAIL_TO", GMAIL_USER)

# Google Sheet CSV linkleri (GitHub Secrets)
SHEET_CSV_URL   = os.environ.get("SHEET_CSV_URL", "")
RESULTS_CSV_URL = os.environ.get("RESULTS_CSV_URL", "")

HIGH_CONF_THRESHOLD = 0.90  # %90 ve üzeri “önemli”

def send_mail(subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = GMAIL_TO
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(GMAIL_USER, GMAIL_PASS)
        s.sendmail(GMAIL_USER, [GMAIL_TO], msg.as_string())

def fetch_csv(url):
    with urllib.request.urlopen(url, timeout=15) as r:
        data = r.read().decode("utf-8", errors="ignore")
    return list(csv.DictReader(io.StringIO(data)))

def get_today_predictions():
    """Sheet varsa oradan oku; yoksa boş döner."""
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    preds = []
    if SHEET_CSV_URL:
        rows = fetch_csv(SHEET_CSV_URL)
        for row in rows:
            try:
                mac = str(row["mac"]).strip()
                tahmin = str(row["tahmin"]).strip()
                guven_raw = str(row.get("guven", "0")).strip().replace(",", ".")
                guven = float(guven_raw)
                if guven > 1.0:  # 92 → 0.92
                    guven /= 100.0
                preds.append({"mac": mac, "tahmin": tahmin, "guven": guven})
            except Exception:
                continue
    return preds, today

def get_finished_results():
    """Sheet yoksa boş döner; varsa gerçek sonuçları okur."""
    results = []
    if RESULTS_CSV_URL:
        rows = fetch_csv(RESULTS_CSV_URL)
        for row in rows:
            try:
                mac = str(row["mac"]).strip()
                tahmin = str(row["tahmin"]).strip()
                sonuc = str(row.get("sonuc", "")).strip().lower()
                if sonuc in ("dogru", "yanlis"):
                    results.append({"mac": mac, "tahmin": tahmin, "sonuc": sonuc})
            except Exception:
                continue
    return results

def daily_main_report():
    preds, today = get_today_predictions()
    lines = [f"⚽ Günün Tahminleri — {today}"]
    if not preds:
        lines.append("Bugün için tahmin bulunamadı.")
    for p in preds:
        pct = int(round(p["guven"] * 100))
        lines.append(f"— {p['mac']} → {p['tahmin']} (güven: %{pct})")
    oneri = [p["mac"] for p in preds if p["guven"] >= 0.8]
    if oneri:
        lines.append("\nBen olsam: " + ", ".join(oneri) + " (güven ≥ %80)")
    send_mail(f"Günün Tahminleri | {today}", "\n".join(lines))

def conditional_alerts():
    """Ek mail: 1) yüksek güven (≥%90), 2) yanlış çıkan sonuç varsa."""
    preds, today = get_today_predictions()
    high_conf = [p for p in preds if p["guven"] >= HIGH_CONF_THRESHOLD]
    if high_conf:
        lines = [f"⚡ Önemli Tahmin Uyarısı — {today}"]
        for p in high_conf:
            pct = int(round(p["guven"] * 100))
            lines.append(f"— {p['mac']} → {p['tahmin']} (güven: %{pct})")
        send_mail(f"⚡ Önemli Tahmin — {today}", "\n".join(lines))

    results = get_finished_results()
    wrong = [r for r in results if r["sonuc"] == "yanlis"]
    if wrong:
        lines = [f"⚠️ Tahmin-Sonuç Uyarısı — {today}"]
        for r in wrong:
            lines.append(f"— {r['mac']} → Tahmin: {r['tahmin']} → Sonuç: ❌ (yanlış)")
        send_mail(f"⚠️ Tahmin-Sonuç Uyarısı — {today}", "\n".join(lines))

if __name__ == "__main__":
    daily_main_report()
    conditional_alerts()
