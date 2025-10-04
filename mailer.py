import os, smtplib, datetime, random
from email.mime.text import MIMEText

GMAIL_USER = os.environ["GMAIL_USER"]
GMAIL_PASS = os.environ["GMAIL_PASS"]
GMAIL_TO   = os.environ.get("GMAIL_TO", GMAIL_USER)

HIGH_CONF_THRESHOLD = 0.90  # %90 ve üzeri “önemli” say

def send_mail(subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = GMAIL_TO
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(GMAIL_USER, GMAIL_PASS)
        s.sendmail(GMAIL_USER, [GMAIL_TO], msg.as_string())

# --- ÖRNEK veri (dummy). Sonra gerçek kaynağa bağlayacağız. ---
def get_today_predictions():
    """Her tahmin: {'mac': 'TakımA-TakımB', 'tahmin': '2.5 Üst', 'guven': 0.0-1.0}"""
    today = datetime.datetime.now().strftime('%Y-%m-%d')
    return [
        {"mac": "Maç A", "tahmin": "2.5 Üst",            "guven": 0.92},
        {"mac": "Maç B", "tahmin": "KG Var",             "guven": 0.78},
        {"mac": "Maç C", "tahmin": "Ev sahibi kazanır", "guven": 0.65},
    ], today

def get_finished_results():
    """
    Gün içinde biten maçların sonuçları (dummy).
    Her kayıt: {'mac': 'TakımA-TakımB', 'tahmin': '...', 'sonuc': 'dogru'|'yanlis'}
    """
    rnd = random.random()
    results = [
        {"mac": "Maç A", "tahmin": "2.5 Üst", "sonuc": "dogru" if rnd > 0.4 else "yanlis"},
        {"mac": "Maç B", "tahmin": "KG Var",  "sonuc": "yanlis" if rnd > 0.6 else "dogru"},
    ]
    return results

# --- Raporlar ---
def daily_main_report():
    preds, today = get_today_predictions()
    lines = [f"⚽ Günün Tahminleri — {today}"]
    for p in preds:
        pct = int(round(p["guven"]*100))
        lines.append(f"— {p['mac']} → {p['tahmin']} (güven: %{pct})")
    oneri = [p["mac"] for p in preds if p["guven"] >= 0.8]
    if oneri:
        lines.append("\nBen olsam: " + ", ".join(oneri) + " (güven ≥ %80)")
    send_mail(f"Günün Tahminleri | {today}", "\n".join(lines))

def conditional_alerts():
    """
    Ek mail at:
      1) Çok yüksek güven (≥ %90) tahmin varsa
      2) Sonuçlanan maçlarda 'yanlış' çıkan varsa
    """
    preds, today = get_today_predictions()
    high_conf = [p for p in preds if p["guven"] >= HIGH_CONF_THRESHOLD]
    if high_conf:
        lines = [f"⚡ Önemli Tahmin Uyarısı — {today}"]
        for p in high_conf:
            pct = int(round(p["guven"]*100))
            lines.append(f"— {p['mac']} → {p['tahmin']} (güven: %{pct})")
        lines.append("\nNot: Yüksek güvenli tahminler.")
        send_mail(f"⚡ Önemli Tahmin — {today}", "\n".join(lines))

    results = get_finished_results()
    wrong = [r for r in results if r["sonuc"] == "yanlis"]
    if wrong:
        lines = [f"⚠️ Tahmin-Sonuç Uyarısı — {today}"]
        for r in wrong:
            lines.append(f"— {r['mac']} → Tahmin: {r['tahmin']} → Sonuç: ❌ (yanlış)")
        lines.append("\nNot: Bir sonraki tahminde düzeltme uygulanacak.")
        send_mail(f"⚠️ Tahmin-Sonuç Uyarısı — {today}", "\n".join(lines))

if __name__ == "__main__":
    # 15:00'te günlük ana rapor + koşullu uyarılar
    daily_main_report()
    conditional_alerts()
