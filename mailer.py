import os, smtplib, datetime
from email.mime.text import MIMEText

GMAIL_USER = os.environ["GMAIL_USER"]
GMAIL_PASS = os.environ["GMAIL_PASS"]
GMAIL_TO   = os.environ.get("GMAIL_TO", GMAIL_USER)

def send_mail(subject, body):
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = GMAIL_USER
    msg["To"] = GMAIL_TO
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(GMAIL_USER, GMAIL_PASS)
        s.sendmail(GMAIL_USER, [GMAIL_TO], msg.as_string())

def sabah_tahmin():
    body = (
        "⚽ Günün Tahminleri\n"
        "- Maç A → 2.5 Üst\n"
        "- Maç B → KG Var\n"
        "- Maç C → Ev sahibi kazanır\n"
        f"\nTarih: {datetime.datetime.now().strftime('%Y-%m-%d')}"
    )
    send_mail("Günün Tahminleri", body)

def gece_sonuclari():
    body = (
        "📊 Günün Sonuç Raporu\n"
        "- Maç A → 3-2 ✅\n"
        "- Maç B → 0-0 ❌\n"
        "- Maç C → 1-2 ❌\n"
        f"\nTarih: {datetime.datetime.now().strftime('%Y-%m-%d')}"
    )
    send_mail("Günün Sonuç Raporu", body)

if __name__ == "__main__":
    import sys
    job = sys.argv[1] if len(sys.argv) > 1 else "morning"
    if job == "morning":
        sabah_tahmin()
    else:
        gece_sonuclari()
