# Hızlı Başlangıç (Tahmin Botu) — Sadece telefondan e-posta takibi

1) GitHub'da boş bir repo aç.
2) Bu arşivdeki dosyaları repo köküne yükle:
   - `requirements.txt`
   - `mailer.py`
   - `.github/workflows/predict.yml`
   - `.github/workflows/results.yml`
3) Secrets ekle (Settings → Secrets and variables → Actions → New repository secret):
   - `GMAIL_USER`  → Gmail adresin
   - `GMAIL_PASS`  → Gmail uygulama şifren (16 karakter)
   - `GMAIL_TO`    → Alıcı Gmail (genelde aynı adres)
   - `FOOTBALL_DATA_TOKEN` → football-data.org FREE token
   - *(İsteğe bağlı)* `ODDS_API_KEY` → ücretsiz odds anahtarı (koymazsan da çalışır)
4) Actions sekmesine git → **Run workflow** ile **Günlük Tahmin (10:00 TR)**’yi bir kez elle çalıştır.
   - Telefonda **“Günün Tahminleri | YYYY-MM-DD”** mailini görmelisin.
5) Artık otomatik:
   - **Her gün 10:00 TR** → Tahmin maili  
   - **Ertesi gün 04:00 TR** → **Dünün** maçları için Sonuç & öğrenme maili  
   - Bilgisayar açık kalmak zorunda değil; her şey bulutta (GitHub Actions).

## Zamanlama (UTC bilgisi)
GitHub Actions cron saatleri **UTC**’dir (Türkiye = UTC+3):
- 10:00 TR → **07:00 UTC**  → `predict.yml` cron: `0 7 * * *`
- 04:00 TR → **01:00 UTC**  → `results.yml` cron: `0 1 * * *`

> Not: `MODE=RESULTS` her zaman **dünün** tarihini raporlar; gece sarkmaları boş maile yol açmaz.

## Öğrenme ve kalıcılık
- Model sonuçlarla birlikte **Elo**, **goal_scale** ve **w_mkt** değerlerini otomatik günceller.
- Tüm durum verileri yerelde `model_state.json` dosyasında saklanır (buluta ayrıca veri yazılmaz).

## İpuçları
- Gmail’de filtre/etiket oluştur:
