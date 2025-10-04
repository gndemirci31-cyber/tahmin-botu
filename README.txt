Hızlı Başlangıç (Tahmin Botu) — Sadece telefondan e-posta takibi

1) GitHub'da boş bir repo aç.
2) Bu arşivdeki dosyaları repo köküne yükle:
   - requirements.txt
   - mailer.py
   - .github/workflows/predict.yml
   - .github/workflows/results.yml
3) Secrets ekle (Settings → Secrets and variables → Actions → New repository secret):
   - GMAIL_USER  → Gmail adresin
   - GMAIL_PASS  → Gmail uygulama şifren (16 karakter)
   - GMAIL_TO    → Aynı Gmail adresi
   - FOOTBALL_DATA_TOKEN → football-data.org FREE token
   - (İsteğe bağlı) ODDS_API_KEY → ücretsiz odds anahtarı (koymazsan da çalışır)
4) Actions sekmesine git → "Run workflow" ile "Günlük Tahmin (10:00 TR)"'yi bir kez elle çalıştır.
   - Telefonda "Günün Tahminleri | YYYY-MM-DD" mailini görmelisin.
5) Artık otomatik:
   - Her gün 10:00 TR → Tahmin maili
   - Her gün 23:59 TR → Sonuç & öğrenme maili
Not: Bilgisayar açık kalmak zorunda değil; her şey bulutta.

İpuçları:
- Gmail'de filtre/etiket oluştur: subject:"Günün Tahminleri |" OR subject:"Günün Sonuçları |"
- ODDS_API_KEY koymazsan oranlar mailde görünmeyebilir; güven yine model+hafıza ile kalibre edilir.
