# TÜM HEDEF LİGLER İÇİN GERÇEK API VERİLERİYLE ANALİZ TEST KODU - GERÇEKÇİ GÜVEN ALGORİTMASI

import requests
import math
import random
from datetime import datetime, timedelta
from difflib import SequenceMatcher
import smtplib
import os
from email.message import EmailMessage

# API AYARLARI
APIFOOTBALL_KEY = "f6570111b0cdddb86828ef25179e6ce7"
HEADERS = {'x-apisports-key': APIFOOTBALL_KEY}
BASE_URL = "https://v3.football.api-sports.io/"

# EMAIL AYARLARI - GITHUB SECRETS'TAN AL
GMAIL_USER = os.getenv("GMAIL_USER", "gndemirci31@gmail.com")
GMAIL_PASS = os.getenv("GMAIL_PASS", "dejd ctjd lxvq vjle")
GMAIL_TO = os.getenv("GMAIL_TO", "gndemirci31@gmail.com")

# GITHUB ACTIONS KONTROLÜ
GITHUB_ACTIONS = os.getenv("GITHUB_ACTIONS", False)

# OTOMATİK SİSTEM AYARLARI
TAHMIN_SAATI = os.getenv("TAHMIN_SAATI", "false").lower() == "true"  # 10:00 için
SONUC_SAATI = os.getenv("SONUC_SAATI", "false").lower() == "true"    # 04:00 için

# HEDEF LİG/KUPA/TURNUVALAR
HEDEF_LIG_IDS = [
    # ULUSLARARASI TURNUVALAR
    '2',    # UEFA Champions League
    '3',    # UEFA Europa League 
    '848',  # UEFA Europa Conference League
    '667',  # UEFA Nations League
    '1',    # FIFA World Cup
    '4',    # UEFA European Championship
    '5',    # Copa América
    '6',    # AFC Asian Cup
    '7',    # Africa Cup of Nations
    '9',    # CONCACAF Gold Cup
    
    # EKLENEN YENİ TURNUVALAR
    '4',    # UEFA European Championship
    '15',   # FIFA World Cup Qualification
    '16',   # UEFA European Championship Qualification
    '17',   # UEFA Champions League Qualification
    '18',   # UEFA Europa League Qualification
    '19',   # UEFA Europa Conference League Qualification
    '20',   # UEFA Nations League
    
    # EKLENEN MİLLİ TAKIM ELEME LİGLERİ:
    '1146', # European Championship Qualification
    '1147', # Euro Qualification - Group Stage  
    '1148', # World Cup Qualification - UEFA
    '1149', # WC Qualification Europe
    '1150', # World Cup Qualifiers UEFA
    '5',    # UEFA Nations League
    '12',   # Nations League A
    '13',   # Nations League B
    '14',   # Nations League C
    '15',   # Nations League D
    '1151', # World Cup Qualification - CONMEBOL
    '1152', # World Cup Qualification - AFC
    '1153', # World Cup Qualification - CAF
    '1154', # World Cup Qualification - CONCACAF
    '17',   # FIFA Confederations Cup
    
    # İNGİLTERE
    '39',   # Premier League
    '40',   # Championship
    '45',   # FA Cup
    '48',   # EFL Cup
    
    # İSPANYA
    '140',  # La Liga
    '141',  # La Liga 2
    '143',  # Copa del Rey
    
    # İTALYA
    '135',  # Serie A
    '136',  # Serie B
    '137',  # Coppa Italia
    
    # ALMANYA
    '78',   # Bundesliga
    '79',   # 2. Bundesliga
    '81',   # DFB-Pokal
    
    # FRANSA
    '61',   # Ligue 1
    '62',   # Ligue 2
    '66',   # Coupe de France
    
    # TÜRKİYE
    '203',  # Süper Lig
    '204',  # TFF 1. Lig
    '206',  # Türkiye Kupası
    
    # PORTEKİZ
    '94',   # Primeira Liga
    '96',   # Taça de Portugal
]

# KADIN/GENÇ LİG FİLTRE KELİMELERİ
FILTRE_KELIMELER = [
    'WOMEN',    # Kadın ligleri
    'U21',      # 21 yaş altı
    'U20',      # 20 yaş altı  
    'U19',      # 19 yaş altı
    'U23',      # 23 yaş altı
    'U18',      # 18 yaş altı
]

def log(message):
    """Log mesajı - GitHub Actions için"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")

def send_mail(subject, body):
    """E-posta gönderme fonksiyonu - GitHub Actions destekli"""
    try:
        # GitHub Actions'da email bilgileri kontrolü
        if not all([GMAIL_USER, GMAIL_PASS, GMAIL_TO]):
            log("❌ Email bilgileri eksik - GitHub Secrets kontrol edin")
            return False
            
        stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        subject = f"{subject} · {stamp}"
        
        body = (body or "").strip()
        if not body:
            body = "(Bu e-postada içerik üretilemedi / maç bulunamadı.)"
            
        msg = EmailMessage()
        msg["From"] = GMAIL_USER
        msg["To"] = GMAIL_TO
        msg["Subject"] = subject
        msg.set_content(body)
        
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(GMAIL_USER, GMAIL_PASS)
            s.send_message(msg)
            
        log(f"✅ Mail gönderildi: {subject}")
        return True
        
    except Exception as e:
        log(f"❌ Mail gönderme hatası: {e}")
        return False

def is_kadin_veya_genc_lig(lig_adi):
    """Lig adında kadın veya genç lig kelimeleri var mı kontrol et"""
    if not lig_adi:
        return False
        
    lig_adi_upper = lig_adi.upper()
    for kelime in FILTRE_KELIMELER:
        if kelime in lig_adi_upper:
            return True
    return False

def format_tahmin_email(tahmin_sonuclari, date_str):
    """Tahmin sonuçlarını e-posta formatında hazırla"""
    if not tahmin_sonuclari:
        return "Bugün için analiz edilecek maç bulunamadı."
    
    lines = []
    lines.append(f"🎯 HEDEF LİGLER TAHMİN RAPORU — {date_str}")
    lines.append("=" * 70)
    lines.append(f"⏰ Üretilme Zamanı: {datetime.now().strftime('%H:%M')}")
    lines.append("")
    
    # İstatistikler
    ortalama_guven = sum(s['confidence'] for s in tahmin_sonuclari) / len(tahmin_sonuclari)
    yuksek_guven = sum(1 for s in tahmin_sonuclari if s['confidence'] >= 70)
    
    tahmin_dagilimi = {
        '1': sum(1 for s in tahmin_sonuclari if s['pick'] == '1'),
        'X': sum(1 for s in tahmin_sonuclari if s['pick'] == 'X'),
        '2': sum(1 for s in tahmin_sonuclari if s['pick'] == '2')
    }
    
    lines.append(f"📊 GENEL İSTATİSTİKLER:")
    lines.append(f"   • Toplam Maç: {len(tahmin_sonuclari)}")
    lines.append(f"   • Ortalama Güven: %{ortalama_guven:.1f}")
    lines.append(f"   • Yüksek Güven (≥%70): {yuksek_guven} maç")
    lines.append(f"   • Tahmin Dağılımı: 1({tahmin_dagilimi['1']}) | X({tahmin_dagilimi['X']}) | 2({tahmin_dagilimi['2']})")
    lines.append("")
    
    # EN YÜKSEK GÜVENLİ TAHMİNLER
    if len(tahmin_sonuclari) >= 3:
        lines.append("🏆 EN YÜKSEK GÜVENLİ 3 TAHMİN:")
        siralı_tahminler = sorted(tahmin_sonuclari, key=lambda x: x['confidence'], reverse=True)[:3]
        for i, tahmin in enumerate(siralı_tahminler, 1):
            lines.append(f"   {i}. {tahmin['match']}")
            lines.append(f"      🎯 Tahmin: {tahmin['pick']} (%{tahmin['confidence']})")
            lines.append(f"      🤖 API-Football AI: {tahmin['api_prediction']} (%{tahmin['api_confidence']})")
            lines.append(f"      ⚽ Skor: {tahmin['skor_tahmini']}")
            lines.append("")
    
    # TÜM TAHMİNLER
    lines.append("📋 TÜM TAHMİNLER:")
    lines.append("=" * 50)
    
    for i, tahmin in enumerate(tahmin_sonuclari, 1):
        confidence_emoji = "🔥" if tahmin['confidence'] >= 70 else "✅"
        lines.append(f"{confidence_emoji} #{i} - %{tahmin['confidence']} GÜVEN")
        lines.append(f"   ⚽ {tahmin['match']}")
        lines.append(f"   🎯 Tahmin: {tahmin['pick']}")
        lines.append(f"   🤖 API-Football AI: {tahmin['api_prediction']} (%{tahmin['api_confidence']})")
        if tahmin['api_advice'] != "N/A":
            lines.append(f"   💡 AI Tavsiye: {tahmin['api_advice']}")
        lines.append(f"   📊 Form: {tahmin['home_form']} vs {tahmin['away_form']}")
        lines.append(f"   ⚽ Gol: 1.5:{tahmin['gol_15']}(%{tahmin['gol_15_prob']}) | 2.5:{tahmin['gol_25']}(%{tahmin['gol_25_prob']})")
        lines.append(f"   🔥 BTTS: {tahmin['btts']} (%{tahmin['btts_prob']})")
        lines.append(f"   📋 Skor: {tahmin['skor_tahmini']}")
        lines.append(f"   ⚠️ Kart 3.5: {tahmin['kart_35']} (%{tahmin['kart_prob']})")
        lines.append(f"   🎯 Korner 7.5: {tahmin['korner_75']} (%{tahmin['korner_prob']})")
        lines.append("")
    
    return "\n".join(lines)

def format_sonuc_email(sonuclar, date_str):
    """Sonuçları e-posta formatında hazırla"""
    if not sonuclar:
        return f"{date_str} tarihi için sonuç bulunamadı."
    
    lines = []
    lines.append(f"📊 HEDEF LİGLER - DÜNÜN MAÇ SONUÇLARI — {date_str}")
    lines.append("=" * 60)
    lines.append(f"⏰ Üretilme Zamanı: {datetime.now().strftime('%H:%M')}")
    lines.append("")
    
    for sonuc in sonuclar:
        lines.append(f"🏆 {sonuc.get('league', 'Bilinmeyen Lig')}")
        lines.append(f"⚽ {sonuc.get('home_team', 'Bilinmiyor')} {sonuc.get('home_score', '?')}-{sonuc.get('away_score', '?')} {sonuc.get('away_team', 'Bilinmiyor')}")
        lines.append("")
    
    lines.append(f"📈 TOPLAM: {len(sonuclar)} maç sonucu")
    
    return "\n".join(lines)

def _apifoot_get(endpoint, params):
    """API-FOOTBALL BAĞLANTI SİSTEMİ"""
    try:
        url = f"{BASE_URL}{endpoint.lstrip('/')}"
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        if response.status_code == 200:
            return response.json().get('response', None)
        else:
            log(f"❌ API Error: Status {response.status_code}")
    except Exception as e:
        log(f"❌ API Error: {e}")
    return None

def get_gercek_form_ultra(team_id, team_name):
    """GERÇEK FORM VERİSİ - API-FOOTBALL"""
    if not team_id:
        return None
        
    try:
        params = {'team': team_id, 'last': 8}
        response = _apifoot_get("fixtures", params)
        
        if not response:
            return None
            
        wins, draws, losses = 0, 0, 0
        goals_for, goals_against = 0, 0
        over_15, over_25, btts = 0, 0, 0
        analyzed_matches = 0
        
        for match in response:
            if match['fixture']['status']['short'] != 'FT':
                continue
                
            home_team = match['teams']['home']['id'] == team_id
            home_goals = match['goals']['home'] or 0
            away_goals = match['goals']['away'] or 0
            
            if home_team:
                goals_for += home_goals
                goals_against += away_goals
                if match['teams']['home']['winner']:
                    wins += 1
                elif match['teams']['away']['winner']:
                    losses += 1
                else:
                    draws += 1
            else:
                goals_for += away_goals
                goals_against += home_goals  
                if match['teams']['away']['winner']:
                    wins += 1
                elif match['teams']['home']['winner']:
                    losses += 1
                else:
                    draws += 1
            
            total_goals = home_goals + away_goals
            if total_goals > 1.5: over_15 += 1
            if total_goals > 2.5: over_25 += 1
            if home_goals > 0 and away_goals > 0: btts += 1
            
            analyzed_matches += 1
        
        if analyzed_matches < 3:
            return None
            
        points = (wins * 3) + draws
        max_points = analyzed_matches * 3
        form_percentage = (points / max_points) * 100 if max_points > 0 else 0
        
        return {
            'form': round(form_percentage, 1),
            'avg_goals_for': round(goals_for / analyzed_matches, 1),
            'avg_goals_against': round(goals_against / analyzed_matches, 1),
            'matches_analyzed': analyzed_matches,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'over_15_percent': round((over_15 / analyzed_matches) * 100, 1),
            'over_25_percent': round((over_25 / analyzed_matches) * 100, 1),
            'btts_percent': round((btts / analyzed_matches) * 100, 1)
        }
        
    except Exception as e:
        log(f"Form verisi hatası ({team_name}): {e}")
        return None

def get_gercek_kart_korner_analiz(home_id, away_id, home_team, away_team, lig_adi):
    """GERÇEK KART/KORNER ANALİZİ"""
    try:
        home_stats = get_takim_kart_korner_istatistikleri(home_id, home_team)
        away_stats = get_takim_kart_korner_istatistikleri(away_id, away_team)
        
        if home_stats and away_stats:
            avg_cards = (home_stats['avg_cards'] + away_stats['avg_cards']) / 2
            avg_corners = (home_stats['avg_corners'] + away_stats['avg_corners']) / 2
            
            kart_prob = calculate_kart_olasilik(avg_cards)
            korner_prob = calculate_korner_olasilik(avg_corners)
            
            return kart_prob, korner_prob
        
        return get_lig_bazli_kart_korner(lig_adi)
        
    except Exception as e:
        return 55, 65

def get_takim_kart_korner_istatistikleri(team_id, team_name):
    """TAKIMIN GERÇEK KART/KORNER İSTATİSTİKLERİ"""
    try:
        params = {'team': team_id, 'last': 6, 'status': 'FT'}
        fixtures = _apifoot_get("fixtures", params)
        
        if not fixtures:
            return None
            
        total_cards, total_corners, match_count = 0, 0, 0
        
        for match in fixtures:
            if match['fixture']['status']['short'] != 'FT':
                continue
                
            fixture_id = match['fixture']['id']
            stats_data = _apifoot_get("fixtures/statistics", {'fixture': fixture_id})
            
            if stats_data:
                for team_stats in stats_data:
                    if team_stats['team']['id'] == team_id:
                        cards = 0
                        corners = 0
                        
                        for stat in team_stats.get('statistics', []):
                            value = stat.get('value', 0)
                            if stat['type'] in ['Yellow Cards', 'Red Cards']:
                                cards += int(value) if value else 0
                            elif stat['type'] == 'Corner Kicks':
                                corners += int(value) if value else 0
                        
                        total_cards += cards
                        total_corners += corners
                        match_count += 1
                        break
        
        if match_count >= 3:
            return {
                'avg_cards': total_cards / match_count,
                'avg_corners': total_corners / match_count
            }
            
    except Exception as e:
        log(f"İstatistik hatası ({team_name}): {e}")
    
    return None

def calculate_kart_olasilik(avg_cards):
    """GERÇEKÇİ KART OLASILIĞI - 3.5 ALT/ÜST"""
    if avg_cards >= 5.0: return min(85, 70 + (avg_cards - 5.0) * 8)
    elif avg_cards >= 4.0: return min(70, 55 + (avg_cards - 4.0) * 15)
    elif avg_cards >= 3.0: return min(55, 40 + (avg_cards - 3.0) * 15)
    else: return max(25, 30 + (avg_cards - 2.0) * 10)

def calculate_korner_olasilik(avg_corners):
    """GERÇEKÇİ KORNER OLASILIĞI - 7.5 ALT/ÜST"""
    if avg_corners >= 10.0: return min(85, 70 + (avg_corners - 10.0) * 5)
    elif avg_corners >= 8.0: return min(70, 55 + (avg_corners - 8.0) * 7.5)
    elif avg_corners >= 6.0: return min(55, 40 + (avg_corners - 6.0) * 7.5)
    else: return max(25, 30 + (avg_corners - 4.0) * 5)

def get_lig_bazli_kart_korner(lig_adi):
    """LİG BAZLI KART/KORNER"""
    lig_ortalamalari = {
        'Premier League': {'kart': (4.0, 4.8), 'korner': (10.0, 11.5)},
        'Bundesliga': {'kart': (3.5, 4.3), 'korner': (9.5, 11.0)},
        'La Liga': {'kart': (4.3, 5.0), 'korner': (8.5, 10.0)},
        'Serie A': {'kart': (4.8, 5.8), 'korner': (8.0, 9.5)},
        'Ligue 1': {'kart': (4.0, 4.8), 'korner': (8.5, 10.0)},
        'Süper Lig': {'kart': (4.8, 5.5), 'korner': (9.0, 10.5)},
        'default': {'kart': (4.3, 5.0), 'korner': (9.0, 10.5)}
    }
    
    for lig, ort in lig_ortalamalari.items():
        if lig.lower() in lig_adi.lower():
            avg_cards = random.uniform(ort['kart'][0], ort['kart'][1])
            avg_corners = random.uniform(ort['korner'][0], ort['korner'][1])
            
            kart_prob = calculate_kart_olasilik(avg_cards)
            korner_prob = calculate_korner_olasilik(avg_corners)
            return kart_prob, korner_prob
    
    return 55, 65

def calculate_realistic_confidence(prediction, advice, pick):
    """GERÇEKÇİ GÜVEN YÜZDESİ HESAPLA"""
    
    base_confidence = 65  # Temel güven
    
    # ADVICE'A GÖRE GÜVEN AYARLA
    advice_lower = advice.lower()
    
    if "winner" in advice_lower and "or" not in advice_lower:
        # Net kazanan tahmini: +15
        base_confidence += 15
    
    elif "double chance" in advice_lower:
        # Çifte şans tahmini: +10
        base_confidence += 10
        
    elif "combo" in advice_lower:
        # Kombine tahmin: +5
        base_confidence += 5
    
    # GOAL TAHMİNLERİNE GÖRE AYAR
    goals = prediction.get('goals', {})
    home_goals = goals.get('home', '')
    away_goals = goals.get('away', '')
    
    if home_goals and away_goals and home_goals != '-3.5' and away_goals != '-3.5':
        # Gerçek gol tahmini varsa: +5
        base_confidence += 5
    
    # UNDER/OVER TAHMİNİ
    under_over = prediction.get('under_over', '')
    if under_over and under_over != 'None':
        # Alt/Üst tahmini: +3
        base_confidence += 3
    
    # WIN OR DRAW DURUMU
    win_or_draw = prediction.get('win_or_draw', False)
    if win_or_draw:
        # Kazan veya beraberlik: +2
        base_confidence += 2
    
    # TAHMİN TÜRÜNE GÖRE SON AYAR
    if pick == "X":
        # Beraberlik tahminleri genelde daha düşük güvenli
        base_confidence = max(55, base_confidence - 5)
    
    # GÜVEN SINIRLARI
    confidence = max(50, min(85, base_confidence))
    
    return confidence

def get_api_football_prediction(fixture_id, home_team, away_team):
    """API-FOOTBALL AI TAHMİNİ - GERÇEKÇİ GÜVEN ALGORİTMASI"""
    try:
        if not fixture_id:
            return {"pick": "N/A", "confidence": 65, "advice": "N/A"}
            
        prediction_data = _apifoot_get("predictions", {"fixture": fixture_id})
        
        if prediction_data and len(prediction_data) > 0:
            prediction = prediction_data[0].get('predictions', {})
            
            if prediction:
                winner = prediction.get('winner', {})
                advice = prediction.get('advice', 'N/A')
                
                log(f"🔍 API GERÇEK VERİ: {home_team} vs {away_team}")
                log(f"   💡 Advice: {advice}")
                
                # TAHMİN SONUCU
                winner_id = winner.get('id')
                home_team_id = prediction_data[0].get('teams', {}).get('home', {}).get('id')
                away_team_id = prediction_data[0].get('teams', {}).get('away', {}).get('id')
                
                if winner_id == home_team_id:
                    pick = "1"
                elif winner_id == away_team_id:
                    pick = "2" 
                else:
                    pick = "X"
                
                # GERÇEKÇİ GÜVEN ALGORİTMASI
                confidence = calculate_realistic_confidence(prediction, advice, pick)
                
                log(f"   🎯 SONUÇ: {pick} (%{confidence})")
                
                return {
                    "pick": pick,
                    "confidence": confidence,
                    "advice": advice
                }
                
    except Exception as e:
        log(f"AI tahmin hatası: {e}")
    
    return {"pick": "N/A", "confidence": 65, "advice": "N/A"}

def find_fixture_id(home_team, away_team):
    """FIXTURE ID BUL"""
    try:
        today = datetime.now().strftime("%Y-%m-%d")
        params = {"date": today}
        fixtures = _apifoot_get("fixtures", params)
        
        if fixtures:
            for fixture in fixtures:
                fixture_home = fixture.get('teams', {}).get('home', {}).get('name', '')
                fixture_away = fixture.get('teams', {}).get('away', {}).get('name', '')
                
                if (home_team.lower() in fixture_home.lower() and 
                    away_team.lower() in fixture_away.lower()):
                    return fixture.get('fixture', {}).get('id')
    except Exception as e:
        log(f"Fixture ID bulma hatası: {e}")
    
    return None

def tahmin_hesapla_gercek_verilerle(home_form, away_form, home_team, away_team, lig_adi, home_id, away_id):
    """GERÇEK VERİLERLE TAHMİN HESAPLA"""
    
    # GÜÇ HESABI
    home_power = home_form['form'] * 1.15
    away_power = away_form['form']
    power_diff = home_power - away_power
    
    # TAHMİN MANTIĞI
    if power_diff > 20:
        pick = "1"
        confidence = min(85, 65 + (power_diff - 20))
    elif power_diff < -20:
        pick = "2" 
        confidence = min(85, 65 + (abs(power_diff) - 20))
    else:
        pick = "X"
        confidence = min(75, 55 + (20 - abs(power_diff)))
    
    # API AI TAHMİNİ - GERÇEKÇİ GÜVEN ALGORİTMASI
    fixture_id = find_fixture_id(home_team, away_team)
    api_prediction_data = get_api_football_prediction(fixture_id, home_team, away_team)
    
    api_pick = api_prediction_data["pick"]
    api_confidence = api_prediction_data["confidence"]
    api_advice = api_prediction_data["advice"]
    
    # GOL OLASILIKLARI
    over_15_prob = (home_form['over_15_percent'] + away_form['over_15_percent']) / 2
    over_25_prob = (home_form['over_25_percent'] + away_form['over_25_percent']) / 2
    btts_prob = (home_form['btts_percent'] + away_form['btts_percent']) / 2
    
    # SKOR TAHMİNİ
    home_goals = max(0.5, min(3.5, home_form['avg_goals_for']))
    away_goals = max(0.5, min(3.0, away_form['avg_goals_for']))
    
    if pick == "1":
        home_goals = min(3.5, home_goals + 0.5)
        away_goals = max(0, away_goals - 0.3)
        skor_tahmini = f"{round(home_goals)}-{round(away_goals)}"
    elif pick == "2":
        away_goals = min(3.0, away_goals + 0.5)
        home_goals = max(0, home_goals - 0.3)
        skor_tahmini = f"{round(home_goals)}-{round(away_goals)}"
    else:
        skor_tahmini = f"{round(home_goals)}-{round(away_goals)}"
    
    # KART/KORNER ANALİZİ
    kart_prob, korner_prob = get_gercek_kart_korner_analiz(home_id, away_id, home_team, away_team, lig_adi)
    
    # VERİ KALİTESİ
    data_quality = min(100, (
        home_form['matches_analyzed'] * 2 + 
        away_form['matches_analyzed'] * 2 +
        (70 if api_pick != "N/A" else 0)
    ))
    
    return {
        'match': f"{home_team} vs {away_team}",
        'pick': pick,
        'confidence': int(confidence),
        'api_prediction': api_pick,
        'api_confidence': api_confidence,
        'api_advice': api_advice,
        'home_form': f"%{home_form['form']}",
        'away_form': f"%{away_form['form']}",
        'gol_15': "ÜST" if over_15_prob > 50 else "ALT",
        'gol_15_prob': int(over_15_prob),
        'gol_25': "ÜST" if over_25_prob > 50 else "ALT",
        'gol_25_prob': int(over_25_prob),
        'btts': "EVET" if btts_prob > 50 else "HAYIR",
        'btts_prob': int(btts_prob),
        'skor_tahmini': skor_tahmini,
        'kart_35': "ÜST" if kart_prob > 50 else "ALT",
        'kart_prob': int(kart_prob),
        'korner_75': "ÜST" if korner_prob > 50 else "ALT",
        'korner_prob': int(korner_prob),
        'data_quality': data_quality
    }

def run_tahmin_analizi():
    """SAAT 10:00 - BUGÜNÜN TAHMİNLERİ"""
    log("🔄 10:00 TAHMİN ANALİZİ BAŞLATILIYOR...")
    
    today = datetime.now().strftime("%Y-%m-%d")
    log(f"📅 Tahmin Tarihi: {today}")
    
    # BUGÜNÜN TÜM MAÇLARINI ÇEK
    params = {"date": today}
    tum_maclar = _apifoot_get("fixtures", params)
    
    if not tum_maclar:
        log("❌ Bugün hiç maç bulunamadı!")
        return "Bugün için analiz edilecek maç bulunamadı."
    
    log(f"📊 API'den gelen toplam maç: {len(tum_maclar)}")
    
    # HEDEF LİGLERİ FİLTRELE
    hedef_maclar = []
    for mac in tum_maclar:
        lig_id = str(mac.get('league', {}).get('id', ''))
        if lig_id in HEDEF_LIG_IDS:
            # KADIN/GENÇ LİG FİLTRELEME
            lig_adi = mac.get('league', {}).get('name', '')
            if is_kadin_veya_genc_lig(lig_adi):
                log(f"⏭️ Atlanan lig (kadın/genç): {lig_adi}")
                continue
            hedef_maclar.append(mac)
    
    log(f"🎯 Hedef lig/kupa/turnuvalardaki maç: {len(hedef_maclar)}")
    
    if not hedef_maclar:
        log("❌ Hedef lig/kupa/turnuvalarda bugün maç bulunamadı!")
        return "Hedef liglerde bugün maç bulunamadı."
    
    # TÜM MAÇLARI ANALİZ ET
    log(f"🧪 Analiz edilecek maç: {len(hedef_maclar)}")
    
    tahmin_sonuclari = []
    
    for i, mac in enumerate(hedef_maclar, 1):
        fixture = mac.get('fixture', {})
        teams = mac.get('teams', {})
        league = mac.get('league', {})
        
        home_team = teams.get('home', {}).get('name', 'Bilinmiyor')
        away_team = teams.get('away', {}).get('name', 'Bilinmiyor')
        lig_adi = league.get('name', 'Bilinmeyen Lig')
        home_id = teams.get('home', {}).get('id')
        away_id = teams.get('away', {}).get('id')
        
        log(f"🔮 #{i} - {home_team} vs {away_team}")
        
        try:
            # GERÇEK FORM VERİLERİ
            home_form = get_gercek_form_ultra(home_id, home_team)
            away_form = get_gercek_form_ultra(away_id, away_team)
            
            if home_form and away_form:
                # TAM TAHMİN
                tahmin = tahmin_hesapla_gercek_verilerle(
                    home_form, away_form, home_team, away_team, 
                    lig_adi, home_id, away_id
                )
                
                tahmin_sonuclari.append(tahmin)
                log(f"   ✅ Tahmin: {tahmin['pick']} (%{tahmin['confidence']})")
                
            else:
                log("   ❌ Form verisi alınamadı")
                
        except Exception as e:
            log(f"   ❌ Hata: {e}")
    
    if tahmin_sonuclari:
        email_icerik = format_tahmin_email(tahmin_sonuclari, today)
        return email_icerik
    else:
        return "Bugün için analiz edilecek maç bulunamadı."

def run_sonuc_analizi():
    """SAAT 04:00 - DÜNKÜ SONUÇLAR"""
    log("🔄 04:00 SONUÇ ANALİZİ BAŞLATILIYOR...")
    
    yesterday = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    log(f"📅 Sonuç Tarihi: {yesterday}")
    
    params = {"date": yesterday}
    tum_maclar = _apifoot_get("fixtures", params)
    
    if not tum_maclar:
        log("❌ Dünkü maç bulunamadı!")
        return "Dünkü maç bulunamadı."
    
    # HEDEF LİGLERİ FİLTRELE
    hedef_sonuclar = []
    for mac in tum_maclar:
        lig_id = str(mac.get('league', {}).get('id', ''))
        if lig_id in HEDEF_LIG_IDS:
            # KADIN/GENÇ LİG FİLTRELEME
            lig_adi = mac.get('league', {}).get('name', '')
            if is_kadin_veya_genc_lig(lig_adi):
                continue
                
            fixture = mac.get('fixture', {})
            status = fixture.get('status', {}).get('short', '')
            
            # Sadece bitmiş maçları al
            if status in ['FT', 'AET', 'PEN']:
                teams = mac.get('teams', {})
                goals = mac.get('goals', {})
                league = mac.get('league', {})
                
                sonuc = {
                    'home_team': teams.get('home', {}).get('name', 'Bilinmiyor'),
                    'away_team': teams.get('away', {}).get('name', 'Bilinmiyor'),
                    'home_score': goals.get('home', '?'),
                    'away_score': goals.get('away', '?'),
                    'league': league.get('name', 'Bilinmeyen Lig'),
                    'status': status
                }
                hedef_sonuclar.append(sonuc)
    
    log(f"✅ Hedef liglerde {len(hedef_sonuclar)} maç sonucu bulundu")
    
    if hedef_sonuclar:
        email_icerik = format_sonuc_email(hedef_sonuclar, yesterday)
        return email_icerik
    else:
        return "Dün hedef liglerde bitmiş maç bulunamadı."

# ANA PROGRAM - OTOMATİK SİSTEM
if __name__ == "__main__":
    log("⚽ OTOMATİK TAHMİN SİSTEMİ BAŞLATILDI")
    log(f"🏠 Ortam: {'GitHub Actions' if TAHMIN_SAATI or SONUC_SAATI else 'Local'}")
    log("=" * 80)
    
    if TAHMIN_SAATI:
        # 10:00 - TAHMİN MAİLİ
        log("🎯 TAHMİN ZAMANI (10:00)")
        email_icerik = run_tahmin_analizi()
        send_mail("🎯 GÜNLÜK TAHMİN RAPORU", email_icerik)
        
    elif SONUC_SAATI:
        # 04:00 - SONUÇ MAİLİ
        log("📊 SONUÇ ZAMANI (04:00)")
        email_icerik = run_sonuc_analizi()
        send_mail("📊 HEDEF LİGLER - DÜNÜN MAÇ SONUÇLARI", email_icerik)
        
    else:
        # Manuel çalıştırma
        log("🔧 MANUEL ÇALIŞTIRMA")
        email_icerik = run_tahmin_analizi()
        send_mail("🎯 TAHMİN RAPORU", email_icerik)
