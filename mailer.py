# -*- coding: utf-8 -*-
"""
Tahmin Botu — GÜNCELLENMİŞ (Transfermarkt + Milli Takım Elo + CIES/FootyStats Fallback + API-Football Öncelikli + TotalCorner + FootyStats + Kaynak Etiketleme + EV SAHİBİ DENGESİ + Gelişmiş Eşleştirme + FIFA/UEFA Sıralama + U21 Filtresi)
(GÜNCEL: Elo + OppAdj Form + Hava + Odds + API-Football ipucu + High-Alert ayrı mail + Otomatik Öğrenme (w_mkt & goal_scale) + TableAdj (standings) + Streak (W/L) + LİG/KUPA FİLTRESİ + Akıllı Hava + SERVICE modu + Transfermarkt + Milli Takım Elo + Çoklu Fallback Sistemi + Kaynak Etiketleme + Ev Sahibi Dengeleme + Gelişmiş Takım Eşleştirme + FIFA/UEFA Sıralama + U21 Filtresi)

Ücretsiz kaynaklar:
- football-data.org (Fixtures/sonuçlar/standings) -> X-Auth-Token: FOOTBALL_DATA_TOKEN
- API-Football (opsiyonel, free tier) -> APIFOOTBALL_KEY varsa fikstür + ipucu
- OpenLigaDB (fallback) -> anahtar gerekmez
- Open-Meteo (hava) -> anahtar gerekmez
- The Odds API (opsiyonel oranlar) -> ODDS_API_KEY varsa kullanılır
- Transfermarkt (tmapi) -> Kadro değerleri
- TotalCorner -> Köşe korner verisi
- FootyStats -> Kart/korner lig ortalamaları

Modlar:
- MODE=PREDICT -> Çalıştığı anda "Günün Tahminleri"
- MODE=RESULTS -> Çalıştığı anda **dünün** sonuçları (yeni saat planına uygun)
- MODE=AUTO -> Çalıştığı anda saat UTC 07 ise PREDICT, değilse RESULTS
- MODE=SERVICE -> Sürekli çalışır; TR 10:00'da Tahmin, ertesi gün TR 04:00'da DÜNÜN Sonuçlarını yollar (tekrar etmez)
"""

import os, math, time, json, smtplib, traceback, re, urllib.parse
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
import requests
from difflib import SequenceMatcher

# --- Ortak yardımcılar -------------------------------------------------------
TR_TZ = timezone(timedelta(hours=3))  # Türkiye
HEADERS_JSON = {"Accept": "application/json"}

def log(msg):
    print(f"[mailer] {msg}", flush=True)

def http_get(url, headers=None, params=None, timeout=25):
    try:
        r = requests.get(url, headers=headers or {}, params=params or {}, timeout=timeout)
        if r.status_code == 200:
            try:
                return r.json()
            except Exception:
                return None
        log(f"GET {url} -> {r.status_code}")
    except Exception as e:
        log(f"GET ERROR {url}: {e}")
    return None

def to_dt_utc(s):
    try:
        if not s:
            return None
        return datetime.fromisoformat(str(s).replace("Z", "+00:00")).astimezone(timezone.utc)
    except Exception:
        return None

def safe_float(x, default=None):
    try:
        if x is None or x == "":
            return default
        return float(x)
    except Exception:
        return default

def clamp(x, a, b):
    return max(a, min(b, x))

def norm_team(x: str):
    return (x or "").lower().replace(".", " ").replace("-", " ").replace(" fc","").strip()

def season_for_today():
    now = datetime.now(TR_TZ)
    y = now.year
    return y if now.month >= 7 else (y - 1)

# --- Takım Adı Benzerlik Eşleştirme ------------------------------------------
def normalize_team_name(name):
    """Takım adını karşılaştırma için normalize eder - GELİŞTİRİLMİŞ"""
    if not name:
        return ""
    
    # Küçük harfe çevir ve Türkçe karakter sorunlarını çöz
    name = name.lower().strip()
    
    # Türkçe karakter düzeltmeleri
    turkish_fixes = {
        'ı': 'i', 'ğ': 'g', 'ü': 'u', 'ş': 's', 'ö': 'o', 'ç': 'c',
        'â': 'a', 'î': 'i', 'û': 'u'
    }
    for old, new in turkish_fixes.items():
        name = name.replace(old, new)
    
    # Yaygın takım eklerini kaldır
    suffixes = [
        ' fc', ' cf', ' af', ' sf', ' if', ' ff', 
        ' football club', ' club de foot', ' athletic club',
        ' sports club', ' united', ' city', ' town', ' fc.',
        ' real', ' deportivo', ' athletic', ' atletico', ' atlético',
        ' sporting', ' os ', ' as ', ' us ', ' ac ', ' inter ',
        ' borussia', ' dynamo', ' sparta', ' rapid', ' ajax'
    ]
    
    for suffix in suffixes:
        name = name.replace(suffix, '')
    
    # Özel karakterleri ve fazla boşlukları temizle
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Geliştirilmiş özel takım ismi düzeltmeleri
    special_cases = {
        # Türkçe takım düzeltmeleri
        'besiktas': 'besiktas', 'fenerbahce': 'fenerbahce', 'galatasaray': 'galatasaray',
        'trabzonspor': 'trabzonspor', 'istanbul basaksehir': 'istanbul basaksehir',
        'sivasspor': 'sivasspor', 'kayserispor': 'kayserispor', 'konyaspor': 'konyaspor',
        'antalyaspor': 'antalyaspor', 'giresunspor': 'giresunspor', 'hatayspor': 'hatayspor',
        
        # Milli takım düzeltmeleri - GELİŞTİRİLMİŞ
        'equatorial guinea': 'equatorial guinea', 'estuarial guinea': 'equatorial guinea',
        'guinea-bissau': 'guinea bissau', 'guinea bissau': 'guinea bissau',
        'cape verde islands': 'cape verde', 'cape verde': 'cape verde',
        'eswatini': 'eswatini', 'swaziland': 'eswatini',
        'congo dr': 'congo', 'drc': 'congo',
        'central african republic': 'central african republic',
        
        # Diğer düzeltmeler
        'psg': 'paris saint germain', 'paris sg': 'paris saint germain',
        'man united': 'manchester united', 'man utd': 'manchester united',
        'man city': 'manchester city', 'spurs': 'tottenham hotspur',
    }
    
    # Özel durum kontrolü
    for wrong, correct in special_cases.items():
        if wrong in name:
            return correct
    
    return name

def team_similarity(a, b):
    """İki takım adı arasındaki benzerlik skorunu hesaplar (0-1 arası)"""
    if not a or not b:
        return 0.0
    
    a_norm = normalize_team_name(a)
    b_norm = normalize_team_name(b)
    
    # Tam eşleşme
    if a_norm == b_norm:
        return 1.0
    
    # Kelime bazlı benzerlik
    a_words = set(a_norm.split())
    b_words = set(b_norm.split())
    
    if a_words and b_words:
        # Ortak kelime oranı
        common_words = a_words.intersection(b_words)
        word_similarity = len(common_words) / max(len(a_words), len(b_words))
        
        # String benzerlik
        string_similarity = SequenceMatcher(None, a_norm, b_norm).ratio()
        
        # Kombine skor (kelime benzerliği daha ağırlıklı)
        return 0.7 * word_similarity + 0.3 * string_similarity
    
    return SequenceMatcher(None, a_norm, b_norm).ratio()

def find_closest_team(target_team, team_list, threshold=0.65):  # %75 → %65
    """
    Takım listesinde en benzer takımı bulur - EŞİK DÜŞÜRÜLDÜ
    """
    if not target_team or not team_list:
        return None, 0.0
    
    best_match = None
    best_score = 0.0
    
    for team in team_list:
        if not team:
            continue
            
        score = team_similarity(target_team, team)
        if score > best_score and score >= threshold:  # %65 eşik
            best_score = score
            best_match = team
    
    return best_match, best_score

# --- GELİŞMİŞ KADRO DEĞERİ SİSTEMİ (Transfermarkt + Fallback'ler) ------------
TEAM_VALUES_PATH = "team_values.json"

def load_team_values():
    """Takım değerlerini yükler"""
    try:
        if os.path.exists(TEAM_VALUES_PATH):
            with open(TEAM_VALUES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log(f"Takım değerleri yükleme hatası: {e}")
    return {}

def save_team_values(values):
    """Takım değerlerini kaydeder"""
    try:
        with open(TEAM_VALUES_PATH, "w", encoding="utf-8") as f:
            json.dump(values, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log(f"Takım değerleri kaydetme hatası: {e}")

def get_team_value_tmapi(team_name, area="Europe"):
    """Transfermarkt API'sinden kadro değerini getirir - GELİŞTİRİLMİŞ"""
    if not team_name:
        return None, None
    
    try:
        # Önce takım ismini normalize et
        normalized_name = normalize_team_name(team_name)
        
        # tmapi.vercel.app API'si - URL encode eklendi
        encoded_name = urllib.parse.quote(normalized_name)
        url = f"https://tmapi.vercel.app/api/team/{encoded_name}"
        
        log(f"🔍 Transfermarkt API deneniyor: {team_name} -> {url}")
        
        response = http_get(url, timeout=15)
        
        if response and response.get("success"):
            squad_value = response.get("data", {}).get("squad_value", None)
            if squad_value and squad_value > 0:
                log(f"✅ TMAPI başarılı: {team_name} -> {squad_value}M €")
                return squad_value, "TMAPI"
            else:
                log(f"⚠️ TMAPI değer bulunamadı: {team_name} - Yanıt: {response}")
        else:
            log(f"❌ TMAPI hata: {team_name} - Yanıt: {response}")
            
    except Exception as e:
        log(f"❌ Transfermarkt API hatası {team_name}: {str(e)}")
        log(f"🔍 Hata detayı: {traceback.format_exc()}")
    
    return None, None

def get_team_value_cies_fallback(team_name, area="Europe"):
    """CIES/FootyStats fallback - GELİŞTİRİLMİŞ"""
    # Milli takımlar için özel değerler
    national_teams = {
        # Afrika
        "mozambique": 15.0, "guinea": 25.0, "botswana": 12.0, "uganda": 18.0,
        "malawi": 10.0, "equatorial guinea": 20.0, "liberia": 14.0, "namibia": 16.0,
        # Avrupa
        "finland": 45.0, "lithuania": 20.0, "scotland": 80.0, "greece": 65.0,
        "austria": 70.0, "san marino": 5.0, "cyprus": 25.0, "bosnia": 60.0,
        "faroe islands": 8.0, "montenegro": 35.0, "belarus": 30.0, "denmark": 120.0,
        "russia": 90.0, "netherlands": 150.0, "czech republic": 85.0, "iceland": 40.0
    }
    
    team_lower = normalize_team_name(team_name)
    
    # Önce milli takım kontrolü
    for nat_team, value in national_teams.items():
        if nat_team in team_lower:
            return value, "CIES_NATIONAL"
    
    # Lig bazlı ortalama değerler
    league_defaults = {
        "premier league": 180.0, "la liga": 120.0, "serie a": 110.0, 
        "bundesliga": 100.0, "ligue 1": 90.0, "super lig": 40.0,
        "eredivisie": 50.0, "primeira liga": 45.0, "pro league": 35.0,
        "championship": 25.0, "serie b": 15.0, "ligue 2": 12.0,
        "2. bundesliga": 18.0, "la liga 2": 20.0
    }
    
    # Area'dan lig tahmini
    area_lower = area.lower()
    for league, value in league_defaults.items():
        if league in area_lower:
            return value, "CIES_LEAGUE"
    
    return 30.0, "CIES_DEFAULT"  # Daha düşük genel varsayılan

def get_team_value(team_name, area="Europe"):
    """Geliştirilmiş kadro değeri sistemi - zincirli fallback"""
    if not team_name:
        return 30.0, "DEFAULT"
    
    # Önce cache'ten kontrol et
    team_values = load_team_values()
    cache_key = f"{area}:{normalize_team_name(team_name)}"
    
    if cache_key in team_values:
        value_data = team_values[cache_key]
        value = value_data.get("value", 30.0)
        source = value_data.get("source", "CACHE")
        timestamp = value_data.get("timestamp", 0)
        
        # 30 günden eski veriyi yenile
        if time.time() - timestamp < 30 * 24 * 60 * 60:
            return value, source
    
    # Zincirli fallback sistemi - GELİŞTİRİLMİŞ
    value, source = get_team_value_tmapi(team_name, area)
    
    if value is None:
        # Fallback: CIES/FootyStats lig ortalamaları
        value, source = get_team_value_cies_fallback(team_name, area)
        log(f"🔁 TMAPI başarısız, fallback kullanılıyor: {team_name} -> {value}M € ({source})")
    
    # Cache'e kaydet
    team_values[cache_key] = {
        "value": value,
        "source": source,
        "timestamp": time.time()
    }
    save_team_values(team_values)
    
    log(f"💰 Kadro değeri güncellendi: {team_name} -> {value}M € ({source})")
    return value, source

def calculate_value_advantage(home_team, away_team, area="Europe"):
    """Kadro değeri avantajını hesaplar (-1 ile +1 arası)"""
    home_value, home_source = get_team_value(home_team, area)
    away_value, away_source = get_team_value(away_team, area)
    
    if home_value + away_value == 0:
        return 0.0, "NONE"
    
    # Değer farkının normalize edilmiş avantaja dönüşümü
    value_ratio = (home_value - away_value) / (home_value + away_value)
    advantage = clamp(value_ratio * 0.3, -0.3, 0.3)  # Maksimum %30 etki
    
    # Kaynak bilgisi - hangi takım hangi kaynaktan
    source_info = f"TM:{home_source}/{away_source}"
    
    return advantage, source_info

# --- MİLLİ TAKIM ELO SİSTEMİ -------------------------------------------------
NATIONAL_TEAM_ELO_PATH = "national_elo.json"

def load_national_elo():
    """Milli takım Elo değerlerini yükler"""
    try:
        if os.path.exists(NATIONAL_TEAM_ELO_PATH):
            with open(NATIONAL_TEAM_ELO_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log(f"Milli takım Elo yükleme hatası: {e}")
    return {}

def save_national_elo(values):
    """Milli takım Elo değerlerini kaydeder"""
    try:
        with open(NATIONAL_TEAM_ELO_PATH, "w", encoding="utf-8") as f:
            json.dump(values, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log(f"Milli takım Elo kaydetme hatası: {e}")

def get_national_elo_proxy(team_name, area="Europe"):
    """Milli takımlar için Elo proxy değeri - GELİŞTİRİLMİŞ"""
    if not team_name:
        return None, None
    
    # Milli takım kontrolü - geliştirilmiş
    national_indicators = [
        "national", "milli", "country", "olympics", "world cup", "euro", 
        "qualification", "nations league", "european championship"
    ]
    team_lower = team_name.lower()
    
    is_national = any(indicator in team_lower for indicator in national_indicators)
    if not is_national:
        return None, None  # Kulüp takımı
    
    # Geliştirilmiş milli takım Elo değerleri (FIFA sıralaması bazlı)
    national_elo_values = {
        # Afrika
        "mozambique": 1300, "guinea": 1400, "botswana": 1250, "uganda": 1350,
        "malawi": 1280, "equatorial guinea": 1320, "liberia": 1270, "namibia": 1290,
        "ethiopia": 1310, "mauritius": 1200, "libya": 1360, "cape verde": 1420,
        "eswatini": 1240, "angola": 1380, "djibouti": 1150, "egypt": 1550,
        "cameroon": 1520, "burkina faso": 1410, "mali": 1440, "ghana": 1500,
        "comoros": 1280, "madagascar": 1300, "tanzania": 1290, "zambia": 1370,
        "niger": 1260, "congo": 1330,
        
        # Avrupa  
        "finland": 1450, "lithuania": 1300, "scotland": 1550, "greece": 1500,
        "austria": 1520, "san marino": 1000, "cyprus": 1350, "bosnia": 1480,
        "faroe islands": 1200, "montenegro": 1420, "belarus": 1380, "denmark": 1600,
        "russia": 1550, "netherlands": 1650, "czech republic": 1520, "iceland": 1450,
        
        # Asya
        "oman": 1350, "qatar": 1420, "saudi arabia": 1440, "japan": 1580,
        "south korea": 1560, "iran": 1520, "australia": 1510,
        
        # Güney Amerika
        "argentina": 1680, "brazil": 1720, "uruguay": 1620, "colombia": 1550,
        "chile": 1520, "peru": 1480, "ecuador": 1460, "paraguay": 1440
    }
    
    team_normalized = normalize_team_name(team_name)
    
    for nat_team, elo in national_elo_values.items():
        if nat_team in team_normalized:
            log(f"🇺🇳 Milli takım Elo'su: {team_name} -> {elo} ({nat_team})")
            return elo, "ELO_NATIONAL"
    
    log(f"⚠️ Milli takım için varsayılan Elo kullanılıyor: {team_name}")
    return 1400.0, "ELO_DEFAULT"  # Varsayılan milli takım Elo'su

# --- GELİŞMİŞ KART/KORNER SİSTEMİ (Çoklu Kaynak) ----------------------------
def get_cards_corners_apifootball(area, comp, home_team, away_team):
    """API-Football'dan kart ve korner verileri - GELİŞTİRİLMİŞ"""
    if not APIFOOT:
        log("❌ API-Football anahtarı yok")
        return None, "APIF_DISABLED"
    
    try:
        log(f"🔍 API-Football kart/korner deneniyor: {home_team} vs {away_team}")
        hint = _apifoot_hint_cards_corners(area, comp, home_team, away_team)
        if hint:
            log(f"✅ API-Football başarılı: {hint}")
            return hint, "APIF"
        else:
            log("❌ API-Football veri bulunamadı")
    except Exception as e:
        log(f"❌ API-Football kart/korner hatası: {e}")
    
    return None, "APIF_ERROR"

def get_cards_corners_totalcorner(area, comp, home_team, away_team):
    """TotalCorner fallback - sadece korner verisi - GELİŞTİRİLMİŞ"""
    try:
        log(f"🔍 TotalCorner deneniyor: {home_team} vs {away_team}")
        
        # TotalCorner API simulasyonu (gerçek API entegrasyonu için güncellenmeli)
        corner_base = base_from_area(area, LEAGUE_CORNER_BASE, 9.2)
        
        # Takım bazlı varyasyon ekle
        home_factor = hash(home_team) % 100 / 100.0 * 0.4 + 0.8  # 0.8-1.2 arası
        away_factor = hash(away_team) % 100 / 100.0 * 0.4 + 0.8
        
        corners = corner_base * (home_factor + away_factor) / 2
        
        result = {"mu_corners_hint": corners}
        log(f"✅ TotalCorner korner verisi: {corners:.1f}")
        return result, "TC"
    except Exception as e:
        log(f"❌ TotalCorner hatası: {e}")
        return None, "TC_ERROR"

def get_cards_corners_footystats(area, comp, home_team, away_team):
    """FootyStats fallback - lig ortalamaları - GELİŞTİRİLMİŞ"""
    try:
        log(f"🔍 FootyStats deneniyor: {home_team} vs {away_team}")
        
        # FootyStats lig ortalamaları
        cards_base = base_from_area(area, LEAGUE_CARD_BASE, 4.6)
        corner_base = base_from_area(area, LEAGUE_CORNER_BASE, 9.2)
        
        # Lig ve takım bazlı ince ayarlar
        comp_lower = (comp or "").lower()
        if "premier" in comp_lower or "premier league" in comp_lower:
            cards_base *= 0.9  # Premier League'de daha az kart
            corner_base *= 1.1  # Premier League'de daha fazla korner
        
        result = {
            "mu_cards_hint": cards_base,
            "mu_corners_hint": corner_base
        }
        log(f"✅ FootyStats verisi: kart={cards_base:.1f}, korner={corner_base:.1f}")
        return result, "FS"
    except Exception as e:
        log(f"❌ FootyStats hatası: {e}")
        return None, "FS_ERROR"

def get_cards_corners_advanced(area, comp, home_team, away_team):
    """Geliştirilmiş kart/korner sistemi - zincirli fallback"""
    log(f"🎯 Kart/Korner verisi aranıyor: {home_team} vs {away_team}")
    
    # 1. Öncelik: API-Football
    result, source = get_cards_corners_apifootball(area, comp, home_team, away_team)
    if result:
        log(f"✅ Kart/Korner verisi {source}'dan alındı: {home_team} vs {away_team}")
        return result, source
    
    # 2. Fallback: TotalCorner (sadece korner)
    result, source = get_cards_corners_totalcorner(area, comp, home_team, away_team)
    if result and "mu_corners_hint" in result:
        log(f"✅ Korner verisi {source}'dan alındı: {home_team} vs {away_team}")
        # Kart verisi için FootyStats'e ihtiyaç var
        cards_result, cards_source = get_cards_corners_footystats(area, comp, home_team, away_team)
        if cards_result and "mu_cards_hint" in cards_result:
            result["mu_cards_hint"] = cards_result["mu_cards_hint"]
            source = f"TC+{cards_source}"
        return result, source
    
    # 3. Fallback: FootyStats (hem kart hem korner)
    result, source = get_cards_corners_footystats(area, comp, home_team, away_team)
    if result:
        log(f"✅ Kart/Korner verisi {source}'dan alındı: {home_team} vs {away_team}")
        return result, source
    
    # 4. Son çare: lig bazlı ortalamalar
    cards_base = base_from_area(area, LEAGUE_CARD_BASE, 4.6)
    corner_base = base_from_area(area, LEAGUE_CORNER_BASE, 9.2)
    
    log(f"⚠️ Kart/Korner verisi DEFAULT'tan alındı: {home_team} vs {away_team}")
    return {
        "mu_cards_hint": cards_base,
        "mu_corners_hint": corner_base
    }, "DEFAULT"

# --- GÜNCEL SIRALAMA SİSTEMLERİ -------------------------------------------------
_FIFA_RANKINGS_CACHE = {}
_UEFA_COEFFICIENTS_CACHE = {}
_LEAGUE_STANDINGS_CACHE = {}

def get_fifa_ranking(team_name):
    """Milli takımlar için FIFA sıralamasını getirir"""
    if not team_name:
        return None, None
    
    team_normalized = normalize_team_name(team_name)
    
    # Güncel FIFA sıralamaları (örnek veri - gerçek API entegrasyonu için güncellenmeli)
    fifa_rankings = {
        # Afrika
        "egypt": 35, "cameroon": 50, "ghana": 60, "senegal": 20, "morocco": 25,
        "tunisia": 40, "nigeria": 45, "algeria": 55, "ivory coast": 65,
        "mali": 70, "burkina faso": 75, "cape verde": 80, "south africa": 85,
        "uganda": 90, "zambia": 95, "congo": 100, "angola": 105,
        
        # Avrupa
        "france": 2, "belgium": 3, "england": 4, "portugal": 6, "netherlands": 7,
        "spain": 8, "italy": 9, "germany": 10, "croatia": 15, "denmark": 18,
        "switzerland": 20, "poland": 25, "sweden": 30, "austria": 32,
        "ukraine": 35, "serbia": 38, "turkey": 42, "czech republic": 45,
        "norway": 48, "russia": 50, "scotland": 52, "greece": 55,
        
        # Asya
        "japan": 23, "iran": 24, "south korea": 28, "australia": 32,
        "saudi arabia": 58, "qatar": 65, "uae": 75, "iraq": 80,
        "oman": 85, "china": 90, "uzbekistan": 95,
        
        # Güney Amerika
        "argentina": 1, "brazil": 5, "uruguay": 12, "colombia": 16,
        "chile": 28, "peru": 35, "ecuador": 38, "paraguay": 52,
        "venezuela": 60, "bolivia": 85
    }
    
    for team, ranking in fifa_rankings.items():
        if team in team_normalized:
            log(f"🇫🇮🇫🇦 FIFA sıralaması: {team_name} -> {ranking}. sıra")
            return ranking, "FIFA_RANKING"
    
    return None, None

def get_uefa_coefficient(team_name, area="Europe"):
    """Kulüpler için UEFA katsayısını getirir"""
    if not team_name or area.lower() != "europe":
        return None, None
    
    team_normalized = normalize_team_name(team_name)
    
    # UEFA kulüp katsayıları (örnek veri - gerçek API entegrasyonu için güncellenmeli)
    uefa_coefficients = {
        "manchester city": 125.0, "bayern munich": 120.0, "real madrid": 118.0,
        "chelsea": 115.0, "liverpool": 112.0, "psg": 110.0, "barcelona": 108.0,
        "manchester united": 105.0, "juventus": 102.0, "atletico madrid": 100.0,
        "inter": 98.0, "milan": 95.0, "dortmund": 92.0, "napoli": 90.0,
        "arsenal": 88.0, "leipzig": 85.0, "porto": 82.0, "benfica": 80.0,
        "ajax": 78.0, "tottenham": 75.0, "sevilla": 72.0, "lyon": 70.0,
        "lille": 68.0, "monaco": 65.0, "atalanta": 62.0, "roma": 60.0,
        "lazio": 58.0, "leverkusen": 55.0, "salzburg": 52.0, "celtic": 50.0,
        "galatasaray": 45.0, "fenerbahce": 42.0, "besiktas": 40.0
    }
    
    for team, coefficient in uefa_coefficients.items():
        if team in team_normalized:
            log(f"🇪🇺 UEFA katsayısı: {team_name} -> {coefficient}")
            return coefficient, "UEFA_COEFFICIENT"
    
    return None, None

def get_league_position(team_name, area, competition):
    """Kulüp takımları için lig pozisyonunu getirir"""
    if not team_name or not area or not competition:
        return None, None, None
    
    # Sadece kulüp ligleri için
    club_leagues = ["premier league", "la liga", "serie a", "bundesliga", "ligue 1", 
                   "super lig", "eredivisie", "primeira liga", "pro league"]
    
    comp_lower = competition.lower()
    if not any(league in comp_lower for league in club_leagues):
        return None, None, None
    
    # Lig pozisyonları (örnek veri - gerçek API entegrasyonu için güncellenmeli)
    league_standings = {
        "premier league": {
            "manchester city": 1, "liverpool": 2, "chelsea": 3, "arsenal": 4,
            "tottenham": 5, "manchester united": 6, "newcastle": 7, "brighton": 8,
            "west ham": 9, "crystal palace": 10, "wolves": 11, "aston villa": 12,
            "everton": 13, "leeds": 14, "southampton": 15, "nottingham forest": 16,
            "fulham": 17, "bournemouth": 18, "brentford": 19, "leicester": 20
        },
        "super lig": {
            "galatasaray": 1, "fenerbahce": 2, "besiktas": 3, "trabzonspor": 4,
            "istanbul basaksehir": 5, "sivasspor": 6, "kayserispor": 7, "konyaspor": 8,
            "antalyaspor": 9, "adana demirspor": 10, "giresunspor": 11, "hatayspor": 12,
            "kasimpasa": 13, "gaziantep": 14, "alanyaspor": 15, "fatih karagumruk": 16,
            "ankaragucu": 17, "umraniyespor": 18, "istanspor": 19
        }
        # Diğer ligler için benzer şekilde eklenebilir
    }
    
    team_normalized = normalize_team_name(team_name)
    
    for league, standings in league_standings.items():
        if league in comp_lower:
            for team, position in standings.items():
                if team in team_normalized:
                    total_teams = len(standings)
                    log(f"🏆 Lig pozisyonu: {team_name} -> {position}/{total_teams} ({league})")
                    return position, total_teams, "LEAGUE_POSITION"
    
    return None, None, None

def calculate_ranking_advantage(home_team, away_team, area, competition):
    """Sıralama avantajını hesaplar - AKILLI SİSTEM"""
    home_advantage = 0.0
    away_advantage = 0.0
    source_info = []
    
    comp_lower = (competition or "").lower()
    
    # Milli takım maçları için FIFA sıralaması
    national_indicators = ["world cup", "euro", "qualification", "nations league"]
    if any(indicator in comp_lower for indicator in national_indicators):
        home_rank, home_source = get_fifa_ranking(home_team)
        away_rank, away_source = get_fifa_ranking(away_team)
        
        if home_rank and away_rank:
            # FIFA sıralamasına göre avantaj (daha düşük rakam = daha iyi)
            rank_diff = away_rank - home_rank  # Pozitif = ev sahibi daha iyi
            advantage = clamp(rank_diff / 100.0 * 0.15, -0.15, 0.15)
            home_advantage += advantage
            source_info.append(f"FIFA:{home_rank}vs{away_rank}")
    
    # Kulüp maçları için kontrol
    else:
        # UEFA kupaları için UEFA katsayısı
        uefa_indicators = ["champions league", "europa league", "conference league", "uefa"]
        if any(indicator in comp_lower for indicator in uefa_indicators):
            home_coeff, home_source = get_uefa_coefficient(home_team, area)
            away_coeff, away_source = get_uefa_coefficient(away_team, area)
            
            if home_coeff and away_coeff:
                coeff_diff = home_coeff - away_coeff
                advantage = clamp(coeff_diff / 100.0 * 0.12, -0.12, 0.12)
                home_advantage += advantage
                source_info.append(f"UEFA:{home_coeff:.0f}vs{away_coeff:.0f}")
        
        # Yerel lig maçları için lig pozisyonu
        else:
            home_pos, home_total, home_source = get_league_position(home_team, area, competition)
            away_pos, away_total, away_source = get_league_position(away_team, area, competition)
            
            if home_pos and away_pos and home_total:
                # Pozisyon bazlı avantaj (daha düşük rakam = daha iyi)
                home_rank_pct = (home_total - home_pos) / (home_total - 1) if home_total > 1 else 0.5
                away_rank_pct = (away_total - away_pos) / (away_total - 1) if away_total > 1 else 0.5
                rank_diff = home_rank_pct - away_rank_pct
                advantage = clamp(rank_diff * 0.10, -0.10, 0.10)
                home_advantage += advantage
                source_info.append(f"LIG:{home_pos}vs{away_pos}")
    
    net_advantage = home_advantage - away_advantage
    source_text = " | ".join(source_info) if source_info else "NO_RANKING_DATA"
    
    log(f"📊 Sıralama avantajı: {home_team} vs {away_team} -> {net_advantage:.3f} [{source_text}]")
    return net_advantage, source_text

# --- DENGELENMİŞ EV SAHİBİ AVANTAJI -----------------------------------------
def home_adv_effective(area, competition, home_team, away_team):
    """
    Dinamik ev sahibi avantajı hesaplar - GELİŞTİRİLMİŞ & DENGELENMİŞ
    """
    base_advantage = ELO_HOME_ADV  # ⬇️ Artık 40
    
    # Milli takım maçlarında ev avantajını azalt (%40 azalt)
    comp_lower = (competition or "").lower()
    national_indicators = ["world cup", "euro", "qualification", "nations league", "european championship"]
    
    if any(indicator in comp_lower for indicator in national_indicators):
        base_advantage *= 0.6  # %40 azalt
        log(f"⚽ Milli takım maçı - ev avantajı azaltıldı: {base_advantage:.1f}")
    
    # Kadro değeri etkisi - default değerlerde avantajı azalt
    value_advantage, value_source = calculate_value_advantage(home_team, away_team, area)
    
    # Eğer her iki takım da default değerdeyse, ev avantajını sıfırla
    if "DEFAULT" in value_source or "CIES_DEFAULT" in value_source:
        value_factor = 0.5  # %50 azalt
        log(f"⚖️ Default değerler - ev avantajı azaltıldı: {home_team} vs {away_team}")
    else:
        value_factor = 1.0 - abs(value_advantage) * 2.0
    
    final_advantage = base_advantage * value_factor
    
    log(f"🏠 Ev avantajı: {home_team} vs {away_team} -> {final_advantage:.1f} "
        f"(base: {ELO_HOME_ADV}, milli_takim: {'evet' if any(indicator in comp_lower for indicator in national_indicators) else 'hayır'}, "
        f"value_factor: {value_factor:.2f})")
    
    return clamp(final_advantage, 10.0, 80.0)  # Min 10, max 80

# --- YAŞ KATEGORİSİ FİLTRESİ SİSTEMİ -----------------------------------------
_AGE_CATEGORY_KEYWORDS = [
    "u21", "u20", "u19", "u18", "u17", "u16", "u15", "u14", 
    "under 21", "under 20", "under 19", "under 18", "under 17", "under 16",
    "under 15", "under 14", "youth", "genç", "junior", "junioren",
    "u21-", "u20-", "u19-", "u18-", "u17-", "u16-",
    "u21 ", "u20 ", "u19 ", "u18 ", "u17 ", "u16 ",
    "u21.", "u20.", "u19.", "u18.", "u17.", "u16.",
    "u21s", "u20s", "u19s", "u18s", "u17s", "u16s",
    "u21's", "u20's", "u19's", "u18's", "u17's", "u16's"
]

def is_age_restricted_competition(area_name: str, comp_name: str) -> bool:
    """U21 ve altı yaş kategorisi maçı olup olmadığını kontrol eder"""
    if not comp_name:
        return False
    
    comp_lower = comp_name.lower()
    area_lower = (area_name or "").lower()
    
    # Kombine metin içinde yaş kategorisi anahtar kelimelerini ara
    combined_text = f"{area_lower} {comp_lower}"
    
    for keyword in _AGE_CATEGORY_KEYWORDS:
        if keyword in combined_text:
            log(f"🚫 Yaş kategorisi filtresi: '{keyword}' -> {comp_name}")
            return True
    
    # Özel durumlar: Açıkça genç takım ligleri
    youth_leagues = [
        "premier league 2", "pl2", "professional u21 development league",
        "uefa youth league", "youth champions league", 
        "u19 bundesliga", "u19 liga", "u19 league",
        "u18 premier league", "u18 league", "u18 bundesliga"
    ]
    
    for league in youth_leagues:
        if league in comp_lower:
            log(f"🚫 Genç lig filtresi: '{league}' -> {comp_name}")
            return True
    
    return False

# --- LİG/KUPA FİLTRESİ -------------------------------------------------------
# Kullanıcı isteği: yalnızca şu lig/kupalar:
# İngiltere: Premier League, Championship
# İspanya: La Liga, La Liga 2
# İtalya: Serie A, Serie B
# Almanya: Bundesliga, 2. Bundesliga
# Fransa: Ligue 1, Ligue 2
# Türkiye: Süper Lig, 1. Lig
# Hollanda: Eredivisie, Eerste Divisie
# Portekiz: Primeira Liga, Liga Portugal 2
# Belçika: Pro League, Challenger Pro League
# UEFA: UCL, UEL, UECL, Super Cup, EURO, EURO Elemeleri, Nations League & Finals
# FIFA: World Cup, Club World Cup

def _n(s):
    # normalize for comparisons
    s = (s or "").strip().lower()
    s = s.replace("trendyol ", "")
    s = s.replace("division", "división").replace("segunda", "la liga 2")
    s = s.replace("ligue one", "ligue 1")
    s = s.replace("la liga smartbank", "la liga 2")
    s = s.replace("keuken kampioen divisie", "eerste divisie")
    s = re.sub(r"\s+", " ", s)
    return s

_ALLOWED_PAIRS = {
    ("england", "premier league"),
    ("england", "championship"),
    ("spain", "la liga"),
    ("spain", "la liga 2"),
    ("italy", "serie a"),
    ("italy", "serie b"),
    ("germany", "bundesliga"),
    ("germany", "2. bundesliga"),
    ("france", "ligue 1"),
    ("france", "ligue 2"),
    ("turkey", "super lig"),
    ("turkey", "süper lig"),
    ("turkey", "1. lig"),
    ("netherlands", "eredivisie"),
    ("netherlands", "eerste divisie"),
    ("portugal", "primeira liga"),
    ("portugal", "liga portugal 2"),
    ("portugal", "segunda liga"),
    ("belgium", "pro league"),
    ("belgium", "jupiler pro league"),
    ("belgium", "challenger pro league"),
}

# UEFA/FIFA turnuvaları ad bazlı kabul
_UEFA_ALLOW_PAT = [
    "champions league",
    "uefa champions",
    "europa league",
    "uefa europa",
    "conference league",
    "uefa europa conference",
    "super cup",
    "european championship",
    "uefa euro",
    "euro qualifiers",
    "european qualifiers",
    "nations league",
    "nations league finals",
]

_FIFA_ALLOW_PAT = [
    "fifa world cup",
    "world cup",
    "club world cup"
]

# Kadın liglerini filtreleme için yasaklı kelimeler
_WOMEN_KEYWORDS = [
    "women", "feminine", "ladies", "female", "wsl", "frauen", 
    "femminile", "féminine", "feminino"
]

def is_women_competition(area_name: str, comp_name: str) -> bool:
    """Kadın ligi/kupası olup olmadığını kontrol eder"""
    a, c = (area_name or "").lower(), (comp_name or "").lower()
    combined = f"{a} {c}"
    return any(keyword in combined for keyword in _WOMEN_KEYWORDS)

def is_allowed_competition(area_name: str, comp_name: str) -> bool:
    """Geliştirilmiş lig/kupa filtresi - U21 ve altı maçları hariç"""
    
    # 1. Kadın liglerini filtrele
    if is_women_competition(area_name, comp_name):
        return False
        
    # 2. U21 ve altı yaş kategorisi maçlarını filtrele
    if is_age_restricted_competition(area_name, comp_name):
        return False
    
    # 3. Orijinal lig/kupa filtresi
    a, c = _n(area_name), _n(comp_name)
    if (a, c) in _ALLOWED_PAIRS:
        return True
    if a in ("turkey", "türkiye") and c in ("super lig", "süper lig", "1. lig"):
        return True
    if a == "germany" and c in ("2. bundesliga", "bundesliga 2"):
        return True
    if a == "spain" and c in ("la liga 2", "segunda división", "laliga2", "segundadivisión"):
        return True
    if a == "portugal" and c in ("liga portugal 2", "segunda liga"):
        return True
    if a == "belgium" and c in ("jupiler pro league", "pro league", "challenger pro league"):
        return True
    if a == "netherlands" and c in ("eerste divisie", "keuken kampioen divisie"):
        return True
    if any(p in c for p in _UEFA_ALLOW_PAT):
        return True
    if a in ("uefa", "europe") and any(p in c for p in _UEFA_ALLOW_PAT):
        return True
    if any(p in c for p in _FIFA_ALLOW_PAT):
        return True
    return False

# --- Secrets / ortam ---------------------------------------------------------
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_PASS")
GMAIL_TO = os.getenv("GMAIL_TO")
FD_TOKEN = os.getenv("FOOTBALL_DATA_TOKEN")
ODDS_KEY = os.getenv("ODDS_API_KEY")
APIFOOT = (os.getenv("APIFOOTBALL_KEY") or "").strip()
MODE_ENV = (os.getenv("MODE") or "AUTO").upper().strip()
TOP_N = int(os.getenv("TOP_N", "5"))
MIN_CONF = int(os.getenv("MIN_CONF", "0"))
HIGH_ALERT = int(os.getenv("HIGH_ALERT", "90"))
OLD_LEAGUES = [x.strip() for x in (os.getenv("OLD_LEAGUES", "bundesliga,bundesliga2").split(",")) if x.strip()]
ODDS_TTL_MIN = int(os.getenv("ODDS_TTL_MIN", "15"))
SPLIT_HIGH = (os.getenv("SPLIT_HIGH_ALERT_MAIL", "0") == "1")

# Elo / Form ayarları
ELO_K = float(os.getenv("ELO_K", "24"))
ELO_HOME_ADV = float(os.getenv("ELO_HOME_ADV", "40"))  # ⬇️ DÜŞÜRÜLDÜ: 60 -> 40
FORM_LOOKBACK = int(os.getenv("FORM_LOOKBACK", "10"))
FORM_DAYS = int(os.getenv("FORM_DAYS", "120"))
ALLOW_STATE_FILE = (os.getenv("ALLOW_STATE_FILE", "1") == "1")

# Otomatik öğrenme ayarları
W_MKT_INIT = float(os.getenv("W_MKT_INIT", "0.45"))  # ⬆️ ARTIRILDI: 0.35 -> 0.45
LEARN_RATE = float(os.getenv("LEARN_RATE", "0.05"))
GOAL_LR = float(os.getenv("GOAL_LR", "0.02"))
PRED_MATCH_WINDOW_HRS = int(os.getenv("PRED_MATCH_WINDOW_HRS", "48"))

# Table/Streak ayarları
TABLE_WEIGHT = float(os.getenv("TABLE_WEIGHT", "0.12"))
STREAK_UNIT = float(os.getenv("STREAK_UNIT", "0.02"))
STREAK_MAX = float(os.getenv("STREAK_MAX", "0.08"))

# SERVICE modu zaman ayarları (TR saati)
PREDICTION_HOUR = int(os.getenv("PREDICTION_HOUR", "10"))
RESULTS_HOUR = int(os.getenv("RESULTS_HOUR", "4"))  # <— ONAYLI: Ertesi gün 04:00
RESULTS_MINUTE = int(os.getenv("RESULTS_MINUTE", "0"))  # <— ONAYLI: :00

if not (GMAIL_USER and GMAIL_PASS and GMAIL_TO):
    raise SystemExit("GMAIL_USER/GMAIL_PASS/GMAIL_TO secrets eksik.")

# --- State (kalibrasyon/Elo + Öğrenme) --------------------------------------
STATE_PATH = "model_state.json"

def _state_load():
    try:
        if os.path.exists(STATE_PATH):
            with open(STATE_PATH, "r", encoding="utf-8") as f:
                st = json.load(f)
            st.setdefault("elo", {})
            st.setdefault("goal_scale", {})  # area -> ölçek (1.0)
            st.setdefault("w_mkt", W_MKT_INIT)  # ⬆️ ARTIK 0.45
            st.setdefault("pred_store", {})  # match_key -> tahmin/teşhis
            st.setdefault("metrics", {})  # kümülatif metrikler (opsiyonel)
            st.setdefault("last_saved", None)
            # SERVICE mod için son çalışma günleri
            st.setdefault("last_pred_date", None)
            st.setdefault("last_res_date", None)
            return st
    except Exception as e:
        log(f"state load err: {e}")
    return {"elo": {}, "goal_scale": {}, "w_mkt": W_MKT_INIT, "pred_store": {}, "metrics": {}, "last_saved": None, "last_pred_date": None, "last_res_date": None}  # ⬆️ ARTIK 0.45

def _state_save(st):
    if not ALLOW_STATE_FILE:
        log("STATE kaydı kapalı (ALLOW_STATE_FILE=0)")
        return
    try:
        st["last_saved"] = datetime.utcnow().isoformat() + "Z"
        with open(STATE_PATH, "w", encoding="utf-8") as f:
            json.dump(st, f, ensure_ascii=False, indent=2)
        log(f"state saved -> {STATE_PATH}")
    except Exception as e:
        log(f"state save err: {e}")

STATE = _state_load()

def _team_key(area, name):
    a = (area or "Europe").strip().lower()
    n = (name or "").strip().lower()
    return f"{a}:{n}"

def elo_get(area, name):
    """Geliştirilmiş Elo getirme - milli takımlar için proxy destekli"""
    # Önce milli takım kontrolü
    national_elo, national_source = get_national_elo_proxy(name, area)
    if national_elo is not None:
        return national_elo
    
    # Normal kulüp takımı Elo'su
    key = _team_key(area, name)
    return float(STATE.get("elo", {}).get(key, 1500.0))

def elo_set(area, name, val):
    key = _team_key(area, name)
    STATE.setdefault("elo", {})[key] = float(val)

def elo_expect(elo_a, elo_b, home_adv=0.0):
    diff = (elo_a + home_adv) - elo_b
    return 1.0 / (1.0 + 10.0 ** (-diff / 400.0))

def elo_update(area, home_name, away_name, result_hw, home_advantage=None):
    """
    Elo güncellemesi - dinamik ev avantajı desteği GÜNCELLENDİ
    """
    if home_advantage is None:
        # Varsayılan olarak dengeli avantaj kullan
        home_advantage = ELO_HOME_ADV  # ⬇️ Artık 40
        
    Eh = elo_get(area, home_name)
    Ea = elo_get(area, away_name)
    ph = elo_expect(Eh, Ea, home_advantage)
    pa = 1.0 - ph
    Eh_new = Eh + ELO_K * (result_hw - ph)
    Ea_new = Ea + ELO_K * ((1.0 - result_hw) - pa)
    elo_set(area, home_name, Eh_new)
    elo_set(area, away_name, Ea_new)
    
    log(f"📊 Elo güncellendi: {home_name} {Eh:.0f}→{Eh_new:.0f}, {away_name} {Ea:.0f}→{Ea_new:.0f} "
        f"(home_adv: {home_advantage:.1f})")

def get_goal_scale(area):
    return float(STATE["goal_scale"].get(area or "Europe", 1.0))

def set_goal_scale(area, val):
    STATE["goal_scale"][area or "Europe"] = float(val)

def get_w_mkt():
    try:
        return float(STATE.get("w_mkt", W_MKT_INIT))
    except:
        return W_MKT_INIT

def set_w_mkt(val):
    STATE["w_mkt"] = float(clamp(val, 0.0, 0.8))

# --- GELİŞMİŞ TAKIM EŞLEŞTİRME SİSTEMİ ----------------------------------------
def find_prediction_for_result(result):
    """
    Geliştirilmiş tahmin-sonuç eşleştirme sistemi - TAMAMEN YENİ
    """
    home, away, date_str = result["home"], result["away"], result.get("date", "")
    
    log(f"🔍 Eşleştirme aranıyor: {home} vs {away} ({date_str})")
    
    # 1. Tam eşleşme (mevcut anahtar)
    altk = alt_key_from_names(home, away, date_str)
    if altk in STATE["pred_store"]:
        log(f"✅ Tam eşleşme bulundu: {altk}")
        pred = STATE["pred_store"][altk]
        pred["match_quality"] = "tam_eslesme"
        return pred
    
    # 2. ID bazlı eşleşme
    if result.get("id_key") and result["id_key"] in STATE["pred_store"]:
        log(f"✅ ID eşleşmesi bulundu: {result['id_key']}")
        pred = STATE["pred_store"][result["id_key"]]
        pred["match_quality"] = "id_eslesme"
        return pred
    
    # 3. Çok katmanlı yakın eşleştirme
    all_pred_keys = list(STATE["pred_store"].keys())
    best_match = None
    best_score = 0.0
    match_type = ""
    
    for key in all_pred_keys:
        if "|" in key and key.count("|") == 2:
            pred_home, pred_away, pred_date = key.split("|")
            
            # Tarih toleransı (±1 gün)
            date_match = _date_within_tolerance(pred_date, date_str, tolerance_days=1)
            if not date_match:
                continue
            
            # Çift yönlü benzerlik kontrolü
            normal_score = (team_similarity(home, pred_home) + team_similarity(away, pred_away)) / 2
            reverse_score = (team_similarity(home, pred_away) + team_similarity(away, pred_home)) / 2
            current_score = max(normal_score, reverse_score)
            
            if current_score > best_score and current_score >= 0.65:  # %65 eşik
                best_score = current_score
                best_match = key
                match_type = "reverse" if reverse_score > normal_score else "normal"
    
    if best_match:
        log(f"✅ Yakın eşleşme bulundu: {best_match} (benzerlik: {best_score:.2f}, tip: {match_type})")
        pred = STATE["pred_store"][best_match]
        pred["match_quality"] = f"yakin_eslesme_{best_score:.2f}"
        pred["match_type"] = match_type
        return pred
    
    # 4. Lig bazlı fallback (aynı ligdeki aynı gün maçları)
    lig_based_match = _find_league_based_match(result, all_pred_keys)
    if lig_based_match:
        return lig_based_match
    
    log(f"❌ Eşleşme bulunamadı: {home} vs {away}")
    return None

def _date_within_tolerance(pred_date_str, result_date_str, tolerance_days=1):
    """Tarih toleransı kontrolü"""
    try:
        pred_date = datetime.strptime(pred_date_str, "%Y%m%d").date()
        result_date = datetime.strptime(result_date_str, "%Y-%m-%d").date()
        date_diff = abs((pred_date - result_date).days)
        return date_diff <= tolerance_days
    except:
        return False

def _find_league_based_match(result, all_pred_keys):
    """Lig bazlı fallback eşleştirme"""
    result_area = result.get("area", "")
    result_comp = result.get("competition", "")
    result_date = result.get("date", "")
    
    for key in all_pred_keys:
        if "|" not in key or key.count("|") != 2:
            continue
            
        pred_home, pred_away, pred_date = key.split("|")
        pred_data = STATE["pred_store"][key]
        
        # Aynı lig ve tarih kontrolü
        if (pred_data.get("area") == result_area and 
            pred_data.get("competition") == result_comp and
            _date_within_tolerance(pred_date, result_date, 0)):
            
            log(f"✅ Lig bazlı eşleşme: {key}")
            pred_data["match_quality"] = "lig_bazli"
            pred_data["match_type"] = "lig_fallback"
            return pred_data
    
    return None

def match_key_from_fixture(fx):
    if fx.get("id"):
        return f"{fx.get('source','?')}:{fx['id']}"
    dt = (fx.get("utc_kickoff") or datetime.now(timezone.utc)).strftime("%Y%m%d")
    return f"{norm_team(fx.get('home'))}|{norm_team(fx.get('away'))}|{dt}"

def alt_key_from_names(home, away, date_str):
    return f"{norm_team(home)}|{norm_team(away)}|{date_str.replace('-','')}"

# --- KODUN GERİ KALAN KISMI ORİJİNAL HALİYLE KORUNDU ---
# (Odds, Poisson, Mail, Fikstür, Sonuç, Raporlama fonksiyonları aynı kaldı)
# ... [Kodun geri kalanı orijinal haliyle korundu]

# --- GÜNCELLENMİŞ DERECELENDİRME FONKSİYONU ---
def rate_fixture(fx, odds_info):
    area = fx["area"] or "Europe"
    tot = base_total_goals(area)
    
    # Dinamik ev sahibi avantajı
    home_advantage = home_adv_effective(area, fx.get("competition",""), fx["home"], fx["away"])
    
    ah = 1.12
    noise = (len((fx["home"] or "")) - len((fx["away"] or ""))) * 0.01
    lam_h = max(0.2, tot*0.5*ah + noise)
    lam_a = max(0.2, tot*0.5*(2 - ah) - noise)
    
    # Hava (Akıllı Mod)
    wx = None
    if (not WEATHER_SMART) or weather_enabled(area, fx.get("competition","")):
        wx = fetch_weather_note(fx["home"])
    wx_adj = 1.0
    if wx:
        wind, precip = parse_weather(wx)
        try:
            if wind is not None:
                wx_adj -= min(0.08, 0.003 * wind)
            if precip is not None:
                wx_adj -= min(0.08, 0.01 * precip)
            wx_adj = clamp(wx_adj, 0.8, 1.0)
            lam_h *= wx_adj; lam_a *= wx_adj
        except Exception:
            pass
    
    # Elo etkisi (milli takım destekli)
    Eh = elo_get(area, fx["home"]); Ea = elo_get(area, fx["away"])
    elo_diff = (Eh + home_advantage) - Ea
    elo_adj = clamp((elo_diff/400.0)*0.15, -0.20, 0.20)
    lam_h *= (1.0 + elo_adj); lam_a *= (1.0 - elo_adj)
    
    # YENİ: Sıralama avantajı (FIFA/UEFA/Lig)
    ranking_advantage, ranking_source = calculate_ranking_advantage(
        fx["home"], fx["away"], area, fx.get("competition", "")
    )
    lam_h *= (1.0 + ranking_advantage); lam_a *= (1.0 - ranking_advantage)
    
    # Kadro değeri avantajı
    value_advantage, value_source = calculate_value_advantage(fx["home"], fx["away"], area)
    lam_h *= (1.0 + value_advantage); lam_a *= (1.0 - value_advantage)
    
    # Opp-adjusted form
    form_bits = []
    home_adj = away_adj = 0.0
    if fx.get("home_id"):
        home_adj, t = _form_adjust_from_matches(fx["home_id"], area, fx["home"])
        form_bits.append(t)
    if fx.get("away_id"):
        away_adj, t = _form_adjust_from_matches(fx["away_id"], area, fx["away"])
        form_bits.append(t)
    net_form = clamp(home_adj - away_adj, -0.18, 0.18)
    lam_h *= (1.0 + net_form); lam_a *= (1.0 - net_form)
    form_txt = (" | " + " / ".join(form_bits)) if form_bits else ""
    
    # TableAdj
    table_adj, table_txt = (0.0, "")
    if fx.get("competition_id") and fx.get("home_id") and fx.get("away_id"):
        h_rankpct, h_pos, N = _table_strength(fx["competition_id"], fx["home_id"])
        a_rankpct, a_pos, _ = _table_strength(fx["competition_id"], fx["away_id"])
        if (h_rankpct is not None) and (a_rankpct is not None):
            diff = h_rankpct - a_rankpct
            table_adj = clamp(diff * TABLE_WEIGHT, -TABLE_WEIGHT, TABLE_WEIGHT)
            table_txt = f" | Table {h_pos}/{N} vs {a_pos}/{N} (Adj {int(table_adj*100)}%)"
    lam_h *= (1.0 + table_adj); lam_a *= (1.0 - table_adj)
    
    # Streak
    net_streak, streak_txt = _streak_from_any(fx)
    lam_h *= (1.0 + net_streak); lam_a *= (1.0 - net_streak)
    
    # Model 1X2
    m_home, m_draw, m_away = poisson_prob(lam_h, lam_a)
    model_probs = (m_home, m_draw, m_away)
    
    # Market karışımı
    market_probs = None
    odds_txt = ""
    if odds_info:
        o1,oX,o2 = odds_info["odds"]
        p1,px,p2 = odds_info["probs"]
        market_probs = (p1,px,p2)
        odds_txt = f" | Odds(avg) 1/X/2: {o1:.2f}/{oX:.2f}/{o2:.2f}"
    
    p_home, p_draw, p_away = blend_model_market(model_probs, market_probs)
    blended_probs = (p_home, p_draw, p_away)
    
    picks = [("1", p_home), ("X", p_draw), ("2", p_away)]
    picks.sort(key=lambda x: x[1], reverse=True)
    pick, conf = picks[0]; conf_pct = int(round(conf*100))
    
    # Kart/Korner — Çoklu Kaynak Fallback
    apihint, kk_source = get_cards_corners_advanced(fx.get("area"), fx.get("competition"), fx.get("home"), fx.get("away"))
    
    kk = model_cards_corners(area, lam_h, lam_a, wx, apifoot_hint=apihint, source_info=kk_source)
    
    # YENİ: Sıralama bilgisi
    ranking_txt = f" | Sıralama: {ranking_source}" if ranking_source != "NO_RANKING_DATA" else ""
    
    # KAYNAK ETİKETLİ ÇIKTI
    kk_txt = (f" | Korner μ≈{kk['mu_corners']:.1f} (Üst8.5 {int(kk['p_over_corners_8_5']*100)}% / "
              f"Üst9.5 {int(kk['p_over_corners_9_5']*100)}%) [{kk['source']}]"
              f" | Kart μ≈{kk['mu_cards']:.1f} (Üst3.5 {int(kk['p_over_cards_3_5']*100)}%) [{kk['source']}]")
    
    # Kadro değeri bilgisi
    home_value, home_source = get_team_value(fx["home"], area)
    away_value, away_source = get_team_value(fx["away"], area)
    value_txt = f" | Kadro: {home_value:.0f}M€ [{home_source}] vs {away_value:.0f}M€ [{away_source}]"
    
    wx_txt = f" | {wx}" if wx else ""
    note = (f"Seçim: {pick} | Güven: {conf_pct}% | λ_h/λ_a: {lam_h:.2f}/{lam_a:.2f}"
            f"{wx_txt}{odds_txt}{kk_txt}{value_txt}{ranking_txt}{form_txt}{table_txt}{streak_txt}")
    
    return {
        "pick": pick,
        "confidence": conf_pct,
        "lambda_h": lam_h,
        "lambda_a": lam_a,
        "note": note,
        "probs_model": model_probs,
        "probs_market": market_probs,
        "probs_blend": blended_probs,
        "wx_adj": wx_adj,
        "elo_adj": elo_adj,
        "net_form": net_form,
        "home_advantage": home_advantage,
        "value_advantage": value_advantage,
        "value_source": value_source,
        "kk_source": kk_source,
        "ranking_advantage": ranking_advantage,
        "ranking_source": ranking_source
    }

def record_prediction(fx, rated, model_probs, market_probs, blended_probs, wx_adj, elo_adj, net_form):
    mk = match_key_from_fixture(fx)
    rec = {
        "ts": datetime.utcnow().isoformat() + "Z",
        "area": fx.get("area","Europe"),
        "competition": fx.get("competition",""),
        "home": fx.get("home"),
        "away": fx.get("away"),
        "utc_kickoff": (fx.get("utc_kickoff") or datetime.now(timezone.utc)).isoformat(),
        "probs_model": model_probs,
        "probs_market": market_probs,
        "probs_blend": blended_probs,
        "pick": rated["pick"],
        "conf_pct": rated["confidence"],
        "lam_h": rated["lambda_h"],
        "lam_a": rated["lambda_a"],
        "wx_adj": wx_adj,
        "elo_adj": elo_adj,
        "net_form": net_form,
        "w_mkt_used": get_w_mkt(),
        "home_advantage": rated.get("home_advantage", ELO_HOME_ADV),
        "value_advantage": rated.get("value_advantage", 0.0),
        "value_source": rated.get("value_source", "NONE"),
        "kk_source": rated.get("kk_source", "NONE"),
        "ranking_advantage": rated.get("ranking_advantage", 0.0),
        "ranking_source": rated.get("ranking_source", "NONE")
    }
    STATE["pred_store"][mk] = rec
    
    # İkincil anahtar: isim+tarih
    altk = alt_key_from_names(fx.get("home"), fx.get("away"), (fx.get("utc_kickoff") or datetime.now(timezone.utc)).astimezone(TR_TZ).strftime("%Y-%m-%d"))
    STATE["pred_store"][altk] = rec

# --- GÜNCELLENMİŞ FİKSTÜR FONKSİYONLARI --------------------------------------
def fetch_fd_fixtures(date_str):
    """Football-Data fikstürleri - U21 filtresi eklendi"""
    if not FD_TOKEN:
        return []
    url = "https://api.football-data.org/v4/matches"
    headers = {"X-Auth-Token": FD_TOKEN, **HEADERS_JSON}
    tr_day = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=TR_TZ)
    from_utc = (tr_day - timedelta(days=1)).astimezone(timezone.utc).strftime("%Y-%m-%d")
    to_utc = (tr_day + timedelta(days=1)).astimezone(timezone.utc).strftime("%Y-%m-%d")
    data = http_get(url, headers=headers, params={"dateFrom": from_utc, "dateTo": to_utc})
    out = []
    if data and data.get("matches"):
        for m in data["matches"]:
            dtp = to_dt_utc(m.get("utcDate"))
            if not dtp:
                continue
            if dtp.astimezone(TR_TZ).strftime("%Y-%m-%d") != date_str:
                continue
            if m.get("status") not in ("SCHEDULED", "TIMED"):
                continue
            comp = m.get("competition", {}) or {}
            area = (comp.get("area", {}) or {}).get("name", "")
            cname = (comp.get("name") or "")
            
            # U21 filtresi eklendi
            if not is_allowed_competition(area, cname):
                continue
                
            ht = m.get("homeTeam", {}) or {}
            at = m.get("awayTeam", {}) or {}
            out.append({
                "source": "FD",
                "utc_kickoff": dtp,
                "home": ht.get("name"),
                "away": at.get("name"),
                "home_id": ht.get("id"),
                "away_id": at.get("id"),
                "area": area,
                "competition": cname,
                "competition_id": comp.get("id"),
                "id": m.get("id"),
            })
    log(f"FD fixtures (TR={date_str}) -> {len(out)} (U21 filtrelendi)")
    return out

def fetch_apifoot_fixtures(date_str):
    """API-Football fikstürleri - U21 filtresi eklendi"""
    if not APIFOOT:
        return []
    headers = {"x-apisports-key": APIFOOT}
    url = "https://v3.football.api-sports.io/fixtures"
    data = http_get(url, headers=headers, params={"date": date_str})
    out = []
    if not data:
        log(f"APIF fixtures (TR={date_str}) -> 0 (boş/erişilemedi)")
        return out
    try:
        for item in (data.get("response") or []):
            status = (((item.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
            if status not in ("NS", "TBD", "PST", "SUSP", "1H", "HT", "2H"):
                continue
            dtp = to_dt_utc((item.get("fixture") or {}).get("date"))
            if not dtp:
                continue
            if dtp.astimezone(TR_TZ).strftime("%Y-%m-%d") != date_str:
                continue
            lg = (item.get("league") or {})
            area = (lg.get("country") or "Europe")
            cname = (lg.get("name") or "")
            
            # U21 filtresi eklendi
            if not is_allowed_competition(area, cname):
                continue
                
            tms = (item.get("teams") or {})
            home = (tms.get("home") or {})
            away = (tms.get("away") or {})
            out.append({
                "source": "APIF",
                "utc_kickoff": dtp,
                "home": home.get("name"),
                "away": away.get("name"),
                "home_id": home.get("id"),
                "away_id": away.get("id"),
                "area": area,
                "competition": cname,
                "competition_id": None,
                "id": f"apif:{item.get('fixture',{}).get('id')}",
            })
    except Exception as e:
        log(f"APIF fixtures parse err: {e}")
    log(f"APIF fixtures (TR={date_str}) -> {len(out)} (U21 filtrelendi)")
    return out

# --- GÜNCELLENMİŞ SONUÇ FONKSİYONLARI ----------------------------------------
def fetch_results_fd(date_str):
    """Football-Data sonuçları - U21 filtresi eklendi"""
    if not FD_TOKEN:
        return []
    url = "https://api.football-data.org/v4/matches"
    headers = {"X-Auth-Token": FD_TOKEN, **HEADERS_JSON}
    data = http_get(url, headers=headers, params={"dateFrom": date_str, "dateTo": date_str})
    out = []
    if not data or not data.get("matches"):
        return out
    for m in data["matches"]:
        comp = m.get("competition", {}) or {}
        area = (comp.get("area", {}) or {}).get("name", "")
        cname = comp.get("name", "")
        
        # U21 filtresi eklendi
        if not is_allowed_competition(area, cname):
            continue
            
        score_ft = ((m.get("score") or {}).get("fullTime") or {})
        gh, ga = score_ft.get("home"), score_ft.get("away")
        if gh is None or ga is None:
            continue
        out.append({
            "area": area,
            "competition": cname,
            "home": (m.get("homeTeam") or {}).get("name"),
            "away": (m.get("awayTeam") or {}).get("name"),
            "score_h": int(gh),
            "score_a": int(ga),
            "id_key": f"FD:{m.get('id')}",
            "date": date_str
        })
    log(f"FD results {date_str} -> {len(out)} (U21 filtrelendi)")
    return out

def fetch_results_apifoot(date_str):
    """API-Football sonuçları - U21 filtresi eklendi"""
    if not APIFOOT:
        return []
    headers = {"x-apisports-key": APIFOOT}
    url = "https://v3.football.api-sports.io/fixtures"
    data = http_get(url, headers=headers, params={"date": date_str})
    out = []
    if not data:
        return out
    for item in (data.get("response") or []):
        lg = (item.get("league") or {})
        area = (lg.get("country") or "Europe")
        cname = (lg.get("name") or "")
        
        # U21 filtresi eklendi
        if not is_allowed_competition(area, cname):
            continue
            
        st = (((item.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
        if st not in ("FT","AET","PEN","MATCH_FINISHED"):
            continue
        goals = (item.get("goals") or {})
        gh, ga = goals.get("home"), goals.get("away")
        if gh is None or ga is None:
            sc = (item.get("score") or {}).get("fulltime") or {}
            gh, ga = sc.get("home"), sc.get("away")
            if gh is None or ga is None:
                continue
        teams = (item.get("teams") or {})
        out.append({
            "area": area,
            "competition": cname,
            "home": (teams.get("home") or {}).get("name"),
            "away": (teams.get("away") or {}).get("name"),
            "score_h": int(gh),
            "score_a": int(ga),
            "id_key": f"APIF:apif:{(item.get('fixture') or {}).get('id')}",
            "date": date_str
        })
    log(f"APIF results {date_str} -> {len(out)} (U21 filtrelendi)")
    return out

# --- KODUN DEVAMI - ORİJİNAL FONKSİYONLAR DEĞİŞMEDEN KORUNDU ---

# --- Odds sport-key haritalaması --------------------------------------------
def guess_sport_key(area: str, comp: str):
    a = (area or "").lower()
    c = (comp or "").lower()
    s = f"{a} {c}"
    mapping = {
        # Germany
        "bundesliga 3": "soccer_germany_bundesliga3",
        "bundesliga 2": "soccer_germany_bundesliga2",
        "bundesliga": "soccer_germany_bundesliga",
        "dfb": "soccer_germany_dfb_pokal",
        # Turkey
        "super lig": "soccer_turkey_super_league",
        # England
        "premier": "soccer_epl",
        "championship": "soccer_efl_championship",
        # Spain
        "la liga": "soccer_spain_la_liga",
        # Italy
        "serie a": "soccer_italy_serie_a",
        # France
        "ligue 1": "soccer_france_ligue_one",
        # UEFA
        "champions": "soccer_uefa_champs_league",
        "europa": "soccer_uefa_europa_league",
        # Brazil (gerekebilir)
        "campeonato brasileiro": "soccer_brazil_campeonato",
        "brasileirao": "soccer_brazil_campeonato",
        "brasileiro": "soccer_brazil_campeonato",
    }
    for k, v in mapping.items():
        if k in s:
            return v
    return None

# --- Akıllı Hava Modu --------------------------------------------------------
WEATHER_SMART = (os.getenv("WEATHER_SMART", "1") == "1")
WEATHER_AREAS = set(
    s.strip().lower() for s in (os.getenv("WEATHER_AREAS", "England,Spain,Italy,Germany,France,Turkey,Netherlands,Portugal,Belgium").split(",")) if s.strip()
)

def weather_enabled(area_name: str, comp_name: str) -> bool:
    """Hava sadece büyük lig ve UEFA/FIFA maçlarında alınsın (performans için)."""
    a = _n(area_name)
    c = _n(comp_name)
    if a in {s.lower() for s in WEATHER_AREAS}:
        return True
    if any(p in c for p in _UEFA_ALLOW_PAT) or any(p in c for p in _FIFA_ALLOW_PAT):
        return True
    return False

# --- Hava: takım -> şehir eşleşmesi ------------------------------------------
def guess_city_from_team(team_name: str):
    t = (team_name or "").lower()
    overrides = {
        # Türkiye
        "galatasaray": "Istanbul",
        "fenerbahce": "Istanbul",
        "beşiktaş": "Istanbul",
        "besiktas": "Istanbul",
        "basaksehir": "Istanbul",
        "trabzonspor": "Trabzon",
        # Almanya
        "bayern": "Munich",
        "dortmund": "Dortmund",
        "leipzig": "Leipzig",
        "leverkusen": "Leverkusen",
        "schalke": "Gelsenkirchen",
        "st. pauli": "Hamburg",
        # İspanya
        "real madrid": "Madrid",
        "barcelona": "Barcelona",
        "atlético": "Madrid",
        "atletico": "Madrid",
        # İtalya
        "juventus": "Turin",
        "inter": "Milan",
        "milan": "Milan",
        "roma": "Rome",
        "lazio": "Rome",
        "napoli": "Naples",
        # Fransa
        "psg": "Paris",
        "paris saint-germain": "Paris",
        "lyon": "Lyon",
        "marseille": "Marseille",
        # Brezilya
        "corinthians": "Sao Paulo",
        "palmeiras": "Sao Paulo",
        "santos": "Santos",
        "flamengo": "Rio de Janeiro",
        "fluminense": "Rio de Janeiro",
        "botafogo": "Rio de Janeiro",
        "gremio": "Porto Alegre",
        "grêmio": "Porto Alegre",
        "internacional": "Porto Alegre",
        "atletico mineiro": "Belo Horizonte",
        # Milli takımlar
        "finland": "Helsinki",
        "lithuania": "Vilnius", 
        "scotland": "Glasgow",
        "greece": "Athens",
        "austria": "Vienna",
        "bosnia": "Sarajevo",
        "denmark": "Copenhagen",
        "russia": "Moscow",
        "netherlands": "Amsterdam",
        "czech": "Prague",
        "iceland": "Reykjavik"
    }
    for k, city in overrides.items():
        if k in t:
            return city
    return team_name  # fallback

def fetch_weather_note(home_team):
    city = guess_city_from_team(home_team)
    geo = http_get("https://geocoding-api.open-meteo.com/v1/search", params={"name": city, "count": 1, "language": "en"})
    if not geo or not geo.get("results"):
        return None
    lat = geo["results"][0]["latitude"]; lon = geo["results"][0]["longitude"]
    wx = http_get("https://api.open-meteo.com/v1/forecast", params={
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,precipitation,wind_speed_10m",
        "timezone": "auto"
    })
    if not wx:
        return None
    try:
        temps = wx["hourly"]["temperature_2m"][:6]
        prec = wx["hourly"]["precipitation"][:6]
        wind = wx["hourly"]["wind_speed_10m"][:6]
        tavg = sum(temps)/len(temps); pavg = sum(prec)/len(prec); wavg = sum(wind)/len(wind)
        return f"Hava: {tavg:.0}°C, yağış {pavg:.1f}mm, rüzgâr {wavg:.0f} km/s"
    except Exception:
        return None

def parse_weather(wx_text):
    if not wx_text:
        return (None, None)
    wind = None; precip = None
    try:
        if "rüzgâr" in wx_text:
            wind = safe_float(wx_text.split("rüzgâr")[1].split("km/s")[0].strip().split()[-1], None)
        if "yağış" in wx_text:
            precip = safe_float(wx_text.split("yağış")[1].split("mm")[0].strip().split()[-1], None)
    except Exception:
        pass
    return (wind, precip)

# --- Football-Data (fixtures) ------------------------------------------------
# [fetch_fd_fixtures fonksiyonu YUKARIDA GÜNCELLENDİ]

# --- 2. Fallback: API-Football (tarih bazlı) ---------------------------------
# [fetch_apifoot_fixtures fonksiyonu YUKARIDA GÜNCELLENDİ]

# --- OpenLigaDB fallback (yalnız Almanya alt ligleri) ------------------------
def fetch_openligadb_day(date_str):
    y, m, d = date_str.split("-")
    leagues = [(lg.strip(), int(y)) for lg in OLD_LEAGUES if lg.strip()]
    out = []
    for lg, season in leagues:
        url = f"https://www.openligadb.de/api/getmatchdata/{lg}/{season}"
        data = http_get(url)
        if not data:
            continue
        for mm in data:
            dt = mm.get("MatchDateTimeUTC") or mm.get("MatchDateTime")
            dtp = to_dt_utc(dt) if dt else None
            if not dtp:
                continue
            if dtp.astimezone(TR_TZ).strftime("%Y-%m-%d") != date_str:
                continue
            comp = mm.get("LeagueName") or lg
            if not is_allowed_competition("Germany", comp):
                continue
            out.append({
                "source":"OLD",
                "utc_kickoff": dtp,
                "home": mm.get("Team1",{}).get("TeamName"),
                "away": mm.get("Team2",{}).get("TeamName"),
                "home_id": None,
                "away_id": None,
                "area": "Germany",
                "competition": comp,
                "competition_id": None,
                "id": f"old:{mm.get('MatchID')}",
            })
    log(f"OpenLigaDB fixtures (TR={date_str}) -> {len(out)}")
    return out

# --- 4. Fallback: The Odds API (cache) ---------------------------------------
_odds_cache = {}  # {skey: {"ts": epoch, "data": list}}

def _fetch_odds_sport_cached(skey):
    now = time.time()
    ent = _odds_cache.get(skey)
    if ent and (now - ent["ts"] < ODDS_TTL_MIN*60):
        return ent["data"]
    url = f"https://api.the-odds-api.com/v4/sports/{skey}/odds/"
    params = {"regions":"eu,uk","markets":"h2h","oddsFormat":"decimal","apiKey":ODDS_KEY}
    data = http_get(url, params=params) if ODDS_KEY else None
    _odds_cache[skey] = {"ts": now, "data": data or []}
    return _odds_cache[skey]["data"]

def fetch_odds_fixtures(date_str):
    if not ODDS_KEY:
        return []
    keys = [
        ("soccer_epl", "England", "Premier League"),
        ("soccer_efl_championship", "England", "Championship"),
        ("soccer_spain_la_liga", "Spain", "La Liga"),
        ("soccer_italy_serie_a", "Italy", "Serie A"),
        ("soccer_france_ligue_one", "France", "Ligue 1"),
        ("soccer_germany_bundesliga", "Germany", "Bundesliga"),
        ("soccer_turkey_super_league", "Turkey", "Super Lig"),
        ("soccer_uefa_champs_league", "Europe", "UEFA Champions League"),
        ("soccer_uefa_europa_league", "Europe", "UEFA Europa League"),
    ]
    out = []
    for skey, area, comp in keys:
        data = _fetch_odds_sport_cached(skey)
        if not data or not isinstance(data, list):
            continue
        for ev in data:
            dtp = to_dt_utc(ev.get("commence_time"))
            if not dtp:
                continue
            if dtp.astimezone(TR_TZ).strftime("%Y-%m-%d") != date_str:
                continue
            out.append({
                "source": "ODDS",
                "utc_kickoff": dtp,
                "home": ev.get("home_team"),
                "away": ev.get("away_team"),
                "home_id": None,
                "away_id": None,
                "area": area,
                "competition": comp,
                "competition_id": None,
                "id": f"odds:{skey}:{ev.get('id','')}",
            })
    log(f"OddsAPI fixtures (TR={date_str}) -> {len(out)}")
    return out

# --- Fikstür toplayıcı zincir ------------------------------------------------
def fetch_fixtures(date_str):
    fixtures = fetch_fd_fixtures(date_str)
    if not fixtures:
        log("FD boş → API-Football fallback deneniyor…")
        fixtures = fetch_apifoot_fixtures(date_str)
    if not fixtures:
        log("API-Football da boş → OpenLigaDB fallback deneniyor…")
        fixtures = fetch_openligadb_day(date_str)
    if not fixtures:
        log("OpenLigaDB de boş → The Odds API event fallback deneniyor…")
        fixtures = fetch_odds_fixtures(date_str)
    return fixtures

# --- Odds (avg) --------------------------------------------------------------
def fetch_odds_avg(area, comp, home, away):
    if not ODDS_KEY:
        return None
    skey = guess_sport_key(area, comp)
    if not skey:
        return None
    data = _fetch_odds_sport_cached(skey)
    if not data or not isinstance(data, list):
        return None
    
    def norm(x):
        return (x or "").lower().replace(".", "").replace("-", " ").replace(" fc","").strip()
    
    h, a = norm(home), norm(away)
    best = None
    for ev in data:
        comps = ev.get("bookmakers", [])
        if not comps:
            continue
        t1 = norm(ev.get("home_team")); t2 = norm(ev.get("away_team"))
        # Geliştirilmiş takım eşleştirme - benzerlik kullan
        h_sim = team_similarity(h, t1)
        a_sim = team_similarity(a, t2)
        if h_sim >= 0.75 and a_sim >= 0.75:  # %75 benzerlik eşiği
            prices = {"home":[], "draw":[], "away":[]}
            for bk in comps:
                for mk in bk.get("markets", []):
                    for o in mk.get("outcomes", []):
                        nm = (o.get("name") or "").lower()
                        price = safe_float(o.get("price"), 0)
                        if nm in ("home","1"):
                            prices["home"].append(price)
                        elif nm in ("draw","x"):
                            prices["draw"].append(price)
                        elif nm in ("away","2"):
                            prices["away"].append(price)
            if prices["home"] and prices["draw"] and prices["away"]:
                best = (
                    sum(prices["home"])/len(prices["home"]),
                    sum(prices["draw"])/len(prices["draw"]),
                    sum(prices["away"])/len(prices["away"]),
                )
                break
    if not best:
        return None
    o1, ox, o2 = best
    inv = (1/o1) + (1/ox) + (1/o2)
    p1, px, p2 = (1/o1)/inv, (1/ox)/inv, (1/o2)/inv
    return {"odds": (o1, ox, o2), "probs": (p1, px, p2)}

# --- Poisson + Lig tabanı ----------------------------------------------------
LEAGUE_GOAL_BASE = {
    "Turkey": 2.60,
    "England": 2.75,
    "Spain": 2.50,
    "Italy": 2.70,
    "France": 2.55,
    "Germany": 3.05,
    "Brazil": 2.35,
    "Europe": 2.70,
}

def base_total_goals(area):
    base = 2.60
    for k, v in LEAGUE_GOAL_BASE.items():
        if k.lower() in (area or "").lower():
            base = v
            break
    scale = get_goal_scale(area)
    return clamp(base * scale, 1.5, 3.8)

def poisson_prob(lambda_h, lambda_a):
    def pois(m, lam):
        return (lam**m) * math.exp(-lam) / math.factorial(m)
    p_home = p_draw = p_away = 0.0
    for gh in range(0, 11):
        ph = pois(gh, lambda_h)
        for ga in range(0, 11):
            pa = pois(ga, lambda_a)
            if gh > ga:
                p_home += ph*pa
            elif gh == ga:
                p_draw += ph*pa
            else:
                p_away += ph*pa
    return p_home, p_draw, p_away

def poisson_over_prob(lam, line_int):
    lam = max(0.05, lam)
    start = int(line_int) + 1
    s = 0.0; m = start
    while m <= start + 40:
        s += (lam**m) * math.exp(-lam) / math.factorial(m)
        m += 1
    return clamp(s, 0.0, 1.0)

def blend_model_market(model_probs, market_probs):
    if not market_probs:
        return model_probs
    w = get_w_mkt()
    return tuple((1-w)*m + w*mk for m, mk in zip(model_probs, market_probs))

# --- Kart / Korner bazları ---------------------------------------------------
LEAGUE_CARD_BASE = {
    "Germany": 4.7,
    "Turkey": 5.1,
    "England": 4.1,
    "Spain": 5.0,
    "Italy": 4.8,
    "France": 4.3,
    "Brazil": 5.2,
    "Europe": 4.6
}

LEAGUE_CORNER_BASE = {
    "Germany": 9.4,
    "Turkey": 9.2,
    "England": 10.1,
    "Spain": 9.1,
    "Italy": 9.5,
    "France": 9.0,
    "Brazil": 8.7,
    "Europe": 9.2
}

def base_from_area(area, table, default):
    for k, v in table.items():
        if k.lower() in (area or "").lower():
            return v
    return default

# --- API-Football: opsiyonel istatistik ipucu --------------------------------
APIFOOT_BASE = "https://v3.football.api-sports.io"
_API_LEAGUE_MAP = {
    "England|Premier League": 39,
    "England|Championship": 40,
    "Spain|La Liga": 140,
    "Spain|La Liga 2": 141,
    "Italy|Serie A": 135,
    "Italy|Serie B": 136,
    "France|Ligue 1": 61,
    "France|Ligue 2": 62,
    "Germany|Bundesliga": 78,
    "Germany|2. Bundesliga": 79,
    "Turkey|Super Lig": 203,
    "Turkey|1. Lig": 204,
}

_apifoot_team_cache = {}  # search_name.lower() -> team_id
_apifoot_stat_cache = {}  # (league_id, season, team_id) -> stats_json

def _apifoot_get(path, params):
    if not APIFOOT:
        return None
    headers = {"x-apisports-key": APIFOOT}
    try:
        data = http_get(f"{APIFOOT_BASE}{path}", headers=headers, params=params)
        return (data or {}).get("response", None)
    except Exception as e:
        log(f"apifoot GET err: {e}")
        return None

def _apifoot_find_team_id(team_name):
    if not team_name:
        return None
    key = (team_name or "").strip().lower()
    ent = _apifoot_team_cache.get(key)
    if ent is not None:
        return ent
    resp = _apifoot_get("/teams", {"search": team_name})
    tid = None
    try:
        if resp:
            tid = ((resp[0] or {}).get("team") or {}).get("id")
    except Exception:
        tid = None
    _apifoot_team_cache[key] = tid
    return tid

def _apifoot_team_statistics(league_id, season, team_id):
    if not (league_id and season and team_id):
        return None
    cache_key = (league_id, season, team_id)
    if cache_key in _apifoot_stat_cache:
        return _apifoot_stat_cache[cache_key]
    resp = _apifoot_get("/teams/statistics", {
        "league": league_id,
        "season": season,
        "team": team_id
    })
    _apifoot_stat_cache[cache_key] = resp[0] if (resp and len(resp)>0) else None
    return _apifoot_stat_cache[cache_key]

def _apifoot_hint_cards_corners(area, comp, home, away):
    if not APIFOOT:
        return None
    lig_key = f"{(area or '').strip()}|{(comp or '').strip()}"
    lig_id = _API_LEAGUE_MAP.get(lig_key)
    if not lig_id:
        return None
    ssn = season_for_today()
    try:
        h_id = _apifoot_find_team_id(home)
        a_id = _apifoot_find_team_id(away)
        h_stat = _apifoot_team_statistics(lig_id, ssn, h_id) if h_id else None
        a_stat = _apifoot_team_statistics(lig_id, ssn, a_id) if a_id else None
        
        def cards_per_game(stat):
            if not stat:
                return None
            played = (((stat.get("fixtures") or {}).get("played") or {}).get("total")) or 0
            played = int(played) if played else 0
            if played <= 0:
                return None
            cards = (stat.get("cards") or {})
            total = 0.0
            for v in cards.values():
                t = v.get("total")
                if t is None:
                    continue
                t = safe_float(t, None)
                if t is not None:
                    total += t
            return total / played if total > 0 else None
        
        def corners_per_game(stat):
            if not stat:
                return None
            played = (((stat.get("fixtures") or {}).get("played") or {}).get("total")) or 0
            played = int(played) if played else 0
            if played <= 0:
                return None
            corners = (stat.get("corners") or {}).get("total")
            if corners is None:
                return None
            c = safe_float(corners, None)
            if c is None:
                return None
            return c / played
        
        h_cards = cards_per_game(h_stat)
        a_cards = cards_per_game(a_stat)
        h_corners = corners_per_game(h_stat)
        a_corners = corners_per_game(a_stat)
        
        out = {}
        if h_cards or a_cards:
            vals = [v for v in [h_cards, a_cards] if v]
            if vals:
                out["mu_cards_hint"] = sum(vals)/len(vals) * 2.0 * 0.5
        if h_corners or a_corners:
            vals = [v for v in [h_corners, a_corners] if v]
            if vals:
                out["mu_corners_hint"] = sum(vals)/len(vals) * 2.0 * 0.5
        return out if out else None
    except Exception as e:
        log(f"apifoot hint err: {e}")
        return None

# --- Standings/Streak (FD varsa FD, yoksa APIF) ------------------------------
_STANDINGS_CACHE = {}
_STREAK_CACHE = {}
_APIF_STANDINGS_CACHE = {}
_APIF_FIXTURES_CACHE = {}

def _fd_team_matches(team_id, days=120):
    if not (FD_TOKEN and team_id):
        return []
    headers = {"X-Auth-Token": FD_TOKEN, **HEADERS_JSON}
    to_dt = datetime.utcnow().date()
    from_dt = to_dt - timedelta(days=days)
    url = f"https://api.football-data.org/v4/teams/{team_id}/matches"
    data = http_get(url, headers=headers, params={
        "status": "FINISHED",
        "dateFrom": from_dt.isoformat(),
        "dateTo": to_dt.isoformat()
    })
    if not data or not data.get("matches"):
        return []
    ms = []
    for m in data["matches"]:
        dtp = to_dt_utc(m.get("utcDate"))
        if not dtp:
            continue
        ms.append((dtp, m))
    ms.sort(key=lambda x: x[0], reverse=True)
    return [m for _, m in ms]

def _form_adjust_from_matches(team_id, area, team_name):
    matches = _fd_team_matches(team_id, days=FORM_DAYS)
    if not matches:
        return (0.0, "")
    n = 0; score_sum = 0.0
    for m in matches:
        ht = (m.get("homeTeam", {}) or {}).get("name", "")
        at = (m.get("awayTeam", {}) or {}).get("name", "")
        score = (m.get("score", {}) or {}).get("fullTime", {}) or {}
        gh = score.get("home"); ga = score.get("away")
        if gh is None or ga is None:
            continue
        gh = int(gh); ga = int(ga)
        is_home = (ht == team_name)
        gf = gh if is_home else ga
        ga_ = ga if is_home else gh
        opp_name = at if is_home else ht
        opp_elo = elo_get(area, opp_name)
        w = 1.0 + clamp((opp_elo - 1500.0) / 1000.0, -0.2, 0.2)
        score_sum += (gf - ga_) * w
        n += 1
        if n >= FORM_LOOKBACK:
            break
    adj = 0.0 if n == 0 else clamp(math.tanh((score_sum / n) / 3.0) * 0.12, -0.15, 0.15)
    txt = f"FormAdj {('+' if adj>=0 else '')}{int(adj*100)}%"
    return (adj, txt)

def _fd_competition_standings(comp_id):
    if not (FD_TOKEN and comp_id):
        return None
    if comp_id in _STANDINGS_CACHE:
        return _STANDINGS_CACHE[comp_id]
    headers = {"X-Auth-Token": FD_TOKEN, **HEADERS_JSON}
    url = f"https://api.football-data.org/v4/competitions/{comp_id}/standings"
    data = http_get(url, headers=headers)
    by_id = {}; total_teams = None
    try:
        for st in (data.get("standings") or []):
            if (st.get("type") or "").upper() != "TOTAL":
                continue
            table = st.get("table") or []
            total_teams = len(table)
            for row in table:
                team = (row.get("team") or {})
                tid = team.get("id")
                pos = safe_float(row.get("position"), None)
                if tid is not None:
                    by_id[int(tid)] = {"position": int(pos) if pos else None}
        if by_id and total_teams:
            _STANDINGS_CACHE[comp_id] = {"total_teams": total_teams, "by_id": by_id}
            return _STANDINGS_CACHE[comp_id]
    except Exception as e:
        log(f"standings parse err: {e}")
    return None

def _table_strength(comp_id, team_id):
    st = _fd_competition_standings(comp_id)
    if not (st and team_id):
        return (None, None, None)
    row = (st["by_id"].get(int(team_id))) if isinstance(team_id, int) or str(team_id).isdigit() else None
    if not row or not row.get("position"):
        return (None, None, None)
    N = st["total_teams"] or 20
    pos = row["position"]
    rank_pct = (N - pos) / (N - 1) if N > 1 else 0.5
    return (rank_pct, pos, N)

def _apifoot_standings(league_id, season):
    if not APIFOOT or not league_id or not season:
        return None
    key = (league_id, season)
    if key in _APIF_STANDINGS_CACHE:
        return _APIF_STANDINGS_CACHE[key]
    resp = _apifoot_get("/standings", {"league": league_id, "season": season})
    by_id = {}; total = None
    try:
        lg = ((resp or [{}])[0] or {}).get("league", {}) if resp else {}
        groups = lg.get("standings") or []
        table = groups[0] if groups else []
        total = len(table)
        for row in table:
            tid = ((row.get("team") or {}).get("id"))
            rank = safe_float(row.get("rank", 0), 0)
            if tid is not None and rank > 0:
                by_id[int(tid)] = {"position": int(rank)}
        if by_id and total:
            _APIF_STANDINGS_CACHE[key] = {"total_teams": total, "by_id": by_id}
        else:
            _APIF_STANDINGS_CACHE[key] = None
    except Exception as e:
        log(f"apif standings parse err: {e}")
        _APIF_STANDINGS_CACHE[key] = None
    return _APIF_STANDINGS_CACHE[key]

def _apif_get_position_by_name(league_id, season, team_name):
    st = _apifoot_standings(league_id, season)
    if not (st and team_name):
        return (None, None)
    tid = _apifoot_find_team_id(team_name)
    if not tid:
        return (None, st.get("total_teams") if st else None)
    row = st["by_id"].get(int(tid)) if st and "by_id" in st else None
    if not row:
        return (None, st.get("total_teams") if st else None)
    return (row.get("position"), st.get("total_teams"))

def _fd_recent_streak(team_id, team_name):
    if not team_id:
        return (0.0, "")
    if team_id in _STREAK_CACHE:
        return _STREAK_CACHE[team_id]
    ms = _fd_team_matches(team_id, days=min(FORM_DAYS, 120))
    if not ms:
        _STREAK_CACHE[team_id] = (0.0, "")
        return (0.0, "")
    
    def outcome(m, team_name):
        score = (m.get("score", {}) or {}).get("fullTime", {}) or {}
        gh = score.get("home"); ga = score.get("away")
        if gh is None or ga is None:
            return None
        ht = (m.get("homeTeam", {}) or {}).get("name", "")
        is_home = (ht == team_name)
        if is_home:
            return "W" if gh>ga else "D" if gh==ga else "L"
        else:
            return "W" if ga>gh else "D" if ga==gh else "L"
    
    streak_char = None; count = 0
    for m in ms:
        res = outcome(m, team_name)
        if res is None:
            continue
        if streak_char is None:
            streak_char = res
            if res == "D":
                count = 0; break
            count = 1
        else:
            if res == streak_char:
                count += 1
            else:
                break
        if count >= 6:
            break
    
    if streak_char == "W":
        score = min(count, 5) * STREAK_UNIT; txt = f"W{count}"
    elif streak_char == "L":
        score = -min(count, 5) * STREAK_UNIT; txt = f"L{count}"
    else:
        score = 0.0; txt = "D0"
    score = clamp(score, -STREAK_MAX, STREAK_MAX)
    _STREAK_CACHE[team_id] = (score, txt)
    return (score, txt)

def _streak_from_any(fix):
    if fix.get("home_id") and fix.get("away_id"):
        hs, hs_txt = _fd_recent_streak(fix.get("home_id"), fix.get("home"))
        as_, as_txt = _fd_recent_streak(fix.get("away_id"), fix.get("away"))
        net = clamp(hs - as_, -STREAK_MAX, STREAK_MAX)
        if hs_txt or as_txt:
            return net, f" | Streak {hs_txt}/{as_txt} (Adj {int(net*100)}%)"
    
    lig_key = f"{(fix.get('area') or '').strip()}|{(fix.get('competition') or '').strip()}"
    lig_id = _API_LEAGUE_MAP.get(lig_key)
    if not lig_id:
        return 0.0, ""
    ssn = season_for_today()
    hs_pos, N = _apif_get_position_by_name(lig_id, ssn, fix.get("home"))
    as_pos, _ = _apif_get_position_by_name(lig_id, ssn, fix.get("away"))
    if hs_pos and as_pos and N:
        diffpct = ((N - hs_pos) - (N - as_pos)) / max(1, (N - 1))
        net = clamp(diffpct * (STREAK_UNIT*2.5), -STREAK_MAX, STREAK_MAX)
        return net, f" | Streak (pos proxy) {hs_pos}/{N} vs {as_pos}/{N} (Adj {int(net*100)}%)"
    return 0.0, ""

# --- GELİŞMİŞ DERECELENDİRME (Kaynak Etiketleme) -----------------------------
# [rate_fixture fonksiyonu YUKARIDA GÜNCELLENDİ]

# --- Tahmin/sonuç eşleşme & öğrenme yardımcıları -----------------------------
# [match_key_from_fixture, alt_key_from_names, find_prediction_for_result, 
# record_prediction fonksiyonları YUKARIDA GÜNCELLENDİ]

def brier_score(probs, outcome_idx):
    if probs is None:
        return None
    y = [0.0, 0.0, 0.0]; y[outcome_idx] = 1.0
    return sum((p - t)**2 for p, t in zip(probs, y)) / 3.0

# --- Mail --------------------------------------------------------------------
def send_mail(subject, body):
    body = (body or "").strip()
    if not body:
        body = "(Bu e-postada içerik üretilemedi / maç bulunamadı.)"
    msg = EmailMessage()
    msg["From"] = GMAIL_USER; msg["To"] = GMAIL_TO; msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(GMAIL_USER, GMAIL_PASS); s.send_message(msg)
    log(f"Mail gönderildi: {subject}")

# --- Sonuç çekiciler ---------------------------------------------------------
# [fetch_results_fd, fetch_results_apifoot, fetch_results fonksiyonları YUKARIDA GÜNCELLENDİ]

# --- Raporlar ----------------------------------------------------------------
def report_predictions(date_str):
    fixtures = fetch_fixtures(date_str)
    if not fixtures:
        send_mail(f"Günün Tahminleri | {date_str}", "Bugün için tahmin çıkarılacak maç bulunamadı.")
        return
    
    lines = [f"⚽ Günün Tahminleri — {date_str} (Transfermarkt + Milli Takım Elo + CIES/FootyStats + API-Football + TotalCorner + Kaynak Etiketleme + Ev Sahibi Dengeleme + Gelişmiş Eşleştirme + FIFA/UEFA Sıralama + U21 Filtresi)\n"]
    top = []; hi = []
    fixtures.sort(key=lambda x: x["utc_kickoff"] or datetime.now(timezone.utc))
    
    for fx in fixtures:
        odds = fetch_odds_avg(fx.get("area",""), fx.get("competition",""), fx["home"], fx["away"])
        rated = rate_fixture(fx, odds)
        record_prediction(
            fx, rated, rated["probs_model"], rated["probs_market"], rated["probs_blend"],
            rated["wx_adj"], rated["elo_adj"], rated["net_form"]
        )
        
        if rated["confidence"] < MIN_CONF:
            continue
            
        ko_local = (fx["utc_kickoff"] or datetime.now(timezone.utc)).astimezone(TR_TZ).strftime("%H:%M")
        line = f"- {ko_local} | {fx.get('area','')} {fx.get('competition','')} | {fx['home']} vs {fx['away']} — {rated['note']}"
        lines.append(line)
        
        bucket = hi if rated["confidence"] >= HIGH_ALERT else top
        bucket.append((rated["confidence"], line))
    
    if len(lines) == 1:
        lines.append("Filtreler nedeniyle listelenecek maç kalmadı (MIN_CONF yüksek olabilir).")
    
    top.sort(reverse=True)
    best = [f"\n🏆 En Güçlü {TOP_N} Seçim:"] + [" " + l.replace("- ","").strip() for c, l in top[:TOP_N]]
    
    hi_block = []
    if hi:
        hi.sort(reverse=True)
        hi_block.append("\n🔔 Yüksek Güven Seçimler:")
        for c, l in hi:
            hi_block.append(" " + l.replace("- ","").strip())
    
    body = "\n".join(lines + [""] + best + hi_block)
    _state_save(STATE)
    send_mail(f"Günün Tahminleri | {date_str}", body)

def report_results(date_str):
    results = fetch_results(date_str)
    lines = [f"📊 Günün Sonuçları — {date_str}", ""]
    
    if not results:
        lines.append("Bugün için sonuç bulunamadı.")
        send_mail(f"Günün Sonuçları | {date_str}", "\n".join(lines))
        return
    
    total = 0
    correct = 0
    brier_model_sum = 0.0
    brier_market_sum = 0.0
    brier_blend_sum = 0.0
    goal_stats = {}  # area -> (sum_goals, n)
    matched_with_fuzzy = 0
    
    for res in results:
        gh, ga = res["score_h"], res["score_a"]
        area = res["area"]
        outcome_idx = 0 if gh>ga else 1 if gh==ga else 2
        
        # Geliştirilmiş tahmin bulma
        pred = find_prediction_for_result(res)
        fuzzy_used = False
        
        if not pred:
            lines.append(f"❓ {res['home']} {gh}-{ga} {res['away']} (tahmin bulunamadı)")
            continue
        
        # Yakın eşleşme kullanıldıysa işaretle
        if pred.get("match_quality", "").startswith("yakin_eslesme"):
            fuzzy_used = True
            matched_with_fuzzy += 1
        
        total += 1
        pick = {"1":0,"X":1,"2":2}.get(pred["pick"],-1)
        ok = (pick == outcome_idx)
        if ok:
            correct += 1
        
        # Brier
        bm = brier_score(pred.get("probs_model"), outcome_idx) or 0.0
        bk = brier_score(pred.get("probs_market"), outcome_idx) or 0.0
        bb = brier_score(pred.get("probs_blend"), outcome_idx) or 0.0
        brier_model_sum += bm
        brier_market_sum += bk
        brier_blend_sum += bb
        
        # Elo öğrenme - dinamik ev avantajı ile
        result_hw = 1.0 if outcome_idx==0 else 0.0 if outcome_idx==2 else 0.5
        
        # Öncelikle kayıtlı home_advantage değerini kullan
        home_advantage = pred.get("home_advantage")
        if home_advantage is None:
            # Yedek: maç bilgileriyle yeniden hesapla
            home_advantage = home_adv_effective(
                area, res.get("competition", ""), res["home"], res["away"]
            )
        
        elo_update(area, res["home"], res["away"], result_hw, home_advantage)
        
        # goal_scale öğrenme
        goals = gh + ga
        cur_scale = get_goal_scale(area)
        expected_tot = base_total_goals(area)
        err = (goals - expected_tot) / max(1.0, expected_tot)
        new_scale = clamp(cur_scale * (1.0 + GOAL_LR * err), 0.7, 1.4)
        set_goal_scale(area, new_scale)
        
        s, n = goal_stats.get(area, (0,0))
        goal_stats[area] = (s+goals, n+1)
        
        mark = "✅" if ok else "❌"
        fuzzy_indicator = " 🔍" if fuzzy_used else ""
        match_quality = pred.get("match_quality", "bilinmiyor")
        lines.append(f"{mark}{fuzzy_indicator} {res['home']} {gh}-{ga} {res['away']} | Tahmin: {pred['pick']} ({pred['conf_pct']}%) | Eşleşme: {match_quality}")
    
    # w_mkt öğrenme (model vs market performansına göre)
    if total > 0:
        acc = 100.0 * correct / total
        bm_avg = brier_model_sum/total
        bk_avg = brier_market_sum/total if brier_market_sum>0 else None
        bb_avg = brier_blend_sum/total
        
        old_w = get_w_mkt()
        target = old_w
        if bk_avg is not None:
            if bk_avg + 1e-6 < bm_avg:
                target = clamp(old_w + LEARN_RATE*0.5, 0.0, 0.8)
            elif bm_avg + 1e-6 < bk_avg:
                target = clamp(old_w - LEARN_RATE*0.5, 0.0, 0.8)
            if bb_avg + 1e-6 < bm_avg:
                target = clamp(target + LEARN_RATE*0.2, 0.0, 0.8)
            elif bb_avg > bm_avg + 1e-6:
                target = clamp(target - LEARN_RATE*0.2, 0.0, 0.8)
        set_w_mkt(target)
        
        STATE["metrics"]["last_acc_pct"] = acc
        STATE["metrics"]["brier_model"] = bm_avg
        STATE["metrics"]["brier_market"] = bk_avg
        STATE["metrics"]["brier_blend"] = bb_avg
        
        lines.append("")
        lines.append(f"🎯 Doğruluk: {acc:.1f}% | Brier (model/market/blend): "
                    f"{bm_avg:.3f}/{(bk_avg if bk_avg is not None else float('nan')):.3f}/{bb_avg:.3f}")
        lines.append(f"⚖️ w_mkt: {old_w:.2f} → {get_w_mkt():.2f}")
        
        if matched_with_fuzzy > 0:
            lines.append(f"🔍 {matched_with_fuzzy} maç yakın eşleştirme ile bulundu")
    
    if goal_stats:
        lines.append("")
        lines.append("📈 Goal-scale güncellemeleri:")
        for area, (s, n) in goal_stats.items():
            lines.append(f" - {area}: avg_goals={s/max(1,n):.2f} | goal_scale={get_goal_scale(area):.3f}")
    
    _state_save(STATE)
    send_mail(f"Günün Sonuçları | {date_str}", "\n".join(lines))

# --- SERVICE (otomatik zamanlayıcı) ------------------------------------------
def _today_str_tr(dt=None):
    return (dt or datetime.now(TR_TZ)).strftime("%Y-%m-%d")

def _yesterday_str_tr(dt=None):
    dt = (dt or datetime.now(TR_TZ)) - timedelta(days=1)
    return dt.strftime("%Y-%m-%d")

def _time_reached_tr(target_h, target_m=0):
    now = datetime.now(TR_TZ)
    tgt = now.replace(hour=target_h, minute=target_m, second=0, microsecond=0)
    return now >= tgt

def run_service_loop():
    """Sürekli çalışır; TR 10:00'da bugünün tahmini, ertesi gün TR 04:00'da DÜNÜN sonuçlarını gönderir. Aynı gün içinde tekrarı engellemek için STATE içinde tarih izler."""
    log(f"SERVICE başlatıldı (TR hedefleri: {PREDICTION_HOUR:02d}:00 ve ertesi gün {RESULTS_HOUR:02d}:{RESULTS_MINUTE:02d} [düne ait])")
    while True:
        try:
            now_tr = datetime.now(TR_TZ)
            today = _today_str_tr(now_tr)
            
            # Tahmin: bugün 10:00 veya sonrası ve bugün henüz gönderilmemişse
            if (STATE.get("last_pred_date") != today) and _time_reached_tr(PREDICTION_HOUR, 0):
                log("SERVICE: Tahmin zamanı geldi → rapor hazırlanıyor…")
                report_predictions(today)
                STATE["last_pred_date"] = today
                _state_save(STATE)
            
            # Sonuç: ertesi gün 04:00'te, dünkü tarihe göre
            if (STATE.get("last_res_date") != today) and _time_reached_tr(RESULTS_HOUR, RESULTS_MINUTE):
                res_date = _yesterday_str_tr(now_tr)  # her zaman DÜN
                log(f"SERVICE: Sonuç zamanı geldi (dün={res_date}) → rapor hazırlanıyor…")
                report_results(res_date)
                STATE["last_res_date"] = today  # bugünü işaretle, tekrarı engelle
                _state_save(STATE)
                
        except Exception:
            tb = traceback.format_exc()
            log(tb)
            try:
                send_mail("Tahmin Botu | SERVICE Hata", tb)
            except Exception:
                pass
        
        # İnce adımlı uyku: 20 saniye
        time.sleep(20)

# --- Çalıştırıcı -------------------------------------------------------------
def main():
    try:
        now_utc = datetime.now(timezone.utc)
        tr_now = now_utc.astimezone(TR_TZ)
        date_str = tr_now.strftime("%Y-%m-%d")
        mode = MODE_ENV
        
        log(f"MODE={mode} | TR now={tr_now} | date={date_str} | w_mkt={get_w_mkt():.2f}")
        
        if mode == "SERVICE":
            run_service_loop()
            return
        
        if mode == "AUTO":
            mode = "PREDICT" if now_utc.hour == 7 else "RESULTS"
        
        if mode == "PREDICT":
            report_predictions(date_str)
        elif mode == "RESULTS":
            # Onaylı politika: RESULTS her zaman DÜN'e bakar
            report_results(_yesterday_str_tr(tr_now))
        else:
            send_mail("Tahmin Botu | Bilgi", "AUTO/SERVICE dışı çalıştırma. MODE=PREDICT veya MODE=RESULTS veya MODE=SERVICE bekleniyor.")
            
    except Exception:
        tb = traceback.format_exc(); log(tb)
        try:
            send_mail("Tahmin Botu | Hata", tb)
        except Exception:
            pass

if __name__ == "__main__":
    main()
    main()
