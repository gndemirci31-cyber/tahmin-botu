def _ensure_state_defaults(state: dict) -> dict:
    try:
        state.setdefault("elo", {})
        state.setdefault("goal_scale", {})
        state.setdefault("w_mkt", 0.45)
        state.setdefault("pred_store", {})
        state.setdefault("metrics", {})
        state.setdefault("last_saved", None)
        state.setdefault("last_pred_date", None)
        state.setdefault("last_res_date", None)
    except Exception:
        state = {"elo": {}, "goal_scale": {}, "w_mkt": 0.45, "pred_store": {}, "metrics": {},
                 "last_saved": None, "last_pred_date": None, "last_res_date": None}
    return state

# -*- coding: utf-8 -*-
"""
Tahmin Botu Ã¢â‚¬â€ GELÃ„Â°Ã…ÂMÃ„Â°Ã…Â FÃ„Â°NAL SÃƒÅ“RÃƒÅ“M (Transfermarkt + Milli TakÃ„Â±m Elo + CIES/FootyStats Fallback + API-Football Ãƒâ€“ncelikli + TotalCorner + FootyStats + Kaynak Etiketleme + EV SAHÃ„Â°BÃ„Â° DENGESÃ„Â° + YENÃ„Â° Ãƒâ€“ZELLÃ„Â°KLER)

GÃƒÅ“NCEL: Elo + OppAdj Form + Hava + Odds + API-Football ipucu + High-Alert ayrÃ„Â± mail + Otomatik Ãƒâ€“Ã„Å¸renme (w_mkt & goal_scale) + TableAdj (standings) + Streak (W/L) + LÃ„Â°G/KUPA FÃ„Â°LTRESÃ„Â° + AkÃ„Â±llÃ„Â± Hava + SERVICE modu + Transfermarkt + Milli TakÃ„Â±m Elo + Ãƒâ€¡oklu Fallback Sistemi + Kaynak Etiketleme + Ev Sahibi Dengeleme + YENÃ„Â° Ãƒâ€“ZELLÃ„Â°KLER

YENÃ„Â° Ãƒâ€“ZELLÃ„Â°KLER:
- Dixon-Coles Modeli
- Ãƒâ€¡ift Elo Sistemi (Attack/Defense)
- KapanÃ„Â±Ã…Å¸ OranÃ„Â± Drift
- Hakem/Dinlenme Etkisi
- SÃ„Â±ralama Decay
- Market Kalibrasyonu
- KÃ„Â±rmÃ„Â±zÃ„Â± Kart Riski
- HiyerarÃ…Å¸ik Gol Modeli
- xG-Proxy Modeli
- Kadro Etkisi
- Multi-market Konsistensi
- GeliÃ…Å¸miÃ…Å¸ Kalite Skoru

ÃƒÅ“cretsiz kaynaklar:
- football-data.org (Fixtures/sonuÃƒÂ§lar/standings) -> X-Auth-Token: FOOTBALL_DATA_TOKEN
- API-Football (opsiyonel, free tier) -> APIFOOTBALL_KEY varsa fikstÃƒÂ¼r + ipucu
- OpenLigaDB (fallback) -> anahtar gerekmez
- Open-Meteo (hava) -> anahtar gerekmez
- The Odds API (opsiyonel oranlar) -> ODDS_API_KEY varsa kullanÃ„Â±lÃ„Â±r
- Transfermarkt (tmapi) -> Kadro deÃ„Å¸erleri
- TotalCorner -> KÃƒÂ¶Ã…Å¸e korner verisi
- FootyStats -> Kart/korner lig ortalamalarÃ„Â±

Modlar:
- MODE=PREDICT -> Ãƒâ€¡alÃ„Â±Ã…Å¸tÃ„Â±Ã„Å¸Ã„Â± anda "GÃƒÂ¼nÃƒÂ¼n Tahminleri"
- MODE=RESULTS -> Ãƒâ€¡alÃ„Â±Ã…Å¸tÃ„Â±Ã„Å¸Ã„Â± anda **dÃƒÂ¼nÃƒÂ¼n** sonuÃƒÂ§larÃ„Â± (yeni saat planÃ„Â±na uygun)
- MODE=AUTO -> Ãƒâ€¡alÃ„Â±Ã…Å¸tÃ„Â±Ã„Å¸Ã„Â± anda saat UTC 07 ise PREDICT, deÃ„Å¸ilse RESULTS
- MODE=SERVICE -> SÃƒÂ¼rekli ÃƒÂ§alÃ„Â±Ã…Å¸Ã„Â±r; TR 10:00'da Tahmin, ertesi gÃƒÂ¼n TR 04:00'da DÃƒÅ“NÃƒÅ“N SonuÃƒÂ§larÃ„Â±nÃ„Â± yollar (tekrar etmez)
"""

import os, math, time, json, smtplib, traceback, re, urllib.parse, random, logging
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
import requests
from difflib import SequenceMatcher
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

# ==================== AYARLAR / SETTINGS ====================
STATE_PATH = os.getenv("STATE_PATH", "model_state.json")
SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "snapshots")

# Dosya yazma izinleri (1: aÃƒÂ§Ã„Â±k, 0: kapalÃ„Â±) / File write permissions (1: on, 0: off)
ALLOW_STATE_FILE = int(os.getenv("ALLOW_STATE_FILE", "1"))

# KadÃ„Â±n ve U-yaÃ…Å¸ maÃƒÂ§larÃ„Â±nÃ„Â± dahil etme bayraklarÃ„Â± / Women and U-age matches inclusion flags
ALLOW_WOMEN = int(os.getenv("ALLOW_WOMEN", "0"))
ALLOW_U21 = int(os.getenv("ALLOW_U21", "0"))

# Model sabitleri / Model constants
ELO_HOME_ADV = 40
W_MKT_INIT = 0.45
MIN_QUALITY = int(os.getenv("MIN_QUALITY", "0"))

# Yeni ÃƒÂ¶zellik bayraklarÃ„Â± / New feature flags
GOAL_MODEL = os.getenv("GOAL_MODEL", "POISSON")  # DC|POISSON|HIER
ELO_MODE = os.getenv("ELO_MODE", "single")  # split|single

# --- Ortak yardÃ„Â±mcÃ„Â±lar -------------------------------------------------------
TR_TZ = timezone(timedelta(hours=3))  # TÃƒÂ¼rkiye
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

# ==================== YARDIMCILAR / HELPERS ====================

def _ensure_dir(p: Path) -> bool:
    """KlasÃƒÂ¶r oluÃ…Å¸turur - gÃƒÂ¼venli versiyon / Create directory - safe version"""
    try:
        p.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        log(f"[FS] KlasÃƒÂ¶r oluÃ…Å¸turulamadÃ„Â± / Directory creation failed: {p} | {e}")
        return False

def save_snapshot(predictions: Dict, date: Optional[str] = None) -> None:
    """Tahmin snapshot'Ã„Â±nÃ„Â± kaydeder / Saves prediction snapshot"""
    date = date or datetime.now().strftime("%Y-%m-%d")
    snap_dir = Path(SNAPSHOT_DIR)
    if not _ensure_dir(snap_dir):
        log("[Snapshot] KayÃ„Â±t YAPILAMADI (klasÃƒÂ¶r aÃƒÂ§Ã„Â±lamadÃ„Â±) / Save FAILED (directory error)")
        return
    path = snap_dir / f"pred_{date}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        log(f"[Snapshot] Kaydedildi / Saved: {path}")
    except Exception as e:
        log(f"[Snapshot] Kaydetme hatasÃ„Â± / Save error: {e}")

def load_snapshot(date: Optional[str] = None) -> Dict:
    """Tahmin snapshot'Ã„Â±nÃ„Â± yÃƒÂ¼kler / Loads prediction snapshot"""
    date = date or datetime.now().strftime("%Y-%m-%d")
    path = Path(SNAPSHOT_DIR) / f"pred_{date}.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log(f"[Snapshot] YÃƒÂ¼klendi / Loaded: {path}")
        return data
    except Exception as e:
        log(f"[Snapshot] YÃƒÂ¼kleme hatasÃ„Â± / Load error: {e}")
        return {}

# ==================== FÃ„Â°LTRELER / FILTERS ====================

_WOMEN_TOKENS = {
    "women","woman","female","ladies","wsl", "fÃƒÂ©minine","feminine","feminino","femenina","femenino","femminile",
    "frauen","damas","dames","mulheres","kobiet","vrouwen","donna","dziewczÃ„â€¦t"
}

_WOMEN_HINTS = {
    "liga mx femenil", "liga f", "liga iberdrola", "primera femenina", "frauen-bundesliga", "frauen bundesliga",
    "serie a femminile", "division 1 fÃƒÂ©minine", "national women's",
}

_U_TOKENS = {
    "u23","u22","u21","u20","u19","u18","u17", "youth","junior","primavera","sub-23","sub-21","sub-20","sub20","sub21"
}

def _norm(s: str) -> str:
    return (s or "").strip().lower()

# Log sayaÃƒÂ§larÃ„Â± / Log counters
_filter_counters = {"women": 0, "u21": 0, "snapshot_used": 0}

def is_women_competition(area_name: str, comp_name: str) -> bool:
    """KadÃ„Â±n ligi/kupasÃ„Â± olup olmadÃ„Â±Ã„Å¸Ã„Â±nÃ„Â± kontrol eder"""
    if ALLOW_WOMEN == 1:
        return False
    
    a, c = (area_name or "").lower(), (comp_name or "").lower()
    combined = f"{a} {c}"
    
    # Token kontrolÃƒÂ¼
    if any(token in combined for token in _WOMEN_TOKENS):
        _filter_counters["women"] += 1
        return True
    
    # Hint kontrolÃƒÂ¼
    if any(hint in combined for hint in _WOMEN_HINTS):
        _filter_counters["women"] += 1
        return True
        
    return False

def is_u21_competition(comp_name: str) -> bool:
    """U-21 ligi olup olmadÃ„Â±Ã„Å¸Ã„Â±nÃ„Â± kontrol eder"""
    if ALLOW_U21 == 1:
        return False
        
    c = (comp_name or "").lower()
    if any(token in c for token in _U_TOKENS):
        _filter_counters["u21"] += 1
        return True
    return False

def get_filter_counts() -> Dict[str, int]:
    """Filtre sayaÃƒÂ§larÃ„Â±nÃ„Â± dÃƒÂ¶ndÃƒÂ¼rÃƒÂ¼r / Returns filter counters"""
    return _filter_counters.copy()

# --- TakÃ„Â±m AdÃ„Â± Benzerlik EÃ…Å¸leÃ…Å¸tirme ------------------------------------------
def normalize_team_name(name):
    """TakÃ„Â±m adÃ„Â±nÃ„Â± karÃ…Å¸Ã„Â±laÃ…Å¸tÃ„Â±rma iÃƒÂ§in normalize eder"""
    if not name:
        return ""
    
    # KÃƒÂ¼ÃƒÂ§ÃƒÂ¼k harfe ÃƒÂ§evir
    name = name.lower().strip()
    
    # YaygÃ„Â±n takÃ„Â±m eklerini kaldÃ„Â±r
    suffixes = [
        ' fc', ' cf', ' af', ' sf', ' if', ' ff', 
        ' football club', ' club de foot', ' athletic club',
        ' sports club', ' united', ' city', ' town', ' fc.',
        ' real', ' deportivo', ' athletic', ' atletico', ' atlÃƒÂ©tico',
        ' sporting', ' os ', ' as ', ' us ', ' ac ', ' inter ',
        ' borussia', ' dynamo', ' sparta', ' rapid', ' ajax'
    ]
    
    for suffix in suffixes:
        name = name.replace(suffix, '')
    
    # Ãƒâ€“zel karakterleri ve fazla boÃ…Å¸luklarÃ„Â± temizle
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Ãƒâ€“zel takÃ„Â±m ismi dÃƒÂ¼zeltmeleri
    special_cases = {
        'psg': 'paris saint germain',
        'psg paris': 'paris saint germain',
        'paris sg': 'paris saint germain',
        'om': 'olympique marseille',
        'olympique de marseille': 'olympique marseille',
        'olympique marseille': 'olympique marseille',
        'man united': 'manchester united',
        'man utd': 'manchester united',
        'man city': 'manchester city',
        'spurs': 'tottenham hotspur',
        'tottenham': 'tottenham hotspur',
        'newcastle': 'newcastle united',
        'west ham': 'west ham united',
        'leeds': 'leeds united',
        'leicester': 'leicester city',
        'wolves': 'wolverhampton wanderers',
        'wolverhampton': 'wolverhampton wanderers',
        'brighton': 'brighton and hove albion',
        'brighton hove': 'brighton and hove albion',
        'sheffield united': 'sheffield united',
        'sheffield wednesday': 'sheffield wednesday',
        'nottingham forest': 'nottingham forest',
        'norwich': 'norwich city',
        'derby': 'derby county',
        'qpr': 'queens park rangers',
        'mk dons': 'mk dons',
        'atalanta bc': 'atalanta',
        'atalanta bergamo': 'atalanta',
        'as roma': 'roma',
        'ac milan': 'milan',
        'inter milan': 'inter',
        'inter milano': 'inter',
        'fc bayern munich': 'bayern munich',
        'bayern munchen': 'bayern munich',
        'bayer leverkusen': 'leverkusen',
        'b mÃƒÂ¶nchengladbach': 'borussia monchengladbach',
        'borussia mgladbach': 'borussia monchengladbach',
        'borussia dortmund': 'dortmund',
        'eintracht frankfurt': 'eintracht frankfurt',
        'tsg hoffenheim': 'hoffenheim',
        'sc freiburg': 'freiburg',
        'vfl wolfsburg': 'wolfsburg',
        '1 fc koln': 'koln',
        '1 fc kÃƒÂ¶ln': 'koln',
        '1 fc cologne': 'koln',
        'fc koln': 'koln',
        'fc schalke 04': 'schalke',
        'schalke 04': 'schalke',
        'rcd espanyol': 'espanyol',
        'real betis': 'betis',
        'atletico madrid': 'atletico madrid',
        'atletico de madrid': 'atletico madrid',
        'athletic bilbao': 'athletic bilbao',
        'athletic club': 'athletic bilbao',
        'real sociedad': 'real sociedad',
        'real sociedad': 'real sociedad',
        'valencia cf': 'valencia',
        'villareal': 'villarreal',
        'cf villareal': 'villarreal',
        'olympique lyon': 'lyon',
        'olympique lyonnais': 'lyon',
        'as monaco': 'monaco',
        'as monaco fc': 'monaco',
        'losc lille': 'lille',
        'stade rennais': 'rennes',
        'stade de rennes': 'rennes',
        'ogc nice': 'nice',
        'fc nantes': 'nantes',
        'olympique marseille': 'marseille',
        'besiktas': 'besiktas',
        'besiktas jk': 'besiktas',
        'fenerbahce': 'fenerbahce',
        'fenerbahce sk': 'fenerbahce',
        'galatasaray': 'galatasaray',
        'galatasaray sk': 'galatasaray',
        'trabzonspor': 'trabzonspor',
        'trabzonspor sk': 'trabzonspor',
        'basaksehir': 'istanbul basaksehir',
        'istanbul basaksehir fk': 'istanbul basaksehir',
        'sivasspor': 'sivasspor',
        'giresunspor': 'giresunspor',
        'gaziantep fk': 'gaziantep',
        'gazisehir gaziantep': 'gaziantep',
        'hatayspor': 'hatayspor',
        'kayserispor': 'kayserispor',
        'konyaspor': 'konyaspor',
        'kasimpasa': 'kasimpasa',
        'alanyaspor': 'alanyaspor',
        'fatih karagumruk': 'karagumruk',
        'karagumruk sk': 'karagumruk',
        'goztepe': 'goztepe',
        'goztepe sk': 'goztepe',
        'ankaragucu': 'ankaragucu',
        'ankara guÃƒÂ§u': 'ankaragucu',
        'erzurumspor': 'erzurumspor',
        'denizlispor': 'denizlispor',
        'genclerbirligi': 'genclerbirligi',
        'gencler birligi': 'genclerbirligi',
        'kayseri': 'kayserispor',
        'antalyaspor': 'antalyaspor',
        'antalya spor': 'antalyaspor',
        # Milli takÃ„Â±m dÃƒÂ¼zeltmeleri
        'equatorial guinea': 'equatorial guinea',
        'estuarial guinea': 'equatorial guinea',
        'namaia': 'namibia',
        'bosna hareke': 'bosnia herzegovina',
        'farko asiatÃ„Â±n': 'faroe islands',
        'karadaÃ„Å¸': 'montenegro',
        'danimaria': 'denmark',
        'rusla': 'russia',
        'holanda': 'netherlands',
        'evl cumhuriyeti': 'czech republic',
        'himalistan': 'iceland',
    }
    
    return special_cases.get(name, name)

def team_similarity(a, b):
    """Ã„Â°ki takÃ„Â±m adÃ„Â± arasÃ„Â±ndaki benzerlik skorunu hesaplar (0-1 arasÃ„Â±)"""
    if not a or not b:
        return 0.0
    
    a_norm = normalize_team_name(a)
    b_norm = normalize_team_name(b)
    
    # Tam eÃ…Å¸leÃ…Å¸me
    if a_norm == b_norm:
        return 1.0
    
    # Kelime bazlÃ„Â± benzerlik
    a_words = set(a_norm.split())
    b_words = set(b_norm.split())
    
    if a_words and b_words:
        # Ortak kelime oranÃ„Â±
        common_words = a_words.intersection(b_words)
        word_similarity = len(common_words) / max(len(a_words), len(b_words))
        
        # String benzerlik
        string_similarity = SequenceMatcher(None, a_norm, b_norm).ratio()
        
        # Kombine skor (kelime benzerliÃ„Å¸i daha aÃ„Å¸Ã„Â±rlÃ„Â±klÃ„Â±)
        return 0.7 * word_similarity + 0.3 * string_similarity
    
    return SequenceMatcher(None, a_norm, b_norm).ratio()

def find_closest_team(target_team, team_list, threshold=0.75):
    """
    TakÃ„Â±m listesinde en benzer takÃ„Â±mÃ„Â± bulur
    
    Args:
        target_team: Aranan takÃ„Â±m adÃ„Â±
        team_list: Arama yapÃ„Â±lacak takÃ„Â±m listesi
        threshold: Minimum benzerlik eÃ…Å¸iÃ„Å¸i (0-1 arasÃ„Â±)
    
    Returns:
        (en_benzer_takÃ„Â±m, benzerlik_skoru) veya (None, 0) eÃ…Å¸leÃ…Å¸me yoksa
    """
    if not target_team or not team_list:
        return None, 0.0
    
    best_match = None
    best_score = 0.0
    
    for team in team_list:
        if not team:
            continue
            
        score = team_similarity(target_team, team)
        if score > best_score and score >= threshold:
            best_score = score
            best_match = team
    
    return best_match, best_score

# ==================== STATE YÃƒâ€“NETÃ„Â°MÃ„Â° / STATE MANAGEMENT ====================

def load_state() -> Dict:
    """STATE yoksa veya kapalÃ„Â±ysa snapshot'tan yÃƒÂ¼kler / Loads from snapshot if STATE missing or disabled"""
    if not ALLOW_STATE_FILE or not Path(STATE_PATH).exists():
        log("[STATE] Dosya yok/kapalÃ„Â±. Snapshot'tan okunacak / File missing/disabled. Loading from snapshot.")
        _filter_counters["snapshot_used"] += 1
        return load_snapshot()
    
    try:
        with open(STATE_PATH, "r", encoding="utf-8") as f:
            state = json.load(f)
        state.setdefault("elo", {})
        state.setdefault("goal_scale", {})
        state.setdefault("w_mkt", W_MKT_INIT)
        state.setdefault("pred_store", {})
        state.setdefault("metrics", {})
        state.setdefault("last_saved", None)
        state.setdefault("last_pred_date", None)
        state.setdefault("last_res_date", None)
        return state
    except Exception as e:
        log(f"[STATE] YÃƒÂ¼kleme hatasÃ„Â± / Load error: {e}. Snapshot'a dÃƒÂ¼Ã…Å¸ÃƒÂ¼lÃƒÂ¼yor / Falling back to snapshot.")
        _filter_counters["snapshot_used"] += 1
        return load_snapshot()

def save_state(state: Dict) -> None:
    """State'i kaydeder / Saves state"""
    if not ALLOW_STATE_FILE:
        log("[STATE] Yazma kapalÃ„Â± (ALLOW_STATE_FILE=0) / Write disabled")
        return
    
    try:
        state["last_saved"] = datetime.utcnow().isoformat() + "Z"
        state_dir = Path(STATE_PATH).parent
        if _ensure_dir(state_dir):
            with open(STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            log(f"[STATE] Kaydedildi Ã¢â€ â€™ / Saved Ã¢â€ â€™ {STATE_PATH}")
    except Exception as e:
        log(f"[STATE] Kaydetme hatasÃ„Â± / Save error: {e}")

STATE = load_state()

STATE = _ensure_state_defaults(STATE)
def _team_key(area, name):
    a = (area or "Europe").strip().lower()
    n = (name or "").strip().lower()
    return f"{a}:{n}"

def elo_get(area, name):
    """GeliÃ…Å¸tirilmiÃ…Å¸ Elo getirme - milli takÃ„Â±mlar iÃƒÂ§in proxy destekli"""
    # Ãƒâ€“nce milli takÃ„Â±m kontrolÃƒÂ¼
    national_elo, national_source = get_national_elo_proxy(name, area)
    if national_elo is not None:
        return national_elo
    
    # Normal kulÃƒÂ¼p takÃ„Â±mÃ„Â± Elo'su
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
    Elo gÃƒÂ¼ncellemesi - dinamik ev avantajÃ„Â± desteÃ„Å¸i
    
    Args:
        area: Lig bÃƒÂ¶lgesi
        home_name: Ev sahibi takÃ„Â±m
        away_name: Deplasman takÃ„Â±mÃ„Â±  
        result_hw: SonuÃƒÂ§ (1.0=ev kazandÃ„Â±, 0.5=berabere, 0.0=deplasman kazandÃ„Â±)
        home_advantage: Ev avantajÃ„Â± (None ise ELO_HOME_ADV kullanÃ„Â±r)
    """
    if home_advantage is None:
        home_advantage = ELO_HOME_ADV
        
    Eh = elo_get(area, home_name)
    Ea = elo_get(area, away_name)
    ph = elo_expect(Eh, Ea, home_advantage)
    pa = 1.0 - ph
    Eh_new = Eh + ELO_K * (result_hw - ph)
    Ea_new = Ea + ELO_K * ((1.0 - result_hw) - pa)
    elo_set(area, home_name, Eh_new)
    elo_set(area, away_name, Ea_new)

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

# --- GELÃ„Â°Ã…ÂMÃ„Â°Ã…Â KADRO DEÃ„ÂERÃ„Â° SÃ„Â°STEMÃ„Â° (Transfermarkt + Fallback'ler) ------------
TEAM_VALUES_PATH = "team_values.json"

def load_team_values():
    """TakÃ„Â±m deÃ„Å¸erlerini yÃƒÂ¼kler"""
    try:
        if os.path.exists(TEAM_VALUES_PATH):
            with open(TEAM_VALUES_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log(f"TakÃ„Â±m deÃ„Å¸erleri yÃƒÂ¼kleme hatasÃ„Â±: {e}")
    return {}

def save_team_values(values):
    """TakÃ„Â±m deÃ„Å¸erlerini kaydeder"""
    try:
        with open(TEAM_VALUES_PATH, "w", encoding="utf-8") as f:
            json.dump(values, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log(f"TakÃ„Â±m deÃ„Å¸erleri kaydetme hatasÃ„Â±: {e}")

def get_team_value_tmapi(team_name, area="Europe"):
    """Transfermarkt API'sinden kadro deÃ„Å¸erini getirir - GÃƒÅ“NCELLENDÃ„Â°"""
    if not team_name:
        return None, None
    
    try:
        # Ãƒâ€“nce takÃ„Â±m ismini normalize et
        normalized_name = normalize_team_name(team_name)
        
        # tmapi.vercel.app API'si - URL encode ekle
        encoded_name = urllib.parse.quote(normalized_name)
        url = f"https://tmapi.vercel.app/api/team/{encoded_name}"
        
        response = http_get(url, timeout=15)
        
        if response and response.get("success"):
            squad_value = response.get("data", {}).get("squad_value", None)
            if squad_value and squad_value > 0:
                log(f"Ã¢Å“â€¦ TMAPI baÃ…Å¸arÃ„Â±lÃ„Â±: {team_name} -> {squad_value}M Ã¢â€šÂ¬")
                return squad_value, "TMAPI"
            else:
                log(f"Ã¢Å¡ Ã¯Â¸Â TMAPI deÃ„Å¸er bulunamadÃ„Â±: {team_name}")
        else:
            log(f"Ã¢ÂÅ’ TMAPI hata: {team_name} - {response}")
            
    except Exception as e:
        log(f"Ã¢ÂÅ’ Transfermarkt API hatasÃ„Â± {team_name}: {e}")
    
    return None, None

def get_team_value_cies_fallback(team_name, area="Europe"):
    """CIES/FootyStats fallback - GELÃ„Â°Ã…ÂTÃ„Â°RÃ„Â°LMÃ„Â°Ã…Â"""
    # Milli takÃ„Â±mlar iÃƒÂ§in ÃƒÂ¶zel deÃ„Å¸erler
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
    
    # Ãƒâ€“nce milli takÃ„Â±m kontrolÃƒÂ¼
    for nat_team, value in national_teams.items():
        if nat_team in team_lower:
            return value, "CIES_NATIONAL"
    
    # Lig bazlÃ„Â± ortalama deÃ„Å¸erler
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
    
    return 30.0, "CIES_DEFAULT"  # Daha dÃƒÂ¼Ã…Å¸ÃƒÂ¼k genel varsayÃ„Â±lan

def get_team_value(team_name, area="Europe"):
    """GeliÃ…Å¸tirilmiÃ…Å¸ kadro deÃ„Å¸eri sistemi - zincirli fallback"""
    if not team_name:
        return 30.0, "DEFAULT"
    
    # Ãƒâ€“nce cache'ten kontrol et
    team_values = load_team_values()
    cache_key = f"{area}:{normalize_team_name(team_name)}"
    
    if cache_key in team_values:
        value_data = team_values[cache_key]
        value = value_data.get("value", 30.0)
        source = value_data.get("source", "CACHE")
        timestamp = value_data.get("timestamp", 0)
        
        # 30 gÃƒÂ¼nden eski veriyi yenile
        if time.time() - timestamp < 30 * 24 * 60 * 60:
            return value, source
    
    # Zincirli fallback sistemi
    value, source = get_team_value_tmapi(team_name, area)
    
    if value is None:
        # Fallback: CIES/FootyStats lig ortalamalarÃ„Â±
        value, source = get_team_value_cies_fallback(team_name, area)
    
    # Cache'e kaydet
    team_values[cache_key] = {
        "value": value,
        "source": source,
        "timestamp": time.time()
    }
    save_team_values(team_values)
    
    log(f"Kadro deÃ„Å¸eri gÃƒÂ¼ncellendi: {team_name} -> {value}M Ã¢â€šÂ¬ ({source})")
    return value, source

def calculate_value_advantage(home_team, away_team, area="Europe"):
    """Kadro deÃ„Å¸eri avantajÃ„Â±nÃ„Â± hesaplar (-1 ile +1 arasÃ„Â±)"""
    home_value, home_source = get_team_value(home_team, area)
    away_value, away_source = get_team_value(away_team, area)
    
    if home_value + away_value == 0:
        return 0.0, "NONE"
    
    # DeÃ„Å¸er farkÃ„Â±nÃ„Â±n normalize edilmiÃ…Å¸ avantaja dÃƒÂ¶nÃƒÂ¼Ã…Å¸ÃƒÂ¼mÃƒÂ¼
    value_ratio = (home_value - away_value) / (home_value + away_value)
    advantage = clamp(value_ratio * 0.3, -0.3, 0.3)  # Maksimum %30 etki
    
    # Kaynak bilgisi - hangi takÃ„Â±m hangi kaynaktan
    source_info = f"TM:{home_source}/{away_source}"
    
    return advantage, source_info

# --- MÃ„Â°LLÃ„Â° TAKIM ELO SÃ„Â°STEMÃ„Â° -------------------------------------------------
NATIONAL_TEAM_ELO_PATH = "national_elo.json"

def load_national_elo():
    """Milli takÃ„Â±m Elo deÃ„Å¸erlerini yÃƒÂ¼kler"""
    try:
        if os.path.exists(NATIONAL_TEAM_ELO_PATH):
            with open(NATIONAL_TEAM_ELO_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception as e:
        log(f"Milli takÃ„Â±m Elo yÃƒÂ¼kleme hatasÃ„Â±: {e}")
    return {}

def save_national_elo(values):
    """Milli takÃ„Â±m Elo deÃ„Å¸erlerini kaydeder"""
    try:
        with open(NATIONAL_TEAM_ELO_PATH, "w", encoding="utf-8") as f:
            json.dump(values, f, ensure_ascii=False, indent=2)
    except Exception as e:
        log(f"Milli takÃ„Â±m Elo kaydetme hatasÃ„Â±: {e}")

def get_national_elo_proxy(team_name, area="Europe"):
    """Milli takÃ„Â±mlar iÃƒÂ§in Elo proxy deÃ„Å¸eri - ACÃ„Â°L DÃƒÅ“ZELTME"""
    if not team_name:
        return None, None
    
    # Ãƒâ€“NEMLÃ„Â°: Ãƒâ€“nce kulÃƒÂ¼p takÃ„Â±mÃ„Â± kontrolÃƒÂ¼ - milli takÃ„Â±m deÃ„Å¸ilse None dÃƒÂ¶n
    national_indicators = ["national", "milli", "country", "olympics", "world cup", "euro", "qualification"]
    team_lower = team_name.lower()
    
    is_national = any(indicator in team_lower for indicator in national_indicators)
    if not is_national:
        return None, None  # KulÃƒÂ¼p takÃ„Â±mÃ„Â±
    
    # Milli takÃ„Â±m Elo deÃ„Å¸erleri (FIFA sÃ„Â±ralamasÃ„Â± bazlÃ„Â±)
    national_elo_values = {
        # Afrika
        "mozambique": 1300, "guinea": 1400, "botswana": 1250, "uganda": 1350,
        "malawi": 1280, "equatorial guinea": 1320, "liberia": 1270, "namibia": 1290,
        # Avrupa  
        "finland": 1450, "lithuania": 1300, "scotland": 1550, "greece": 1500,
        "austria": 1520, "san marino": 1000, "cyprus": 1350, "bosnia": 1480,
        "faroe islands": 1200, "montenegro": 1420, "belarus": 1380, "denmark": 1600,
        "russia": 1550, "netherlands": 1650, "czech republic": 1520, "iceland": 1450
    }
    
    team_normalized = normalize_team_name(team_name)
    
    for nat_team, elo in national_elo_values.items():
        if nat_team in team_normalized:
            return elo, "ELO_NATIONAL"
    
    return 1400.0, "ELO_DEFAULT"  # VarsayÃ„Â±lan milli takÃ„Â±m Elo'su

# ==================== YENÃ„Â° Ãƒâ€“ZELLÃ„Â°KLER / NEW FEATURES ====================

# 3. EÃ…Å¸leÃ…Å¸tirme ToleransÃ„Â± / Matching Tolerance
def match_with_tolerance(team1: str, team2: str, date1: str, date2: str, tolerance_days: int = 2) -> bool:
    """TakÃ„Â±m ve tarih eÃ…Å¸leÃ…Å¸tirme toleransÃ„Â± / Team and date matching with tolerance"""
    try:
        date_obj1 = datetime.strptime(date1, "%Y-%m-%d")
        date_obj2 = datetime.strptime(date2, "%Y-%m-%d")
        days_diff = abs((date_obj2 - date_obj1).days)
        return days_diff <= tolerance_days
    except:
        return False

# 4. KapanÃ„Â±Ã…Å¸ OranÃ„Â± Drift / Closing Line Drift
def calculate_closing_drift(opening_odds: float, closing_odds: float) -> Tuple[float, str]:
    """KapanÃ„Â±Ã…Å¸ oranÃ„Â± drift hesaplama / Closing line drift calculation"""
    if opening_odds == 0:
        return 0.0, "Drift: 0%"
    drift_pct = ((closing_odds - opening_odds) / opening_odds) * 100
    drift_pct = max(min(drift_pct, 3.0), -3.0)  # Ã‚Â±3% tavan / ceiling
    confidence_impact = drift_pct  # GÃƒÂ¼ven skoruna direkt etki / Direct impact to confidence score
    note = f"Drift: {drift_pct:+.1f}%"
    return confidence_impact, note

# 5. Hakem / Dinlenme Etkisi / Referee / Rest Effect
def calculate_referee_impact(referee_stats: Dict) -> float:
    """Hakem kart etkisi hesaplama / Referee card impact calculation"""
    avg_cards = referee_stats.get("avg_cards_per_match", 2.0)
    base_cards = 2.0  # Ortalama baz / Average base
    impact = (avg_cards - base_cards) / base_cards * 10  # Ã‚Â±10% tavan / ceiling
    impact = max(min(impact, 10.0), -10.0)
    return impact

def calculate_fatigue_impact(matches_last_10_days: int, days_since_last_match: int) -> float:
    """Dinlenme-gÃƒÂ¼n etkisi hesaplama / Rest-day impact calculation"""
    # MaÃƒÂ§ yoÃ„Å¸unluÃ„Å¸u etkisi / Match density impact
    density_impact = min(matches_last_10_days * 2, 6.0)  # Ã‚Â±6% tavan / ceiling
    # Dinlenme etkisi / Rest impact
    rest_impact = max((3 - days_since_last_match) * 2, -6.0)  # Ã‚Â±6% tavan
    total_impact = (density_impact + rest_impact) / 2
    return max(min(total_impact, 6.0), -6.0)

# 6. Dixon-Coles Modeli / Dixon-Coles Model
class DixonColesModel:
    """Dixon-Coles bivariate Poisson modeli / bivariate Poisson model"""
    def __init__(self):
        self.attack = {}
        self.defense = {}
        self.rho = 0.0  # Beraberlik korelasyonu / Draw correlation
    
    def predict(self, home_team: str, away_team: str) -> Tuple[float, float, float]:
        """MaÃƒÂ§ sonucu olasÃ„Â±lÃ„Â±klarÃ„Â± / Match outcome probabilities"""
        home_attack = self.attack.get(home_team, 1.0)
        home_defense = self.defense.get(home_team, 1.0)
        away_attack = self.attack.get(away_team, 1.0)
        away_defense = self.defense.get(away_team, 1.0)
        
        # Basit Poisson hesaplama / Simple Poisson calculation
        lambda_home = home_attack * away_defense
        lambda_away = away_attack * home_defense
        
        # Dixon-Coles dÃƒÂ¼zeltmesi / Dixon-Coles correction
        if GOAL_MODEL == "DC":
            # Beraberlik olasÃ„Â±lÃ„Â±Ã„Å¸Ã„Â± iyileÃ…Å¸tirmesi / Draw probability improvement
            draw_bias = 1.0 + self.rho
            lambda_home *= draw_bias
            lambda_away *= draw_bias
        
        return lambda_home, lambda_away, self.rho

# 7. Ãƒâ€¡ift Elo / Dual Elo
class DualEloSystem:
    """Attack/Defense ayrÃ„Â± Elo sistemi / Separate Attack/Defense Elo system"""
    def __init__(self, k_factor: int = 20, home_advantage: int = 40):
        self.k = k_factor
        self.home_adv = home_advantage
        self.attack_elo = {}
        self.defense_elo = {}
    
    def get_ratings(self, team: str) -> Tuple[float, float]:
        """TakÃ„Â±m rating'lerini dÃƒÂ¶ndÃƒÂ¼rÃƒÂ¼r / Returns team ratings"""
        return self.attack_elo.get(team, 1500.0), self.defense_elo.get(team, 1500.0)
    
    def update_ratings(self, home_team: str, away_team: str, home_goals: int, away_goals: int):
        """Rating'leri gÃƒÂ¼nceller / Updates ratings"""
        # Attack/defense ayrÃ„Â± gÃƒÂ¼ncelleme mantÃ„Â±Ã„Å¸Ã„Â± / Separate attack/defense update logic
        pass

# 8. SÃ„Â±ralama Decay / Ranking Decay
def apply_ranking_decay(rankings: Dict[str, float], months_ago: int) -> float:
    """SÃ„Â±ralama sinyali time-decay uygular / Applies time-decay to ranking signals"""
    half_life = 12  # 12 ay yarÃ„Â±-ÃƒÂ¶mÃƒÂ¼r / 12 month half-life
    decay_factor = 0.5 ** (months_ago / half_life)
    return rankings.get("fifa", 0.0) * decay_factor

# 9. Market Kalibrasyonu / Market Calibration
class MarketCalibrator:
    """Piyasa olasÃ„Â±lÃ„Â±k kalibrasyonu / Market probability calibration"""
    def __init__(self):
        self.isotonic_model = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = True
    
    def calibrate_probabilities(self, raw_probs: np.ndarray, actual_results: np.ndarray) -> np.ndarray:
        """OlasÃ„Â±lÃ„Â±klarÃ„Â± kalibre eder / Calibrates probabilities"""
        if len(raw_probs) < 10 or not self.is_fitted:
            return raw_probs  # Yeterli veri yoksa / Not enough data
        return self.isotonic_model.transform(raw_probs)

# 10. KÃ„Â±rmÃ„Â±zÃ„Â± Kart Riski / Red Card Risk
def calculate_red_card_risk(referee_red_rate: float, team_red_rate: float) -> float:
    """KÃ„Â±rmÃ„Â±zÃ„Â± kart risk skoru hesaplar / Calculates red card risk score"""
    base_risk = (referee_red_rate + team_red_rate) / 2
    risk_impact = max(min(base_risk * 20, 6.0), 3.0)  # %3-6 arasÃ„Â± etki / 3-6% impact
    return -risk_impact  # Negatif etki / Negative impact

# 11. Basit Bayes Modeli / Simple Bayes Model
class HierarchicalGoalModel:
    """HiyerarÃ…Å¸ik gol modeli / Hierarchical goal model"""
    def __init__(self):
        self.league_priors = {}
        self.team_offense = {}
        self.team_defense = {}
    
    def predict_goals(self, home_team: str, away_team: str, league: str) -> Tuple[float, float]:
        """Gol tahmini / Goal prediction"""
        # Lig bazlÃ„Â± prior + takÃ„Â±m regularizasyon / League-based prior + team regularization
        home_prior = self.league_priors.get(league, 1.0)
        away_prior = self.league_priors.get(league, 1.0)
        home_attack = self.team_offense.get(home_team, home_prior)
        away_attack = self.team_offense.get(away_team, away_prior)
        home_defense = self.team_defense.get(home_team, home_prior)
        away_defense = self.team_defense.get(away_team, away_prior)
        
        lambda_home = home_attack * away_defense
        lambda_away = away_attack * home_defense
        return lambda_home, lambda_away

# 12. xG-Proxy Modeli / xG-Proxy Model
def calculate_xg_proxy(shots: int, on_target: int, dangerous_attacks: int) -> float:
    """xG proxy deÃ„Å¸eri hesaplar / Calculates xG proxy value"""
    if shots == 0:
        return 0.0
    conversion_rate = on_target / shots
    danger_ratio = dangerous_attacks / max(shots, 1)
    xg_proxy = (conversion_rate * 0.3 + danger_ratio * 0.7) * shots
    return min(xg_proxy, 8.0)  # Maksimum sÃ„Â±nÃ„Â±r / Maximum limit

# 13. Kadro Etkisi / Squad Effect
def calculate_squad_impact(missing_players: int, team_value_change: float) -> float:
    """Kadro/eksik oyuncu etkisi hesaplar / Calculates squad/missing player impact"""
    if missing_players >= 3:
        return -8.0  # %8 negatif etki / 8% negative impact
    elif missing_players >= 2:
        return -5.0  # %5 negatif etki / 5% negative impact
    elif missing_players == 1:
        return -2.0  # %2 negatif etki / 2% negative impact
    else:
        # TakÃ„Â±m deÃ„Å¸eri deÃ„Å¸iÃ…Å¸imine gÃƒÂ¶re etki / Impact based on team value change
        return max(min(team_value_change * 10, 5.0), -5.0)

# 14. Multi-market Konsistensi / Multi-market Consistency
def check_market_consistency(odds_1x2: Dict, odds_ah: Dict, odds_total: Dict) -> float:
    """Ãƒâ€¡oklu pazar tutarlÃ„Â±lÃ„Â±k kontrolÃƒÂ¼ / Multi-market consistency check"""
    consistency_score = 100.0
    
    # 1X2 vs Asian Handicap tutarlÃ„Â±lÃ„Â±k / 1X2 vs Asian Handicap consistency
    if odds_1x2 and odds_ah:
        # Basit tutarlÃ„Â±lÃ„Â±k kontrolÃƒÂ¼ / Simple consistency check
        home_prob = 1.0 / odds_1x2.get('home', 3.0)
        ah_prob = 0.5  # Basit varsayÃ„Â±m / Simple assumption
        diff = abs(home_prob - ah_prob)
        if diff > 0.1:  # %10'dan fazla fark / More than 10% difference
            consistency_score -= 20
    
    # TutarsÃ„Â±zlÃ„Â±k iÃƒÂ§in kalibrasyon / Calibration for inconsistency
    calibration = max(consistency_score / 100, 0.8)  # Minimum %80 / Minimum 80%
    return calibration

# 15. GeliÃ…Å¸miÃ…Å¸ Kalite Skoru / Advanced Quality Score
def calculate_advanced_quality(features: Dict) -> float:
    """GeliÃ…Å¸miÃ…Å¸ kalite skoru hesaplama / Advanced quality score calculation"""
    base_score = calculate_quality(features)  # Temel skor / Base score
    
    # Ek faktÃƒÂ¶rler / Additional factors
    bonus_points = 0
    
    # Veri kaynaÃ„Å¸Ã„Â± ÃƒÂ§eÃ…Å¸itliliÃ„Å¸i / Data source diversity
    sources = features.get("data_sources", [])
    if len(sources) >= 3:
        bonus_points += 10
    elif len(sources) >= 2:
        bonus_points += 5
    
    # GÃƒÂ¼ncellik / Freshness
    data_age = features.get("data_age_hours", 48)
    if data_age <= 1:
        bonus_points += 10
    elif data_age <= 6:
        bonus_points += 5
    
    # Model ÃƒÂ§eÃ…Å¸itliliÃ„Å¸i / Model diversity
    models_used = features.get("models_used", 1)
    if models_used >= 3:
        bonus_points += 10
    elif models_used >= 2:
        bonus_points += 5
    
    final_score = min(base_score + bonus_points, 100.0)
    return max(final_score, 0.0)

# ==================== ORTAK FONKSÃ„Â°YONLAR / COMMON FUNCTIONS ====================

def calculate_quality(features: Dict) -> float:
    """Basit kalite skoru (0-100) / Simple quality score (0-100)"""
    keys = ("odds", "weather", "standings", "form", "ranking", "value")
    if not isinstance(features, dict) or not keys:
        return 0.0
    score = sum(1 for k in keys if features.get(k)) / len(keys) * 100.0
    return round(score, 1)

# --- GELÃ„Â°Ã…ÂMÃ„Â°Ã…Â KART/KORNER SÃ„Â°STEMÃ„Â° (Ãƒâ€¡oklu Kaynak) ----------------------------
def get_cards_corners_apifootball(area, comp, home_team, away_team):
    """API-Football'dan kart ve korner verileri"""
    if not APIFOOT:
        return None, "APIF"
    
    try:
        hint = _apifoot_hint_cards_corners(area, comp, home_team, away_team)
        if hint:
            return hint, "APIF"
    except Exception as e:
        log(f"API-Football kart/korner hatasÃ„Â±: {e}")
    
    return None, "APIF"

def get_cards_corners_totalcorner(area, comp, home_team, away_team):
    """TotalCorner fallback - sadece korner verisi"""
    try:
        # TotalCorner API simulasyonu (gerÃƒÂ§ek API entegrasyonu iÃƒÂ§in gÃƒÂ¼ncellenmeli)
        # Bu ÃƒÂ¶rnekte lig ortalamalarÃ„Â± dÃƒÂ¶ndÃƒÂ¼rÃƒÂ¼yoruz
        corner_base = base_from_area(area, LEAGUE_CORNER_BASE, 9.2)
        
        # Basit varyasyon
        corners = corner_base * random.uniform(0.9, 1.1)
        
        return {"mu_corners_hint": corners}, "TC"
    except Exception as e:
        log(f"TotalCorner hatasÃ„Â±: {e}")
        return None, "TC"

def get_cards_corners_footystats(area, comp, home_team, away_team):
    """FootyStats fallback - lig ortalamalarÃ„Â±"""
    try:
        # FootyStats lig ortalamalarÃ„Â±
        cards_base = base_from_area(area, LEAGUE_CARD_BASE, 4.6)
        corner_base = base_from_area(area, LEAGUE_CORNER_BASE, 9.2)
        
        return {
            "mu_cards_hint": cards_base,
            "mu_corners_hint": corner_base
        }, "FS"
    except Exception as e:
        log(f"FootyStats hatasÃ„Â±: {e}")
        return None, "FS"


# --- 6C: Kart/Korner kayÃ„Â±tlarÃ„Â±ndan takÃ„Â±m bazlÃ„Â± ortalama ÃƒÂ¼retimi ---
def _num_or_none(x):
    try:
        if x in (None, ""): return None
        s = str(x).strip().replace(",", ".")
        return float(s)
    except Exception:
        return None

def _aggregate_cards_corners_from_recs(home_team: str, away_team: str, recs: list[dict]):
    """recs: list[dict] (JSON/CSV kayÃ„Â±tlarÃ„Â±). DÃƒÂ¶ndÃƒÂ¼rÃƒÂ¼r: {"mu_cards_hint": x?, "mu_corners_hint": y?} veya None"""
    if not recs:
        return None
    ht = norm_team(home_team); at = norm_team(away_team)
    # candidate columns
    home_keys = ["home","home_team","hometeam","team1","ev","evsahibi","host"]
    away_keys = ["away","away_team","awayteam","team2","dep","deplasman","guest"]
    yh_keys = ["yh","yellow_home","home_yellow","y_h","yellowcards_home","yellow_cards_home","yellow_h","yhcards","hy"]
    ya_keys = ["ya","yellow_away","away_yellow","y_a","yellowcards_away","yellow_cards_away","yellow_a","yacards","ay"]
    rh_keys = ["rh","red_home","home_red","r_h","redcards_home","red_cards_home","red_h","rhcards","hr"]
    ra_keys = ["ra","red_away","away_red","r_a","redcards_away","red_cards_away","red_a","racards","ar"]
    ch_keys = ["ch","corners_home","home_corners","c_h","corner_home","homecorner","corner_h","cornersh","hc"]
    ca_keys = ["ca","corners_away","away_corners","c_a","corner_away","awaycorner","corner_a","cornersa","ac"]
    tot_cards_keys = ["cards","total_cards","cards_total","fulltime_cards","ft_cards"]
    tot_corners_keys = ["corners","total_corners","corners_total","ft_corners","fulltime_corners"]

    def pick(d, keys):
        for k in keys:
            if k in d and d[k] not in (None, ""):
                return d[k]
        return None

    # accumulate per team
    stats = {ht: {"cards": [], "corners": []}, at: {"cards": [], "corners": []}}
    for r in recs:
        if not isinstance(r, dict): 
            continue
        d = { (k.strip().lower() if isinstance(k,str) else k): v for k,v in r.items() }
        hv, av = None, None
        for k in home_keys:
            if k in d and d[k]:
                hv = norm_team(d[k]); break
        for k in away_keys:
            if k in d and d[k]:
                av = norm_team(d[k]); break
        if not hv or not av:
            continue
        # Only consider rows where at least one of the two teams matches
        if hv not in (ht, at) and av not in (ht, at):
            continue
        # cards
        yh = _num_or_none(pick(d, yh_keys)); ya = _num_or_none(pick(d, ya_keys))
        rh = _num_or_none(pick(d, rh_keys)); ra = _num_or_none(pick(d, ra_keys))
        tot_cards = _num_or_none(pick(d, tot_cards_keys))
        if tot_cards is None:
            # Common assumption: red ~ 2 yellows
            comp = 0.0
            cnt = 0
            if yh is not None and ya is not None:
                comp = (yh + ya); cnt = 1
            if rh is not None and ra is not None:
                comp = (comp if cnt else 0.0) + 2.0*(rh + ra); cnt = 1
            if cnt:
                tot_cards = comp
        # corners
        ch = _num_or_none(pick(d, ch_keys)); ca = _num_or_none(pick(d, ca_keys))
        tot_corners = _num_or_none(pick(d, tot_corners_keys))
        if tot_corners is None and ch is not None and ca is not None:
            tot_corners = ch + ca
        # push per matched team
        if hv == ht or av == ht:
            if tot_cards is not None:   stats[ht]["cards"].append(float(tot_cards))
            if tot_corners is not None: stats[ht]["corners"].append(float(tot_corners))
        if hv == at or av == at:
            if tot_cards is not None:   stats[at]["cards"].append(float(tot_cards))
            if tot_corners is not None: stats[at]["corners"].append(float(tot_corners))

    out_cards = None; out_corners = None
    def mean(lst): 
        return (sum(lst)/len(lst)) if lst else None

    m_ht_c = mean(stats[ht]["cards"]);    m_at_c = mean(stats[at]["cards"])
    m_ht_cn = mean(stats[ht]["corners"]); m_at_cn = mean(stats[at]["corners"])
    # combine: average of each team means (robust, simetrik)
    if m_ht_c is not None and m_at_c is not None:
        out_cards = (m_ht_c + m_at_c) / 2.0
    if m_ht_cn is not None and m_at_cn is not None:
        out_corners = (m_ht_cn + m_at_cn) / 2.0

    out = {}
    if out_cards is not None: out["mu_cards_hint"] = float(out_cards)
    if out_corners is not None: out["mu_corners_hint"] = float(out_corners)
    return out or None

def get_cards_corners_advanced(area, comp, home_team, away_team):
    """GeliÃ…Å¸tirilmiÃ…Å¸ kart/korner sistemi - zincirli fallback"""
    # 1. Ãƒâ€“ncelik: API-Football
    result, source = get_cards_corners_apifootball(area, comp, home_team, away_team)
    if result:
        log(f"Kart/Korner verisi {source}'dan alÃ„Â±ndÃ„Â±: {home_team} vs {away_team}")
        return result, source
    
    # 2. Fallback: TotalCorner (sadece korner)
    result, source = get_cards_corners_totalcorner(area, comp, home_team, away_team)
    if result and "mu_corners_hint" in result:
        log(f"Korner verisi {source}'dan alÃ„Â±ndÃ„Â±: {home_team} vs {away_team}")
        # Kart verisi iÃƒÂ§in FootyStats'e ihtiyaÃƒÂ§ var
        cards_result, cards_source = get_cards_corners_footystats(area, comp, home_team, away_team)
        if cards_result and "mu_cards_hint" in cards_result:
            result["mu_cards_hint"] = cards_result["mu_cards_hint"]
            source = f"TC+{cards_source}"
        return result, source
    
    # 3. Fallback: FootyStats (hem kart hem korner)
    result, source = get_cards_corners_footystats(area, comp, home_team, away_team)
    if result:
        log(f"Kart/Korner verisi {source}'dan alÃ„Â±ndÃ„Â±: {home_team} vs {away_team}")
        return result, source
    
    # 4. Son ÃƒÂ§are: lig bazlÃ„Â± ortalamalar
    cards_base = base_from_area(area, LEAGUE_CARD_BASE, 4.6)
    corner_base = base_from_area(area, LEAGUE_CORNER_BASE, 9.2)    # 3. Fallback: KullanÃ„Â±cÃ„Â± URL listeleri (resmÃƒÂ® Ã¢â€ â€™ aÃƒÂ§Ã„Â±k kaynak)
    try:
        urls_cards = (os.getenv("CARDS_URLS") or "").strip()
        urls_corners = (os.getenv("CORNERS_URLS") or "").strip()
        merged = None
        if urls_cards:
            recs = fetch_records(urls_cards, kind="club")
            agg = _aggregate_cards_corners_from_recs(home_team, away_team, recs) if recs else None
            if agg:
                merged = dict(merged or {}); merged.update(agg)
        if urls_corners:
            recs2 = fetch_records(urls_corners, kind="club")
            agg2 = _aggregate_cards_corners_from_recs(home_team, away_team, recs2) if recs2 else None
            if agg2:
                merged = dict(merged or {}); merged.update(agg2)
        if merged and ("mu_cards_hint" in merged or "mu_corners_hint" in merged):
            log(f"Kart/Korner verisi URL fallback ile alÃ„Â±ndÃ„Â±: {home_team} vs {away_team}")
            return merged, "URL"
    except Exception as e:
        log(f"URL kart/korner fallback hatasÃ„Â±: {e}")
    # 4. Fallback: Yerel snapshot
    try:
        import json, csv, io
        # TekleÃ…Å¸miÃ…Å¸ bir snapshot dosyasÃ„Â± varsayalÃ„Â±m; isim yoksa atla
        for path in ["data/cards_corners.json", "data/cards_corners.csv"]:
            if not os.path.exists(path):
                continue
            if path.endswith(".json"):
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
                recs = obj if isinstance(obj, list) else None
            else:
                with open(path, "r", encoding="utf-8") as f:
                    rdr = csv.DictReader(f)
                    recs = [ { (k.strip().lower() if isinstance(k,str) else k): v for k,v in row.items() } for row in rdr ]
            if recs:
                agg = _aggregate_cards_corners_from_recs(home_team, away_team, recs)
                if agg and ("mu_cards_hint" in agg or "mu_corners_hint" in agg):
                    log(f"Kart/Korner verisi yerel snapshot ile alÃ„Â±ndÃ„Â±: {home_team} vs {away_team}")
                    return agg, "LOCAL"
    except Exception as e:
        log(f"Yerel kart/korner fallback hatasÃ„Â±: {e}")

    
    log(f"Kart/Korner verisi DEFAULT'tan alÃ„Â±ndÃ„Â±: {home_team} vs {away_team}")
    return {
        "mu_cards_hint": cards_base,
        "mu_corners_hint": corner_base
    }, "DEFAULT"

# --- DENGELENMÃ„Â°Ã…Â EV SAHÃ„Â°BÃ„Â° AVANTAJI -----------------------------------------
def home_adv_effective(area, competition, home_team, away_team):
    """
    Dinamik ev sahibi avantajÃ„Â± hesaplar - DENGELENMÃ„Â°Ã…Â
    """
    base_advantage = ELO_HOME_ADV
    
    # Milli takÃ„Â±m maÃƒÂ§larÃ„Â±nda ev avantajÃ„Â±nÃ„Â± azalt
    comp_lower = (competition or "").lower()
    if "world cup" in comp_lower or "euro" in comp_lower or "qualification" in comp_lower:
        base_advantage *= 0.6  # %40 azalt
        log(f"Ã¢Å¡Â½ Milli takÃ„Â±m maÃƒÂ§Ã„Â± - ev avantajÃ„Â± azaltÃ„Â±ldÃ„Â±: {base_advantage:.1f}")
    
    # Kadro deÃ„Å¸eri etkisi
    value_advantage, value_source = calculate_value_advantage(home_team, away_team, area)
    
    # EÃ„Å¸er her iki takÃ„Â±m da default deÃ„Å¸erdeyse, ev avantajÃ„Â±nÃ„Â± sÃ„Â±fÃ„Â±rla
    if "DEFAULT" in value_source or "CIES_DEFAULT" in value_source:
        value_factor = 0.5  # %50 azalt
        log(f"Ã¢Å¡â€“Ã¯Â¸Â Default deÃ„Å¸erler - ev avantajÃ„Â± azaltÃ„Â±ldÃ„Â±")
    else:
        value_factor = 1.0 - abs(value_advantage) * 2.0
    
    final_advantage = base_advantage * value_factor
    
    log(f"Ev avantajÃ„Â±: {home_team} vs {away_team} -> {final_advantage:.1f} "
        f"(base: {ELO_HOME_ADV}, value_factor: {value_factor:.2f})")
    
    return clamp(final_advantage, 10.0, 80.0)  # Min 10, max 80

# --- LÃ„Â°G/KUPA FÃ„Â°LTRESÃ„Â° -------------------------------------------------------
# KullanÃ„Â±cÃ„Â± isteÃ„Å¸i: yalnÃ„Â±zca Ã…Å¸u lig/kupalar:
# Ã„Â°ngiltere: Premier League, Championship
# Ã„Â°spanya: La Liga, La Liga 2
# Ã„Â°talya: Serie A, Serie B
# Almanya: Bundesliga, 2. Bundesliga
# Fransa: Ligue 1, Ligue 2
# TÃƒÂ¼rkiye: SÃƒÂ¼per Lig, 1. Lig
# Hollanda: Eredivisie, Eerste Divisie
# Portekiz: Primeira Liga, Liga Portugal 2
# BelÃƒÂ§ika: Pro League, Challenger Pro League
# UEFA: UCL, UEL, UECL, Super Cup, EURO, EURO Elemeleri, Nations League & Finals
# FIFA: World Cup, Club World Cup

def _n(s):
    # normalize for comparisons
    s = (s or "").strip().lower()
    s = s.replace("trendyol ", "")
    s = s.replace("division", "divisiÃƒÂ³n").replace("segunda", "la liga 2")
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
    ("turkey", "sÃƒÂ¼per lig"),
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

# UEFA/FIFA turnuvalarÃ„Â± ad bazlÃ„Â± kabul
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

def is_allowed_competition(area_name: str, comp_name: str) -> bool:
    # KadÃ„Â±n liglerini filtrele
    if is_women_competition(area_name, comp_name):
        return False
        
    # U-21 liglerini filtrele
    if is_u21_competition(comp_name):
        return False
        
    a, c = _n(area_name), _n(comp_name)
    if (a, c) in _ALLOWED_PAIRS:
        return True
    if a in ("turkey", "tÃƒÂ¼rkiye") and c in ("super lig", "sÃƒÂ¼per lig", "1. lig"):
        return True
    if a == "germany" and c in ("2. bundesliga", "bundesliga 2"):
        return True
    if a == "spain" and c in ("la liga 2", "segunda divisiÃƒÂ³n", "laliga2", "segundadivisiÃƒÂ³n"):
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

# Elo / Form ayarlarÃ„Â±
ELO_K = float(os.getenv("ELO_K", "24"))
ELO_HOME_ADV = float(os.getenv("ELO_HOME_ADV", "40"))  # DÃƒÅ“Ã…ÂÃƒÅ“RÃƒÅ“LDÃƒÅ“: 60 -> 40
FORM_LOOKBACK = int(os.getenv("FORM_LOOKBACK", "10"))
FORM_DAYS = int(os.getenv("FORM_DAYS", "120"))
ALLOW_STATE_FILE = (os.getenv("ALLOW_STATE_FILE", "1") == "1")

# Otomatik ÃƒÂ¶Ã„Å¸renme ayarlarÃ„Â±
W_MKT_INIT = float(os.getenv("W_MKT_INIT", "0.45"))  # ARTIRILDI: 0.35 -> 0.45
LEARN_RATE = float(os.getenv("LEARN_RATE", "0.05"))
GOAL_LR = float(os.getenv("GOAL_LR", "0.02"))
PRED_MATCH_WINDOW_HRS = int(os.getenv("PRED_MATCH_WINDOW_HRS", "48"))

# Table/Streak ayarlarÃ„Â±
TABLE_WEIGHT = float(os.getenv("TABLE_WEIGHT", "0.12"))
STREAK_UNIT = float(os.getenv("STREAK_UNIT", "0.02"))
STREAK_MAX = float(os.getenv("STREAK_MAX", "0.08"))

# SERVICE modu zaman ayarlarÃ„Â± (TR saati)
PREDICTION_HOUR = int(os.getenv("PREDICTION_HOUR", "10"))
RESULTS_HOUR = int(os.getenv("RESULTS_HOUR", "4"))  # <Ã¢â‚¬â€ ONAYLI: Ertesi gÃƒÂ¼n 04:00
RESULTS_MINUTE = int(os.getenv("RESULTS_MINUTE", "0"))  # <Ã¢â‚¬â€ ONAYLI: :00

if not (GMAIL_USER and GMAIL_PASS and GMAIL_TO):
    raise SystemExit("GMAIL_USER/GMAIL_PASS/GMAIL_TO secrets eksik.")

# --- AkÃ„Â±llÃ„Â± Hava Modu --------------------------------------------------------
WEATHER_SMART = (os.getenv("WEATHER_SMART", "1") == "1")
WEATHER_AREAS = set(
    s.strip().lower() for s in (os.getenv("WEATHER_AREAS", "England,Spain,Italy,Germany,France,Turkey,Netherlands,Portugal,Belgium").split(",")) if s.strip()
)

def weather_enabled(area_name: str, comp_name: str) -> bool:
    """Hava sadece bÃƒÂ¼yÃƒÂ¼k lig ve UEFA/FIFA maÃƒÂ§larÃ„Â±nda alÃ„Â±nsÃ„Â±n (performans iÃƒÂ§in)."""
    a = _n(area_name)
    c = _n(comp_name)
    if a in {s.lower() for s in WEATHER_AREAS}:
        return True
    if any(p in c for p in _UEFA_ALLOW_PAT) or any(p in c for p in _FIFA_ALLOW_PAT):
        return True
    return False

# --- Hava: takÃ„Â±m -> Ã…Å¸ehir eÃ…Å¸leÃ…Å¸mesi ------------------------------------------
def guess_city_from_team(team_name: str):
    t = (team_name or "").lower()
    overrides = {
        # TÃƒÂ¼rkiye
        "galatasaray": "Istanbul",
        "fenerbahce": "Istanbul",
        "beÃ…Å¸iktaÃ…Å¸": "Istanbul",
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
        # Ã„Â°spanya
        "real madrid": "Madrid",
        "barcelona": "Barcelona",
        "atlÃƒÂ©tico": "Madrid",
        "atletico": "Madrid",
        # Ã„Â°talya
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
        "grÃƒÂªmio": "Porto Alegre",
        "internacional": "Porto Alegre",
        "atletico mineiro": "Belo Horizonte",
        # Milli takÃ„Â±mlar
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
        return f"Hava: {tavg:.0f}Ã‚Â°C, yaÃ„Å¸Ã„Â±Ã…Å¸ {pavg:.1f}mm, rÃƒÂ¼zgÃƒÂ¢r {wavg:.0f} km/s"
    except Exception:
        return None

def parse_weather(wx_text):
    if not wx_text:
        return (None, None)
    wind = None; precip = None
    try:
        if "rÃƒÂ¼zgÃƒÂ¢r" in wx_text:
            wind = safe_float(wx_text.split("rÃƒÂ¼zgÃƒÂ¢r")[1].split("km/s")[0].strip().split()[-1], None)
        if "yaÃ„Å¸Ã„Â±Ã…Å¸" in wx_text:
            precip = safe_float(wx_text.split("yaÃ„Å¸Ã„Â±Ã…Å¸")[1].split("mm")[0].strip().split()[-1], None)
    except Exception:
        pass
    return (wind, precip)

# --- Football-Data (fixtures) ------------------------------------------------
def fetch_fd_fixtures(date_str):
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
    log(f"FD fixtures (TR={date_str}) -> {len(out)}")
    return out

# --- 2. Fallback: API-Football (tarih bazlÃ„Â±) ---------------------------------
def fetch_apifoot_fixtures(date_str):
    if not APIFOOT:
        return []
    headers = {"x-apisports-key": APIFOOT}
    url = "https://v3.football.api-sports.io/fixtures"
    data = http_get(url, headers=headers, params={"date": date_str})
    out = []
    if not data:
        log(f"APIF fixtures (TR={date_str}) -> 0 (boÃ…Å¸/eriÃ…Å¸ilemedi)")
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
    log(f"APIF fixtures (TR={date_str}) -> {len(out)}")
    return out

# --- OpenLigaDB fallback (yalnÃ„Â±z Almanya alt ligleri) ------------------------
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

# --- FikstÃƒÂ¼r toplayÃ„Â±cÃ„Â± zincir ------------------------------------------------
def fetch_fixtures(date_str):
    fixtures = fetch_fd_fixtures(date_str)
    if not fixtures:
        log("FD boÃ…Å¸ Ã¢â€ â€™ API-Football fallback deneniyorÃ¢â‚¬Â¦")
        fixtures = fetch_apifoot_fixtures(date_str)
    if not fixtures:
        log("API-Football da boÃ…Å¸ Ã¢â€ â€™ OpenLigaDB fallback deneniyorÃ¢â‚¬Â¦")
        fixtures = fetch_openligadb_day(date_str)
    if not fixtures:
        log("OpenLigaDB de boÃ…Å¸ Ã¢â€ â€™ The Odds API event fallback deneniyorÃ¢â‚¬Â¦")
        fixtures = fetch_odds_fixtures(date_str)
    return fixtures

# --- Odds (avg) --------------------------------------------------------------

# --- Guess sport key for The Odds API --------------------------------------
def guess_sport_key(area, comp):
    area_l = (area or "").lower()
    comp_l = (comp or "").lower()
    mapping = {
        ("england", "premier league"): "soccer_epl",
        ("england", "championship"): "soccer_efl_championship",
        ("spain", "la liga"): "soccer_spain_la_liga",
        ("italy", "serie a"): "soccer_italy_serie_a",
        ("germany", "bundesliga"): "soccer_germany_bundesliga",
        ("france", "ligue 1"): "soccer_france_ligue_one",
        ("turkey", "super lig"): "soccer_turkey_super_league",
        ("europe", "uefa champions league"): "soccer_uefa_champs_league",
        ("europe", "uefa europa league"): "soccer_uefa_europa_league",
    }
    for (a, c), skey in mapping.items():
        if a in area_l and c in comp_l:
            return skey
    return None


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
        # GeliÃ…Å¸tirilmiÃ…Å¸ takÃ„Â±m eÃ…Å¸leÃ…Å¸tirme - benzerlik kullan
        h_sim = team_similarity(h, t1)
        a_sim = team_similarity(a, t2)
        if h_sim >= 0.75 and a_sim >= 0.75:  # %75 benzerlik eÃ…Å¸iÃ„Å¸i
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

# --- Poisson + Lig tabanÃ„Â± ----------------------------------------------------
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

# --- Kart / Korner bazlarÃ„Â± ---------------------------------------------------
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

# --- GELÃ„Â°Ã…ÂMÃ„Â°Ã…Â DERECELENDÃ„Â°RME (Kaynak Etiketleme) -----------------------------
def model_cards_corners(area, lam_h, lam_a, wx_text, apifoot_hint=None, source_info=""):
    cards_base = base_from_area(area, LEAGUE_CARD_BASE, 4.6)
    corner_base = base_from_area(area, LEAGUE_CORNER_BASE, 9.2)
    
    wind, precip = parse_weather(wx_text) if wx_text else (None, None)
    tempo_factor = clamp((lam_h + lam_a) / 2.6, 0.7, 1.4)
    
    cards = cards_base
    corners = corner_base * tempo_factor
    
    if apifoot_hint:
        if apifoot_hint.get("mu_cards_hint"):
            cards = 0.6 * cards + 0.4 * max(0.1, apifoot_hint["mu_cards_hint"])
        if apifoot_hint.get("mu_corners_hint"):
            corners = 0.6 * corners + 0.4 * max(0.1, apifoot_hint["mu_corners_hint"])
    
    if precip is not None:
        cards *= 1.00 + min(0.10, 0.02 * max(0.0, precip))
        corners *= 1.00 - min(0.10, 0.015 * max(0.0, precip))
    if wind is not None:
        corners *= 1.00 + min(0.08, 0.003 * max(0.0, wind))
        cards *= 1.00 + min(0.05, 0.002 * max(0.0, wind))
    
    p_corners_8_5 = poisson_over_prob(max(0.1, corners), 8.5)
    p_corners_9_5 = poisson_over_prob(max(0.1, corners), 9.5)
    p_cards_3_5 = poisson_over_prob(max(0.1, cards), 3.5)
    p_cards_4_5 = poisson_over_prob(max(0.1, cards), 4.5)
    
    return {
        "mu_cards": cards,
        "mu_corners": corners,
        "p_over_cards_3_5": p_cards_3_5,
        "p_over_cards_4_5": p_cards_4_5,
        "p_over_corners_8_5": p_corners_8_5,
        "p_over_corners_9_5": p_corners_9_5,
        "source": source_info
    }

def rate_fixture(fx, odds_info):
    area = fx["area"] or "Europe"
    tot = base_total_goals(area)
    
    # Dinamik ev sahibi avantajÃ„Â± - DENGELENMÃ„Â°Ã…Â
    home_advantage = home_adv_effective(area, fx.get("competition",""), fx["home"], fx["away"])
    
    ah = 1.12
    noise = (len((fx["home"] or "")) - len((fx["away"] or ""))) * 0.01
    lam_h = max(0.2, tot*0.5*ah + noise)
    lam_a = max(0.2, tot*0.5*(2 - ah) - noise)
    
    # Hava (AkÃ„Â±llÃ„Â± Mod)
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
    
    # Elo etkisi (milli takÃ„Â±m destekli)
    Eh = elo_get(area, fx["home"]); Ea = elo_get(area, fx["away"])
    elo_diff = (Eh + home_advantage) - Ea
    elo_adj = clamp((elo_diff/400.0)*0.15, -0.20, 0.20)
    lam_h *= (1.0 + elo_adj); lam_a *= (1.0 - elo_adj)
    
    # Kadro deÃ„Å¸eri avantajÃ„Â± (Transfermarkt + Fallback)
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
    else:
        lig_key = f"{(fx.get('area') or '').strip()}|{(fx.get('competition') or '').strip()}"
        lig_id = _API_LEAGUE_MAP.get(lig_key)
        if lig_id:
            ssn = season_for_today()
            h_pos, N = _apif_get_position_by_name(lig_id, ssn, fx.get("home"))
            a_pos, _ = _apif_get_position_by_name(lig_id, ssn, fx.get("away"))
            if h_pos and a_pos and N:
                h_rankpct = (N - h_pos) / (N - 1) if N > 1 else 0.5
                a_rankpct = (N - a_pos) / (N - 1) if N > 1 else 0.5
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
    
    # Market karÃ„Â±Ã…Å¸Ã„Â±mÃ„Â±
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
    
    # GELÃ„Â°Ã…ÂMÃ„Â°Ã…Â Kart/Korner Ã¢â‚¬â€ Ãƒâ€¡oklu Kaynak Fallback
    apihint, kk_source = get_cards_corners_advanced(fx.get("area"), fx.get("competition"), fx.get("home"), fx.get("away"))
    
    kk = model_cards_corners(area, lam_h, lam_a, wx, apifoot_hint=apihint, source_info=kk_source)
    
    # Kaynak etiketli ÃƒÂ§Ã„Â±ktÃ„Â±
    kk_txt = (f" | Korner ÃÂ¼Ã¢â€°Ë†{kk['mu_corners']:.1f} (ÃƒÅ“st8.5 {int(kk['p_over_corners_8_5']*100)}% / "
              f"ÃƒÅ“st9.5 {int(kk['p_over_corners_9_5']*100)}%) [{kk['source']}]"
              f" | Kart ÃÂ¼Ã¢â€°Ë†{kk['mu_cards']:.1f} (ÃƒÅ“st3.5 {int(kk['p_over_cards_3_5']*100)}%) [{kk['source']}]")
    
    # Kadro deÃ„Å¸eri bilgisi (kaynak etiketli)
    home_value, home_source = get_team_value(fx["home"], area)
    away_value, away_source = get_team_value(fx["away"], area)
    value_txt = f" | Kadro: {home_value:.0f}MÃ¢â€šÂ¬ [{home_source}] vs {away_value:.0f}MÃ¢â€šÂ¬ [{away_source}]"
    
    wx_txt = f" | {wx}" if wx else ""
    note = (f"SeÃƒÂ§im: {pick} | GÃƒÂ¼ven: {conf_pct}% | ÃÂ»_h/ÃÂ»_a: {lam_h:.2f}/{lam_a:.2f}"
            f"{wx_txt}{odds_txt}{kk_txt}{value_txt}{form_txt}{table_txt}{streak_txt}")
    
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
        "kk_source": kk_source
    }

# --- Tahmin/sonuÃƒÂ§ eÃ…Å¸leÃ…Å¸me & ÃƒÂ¶Ã„Å¸renme yardÃ„Â±mcÃ„Â±larÃ„Â± -----------------------------
def match_key_from_fixture(fx):
    if fx.get("id"):
        return f"{fx.get('source','?')}:{fx['id']}"
    dt = (fx.get("utc_kickoff") or datetime.now(timezone.utc)).strftime("%Y%m%d")
    return f"{norm_team(fx.get('home'))}|{norm_team(fx.get('away'))}|{dt}"

def alt_key_from_names(home, away, date_str):
    return f"{norm_team(home)}|{norm_team(away)}|{date_str.replace('-','')}"

def find_prediction_for_result(result):
    """SonuÃƒÂ§ iÃƒÂ§in tahmin bulur (yakÃ„Â±n isim eÃ…Å¸leÃ…Å¸tirmeli)"""
    home, away, date_str = result["home"], result["away"], result.get("date", "")
    
    # Ãƒâ€“nce tam eÃ…Å¸leÃ…Å¸me dene
    altk = alt_key_from_names(home, away, date_str)
    if altk in STATE["pred_store"]:
        return STATE["pred_store"][altk]
    
    # ID bazlÃ„Â± eÃ…Å¸leÃ…Å¸me
    if result.get("id_key") and result["id_key"] in STATE["pred_store"]:
        return STATE["pred_store"][result["id_key"]]
    
    # YakÃ„Â±n isim eÃ…Å¸leÃ…Å¸tirmesi - GELÃ„Â°Ã…ÂTÃ„Â°RÃ„Â°LMÃ„Â°Ã…Â VERSÃ„Â°YON
    all_pred_keys = list(STATE["pred_store"].keys())
    all_team_pairs = []
    
    for key in all_pred_keys:
        if "|" in key and key.count("|") == 2:
            parts = key.split("|")
            if len(parts) == 3:
                pred_home, pred_away, pred_date = parts
                if pred_date == date_str.replace("-", ""):
                    all_team_pairs.append((pred_home, pred_away, key))
    
    # Ãƒâ€¡ift yÃƒÂ¶nlÃƒÂ¼ eÃ…Å¸leÃ…Å¸tirme - GELÃ„Â°Ã…ÂTÃ„Â°RÃ„Â°LMÃ„Â°Ã…Â
    best_match = None
    best_score = 0.0
    
    for pred_home, pred_away, key in all_team_pairs:
        # Normal eÃ…Å¸leÃ…Å¸me
        home_sim = team_similarity(home, pred_home)
        away_sim = team_similarity(away, pred_away)
        normal_score = (home_sim + away_sim) / 2
        
        # Ters eÃ…Å¸leÃ…Å¸me (API'de home/away Ã…Å¸aÃ…Å¸mÃ„Â±Ã…Å¸ olabilir)
        home_sim_rev = team_similarity(home, pred_away)
        away_sim_rev = team_similarity(away, pred_home)
        reverse_score = (home_sim_rev + away_sim_rev) / 2
        
        # En iyi skoru seÃƒÂ§
        current_score = max(normal_score, reverse_score)
        
        if current_score > best_score and current_score >= 0.75:  # %75 benzerlik eÃ…Å¸iÃ„Å¸i
            best_score = current_score
            best_match = key
            
            if current_score == reverse_score:
                log(f"Ters eÃ…Å¸leÃ…Å¸me bulundu: {home}/{away} Ã¢â€°Ë† {pred_away}/{pred_home} "
                    f"(benzerlik: {current_score:.2f})")
            else:
                log(f"Normal eÃ…Å¸leÃ…Å¸me bulundu: {home}/{away} Ã¢â€°Ë† {pred_home}/{pred_away} "
                    f"(benzerlik: {current_score:.2f})")
    
    if best_match:
        return STATE["pred_store"][best_match]
    
    return None

def brier_score(probs, outcome_idx):
    if probs is None:
        return None
    y = [0.0, 0.0, 0.0]; y[outcome_idx] = 1.0
    return sum((p - t)**2 for p, t in zip(probs, y)) / 3.0

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
        "kk_source": rated.get("kk_source", "NONE")
    }
    STATE["pred_store"][mk] = rec
    
    # Ã„Â°kincil anahtar: isim+tarih
    altk = alt_key_from_names(fx.get("home"), fx.get("away"), (fx.get("utc_kickoff") or datetime.now(timezone.utc)).astimezone(TR_TZ).strftime("%Y-%m-%d"))
    STATE["pred_store"][altk] = rec

# --- Mail --------------------------------------------------------------------
def send_mail(subject, body):
    body = (body or "").strip()
    if not body:
        body = "(Bu e-postada iÃƒÂ§erik ÃƒÂ¼retilemedi / maÃƒÂ§ bulunamadÃ„Â±.)"
    msg = EmailMessage()
    msg["From"] = GMAIL_USER; msg["To"] = GMAIL_TO; msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(GMAIL_USER, GMAIL_PASS); s.send_message(msg)
    log(f"Mail gÃƒÂ¶nderildi: {subject}")

# --- SonuÃƒÂ§ ÃƒÂ§ekiciler ---------------------------------------------------------
def fetch_results_fd(date_str):
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
        if not is_allowed_competition(area, cname):
            continue
        score_ft = ((m.get("score") or {}).get("fullTime") or {})
        gh, ga = score_ft.get("home"), score_ft.get("away")
        if gh is None or ga is None:
            # bazen PENS vs, yine de fulltime al
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
    log(f"FD results {date_str} -> {len(out)}")
    return out

def fetch_results_apifoot(date_str):
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
        if not is_allowed_competition(area, cname):
            continue
        st = (((item.get("fixture") or {}).get("status") or {}).get("short") or "").upper()
        if st not in ("FT","AET","PEN","MATCH_FINISHED"):
            # bitmiÃ…Å¸ say
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
    log(f"APIF results {date_str} -> {len(out)}")
    return out

def fetch_results(date_str):
    r = fetch_results_fd(date_str)
    if not r:
        log("FD sonuÃƒÂ§ yok Ã¢â€ â€™ API-Football sonuÃƒÂ§ deneniyorÃ¢â‚¬Â¦")
        r = fetch_results_apifoot(date_str)
    return r

# --- Raporlar ----------------------------------------------------------------
def report_predictions(date_str):
    fixtures = fetch_fixtures(date_str)
    if not fixtures:
        send_mail(f"GÃƒÂ¼nÃƒÂ¼n Tahminleri | {date_str}", "BugÃƒÂ¼n iÃƒÂ§in tahmin ÃƒÂ§Ã„Â±karÃ„Â±lacak maÃƒÂ§ bulunamadÃ„Â±.")
        return
    
    lines = [f"Ã¢Å¡Â½ GÃƒÂ¼nÃƒÂ¼n Tahminleri Ã¢â‚¬â€ {date_str} (Transfermarkt + Milli TakÃ„Â±m Elo + CIES/FootyStats + API-Football + TotalCorner + Kaynak Etiketleme + Ev Sahibi Dengeleme)\n"]
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
        line = f"- {ko_local} | {fx.get('area','')} {fx.get('competition','')} | {fx['home']} vs {fx['away']} Ã¢â‚¬â€ {rated['note']}"
        lines.append(line)
        
        bucket = hi if rated["confidence"] >= HIGH_ALERT else top
        bucket.append((rated["confidence"], line))
    
    if len(lines) == 1:
        lines.append("Filtreler nedeniyle listelenecek maÃƒÂ§ kalmadÃ„Â± (MIN_CONF yÃƒÂ¼ksek olabilir).")
    
    top.sort(reverse=True)
    best = [f"\nÄŸÅ¸Ââ€  En GÃƒÂ¼ÃƒÂ§lÃƒÂ¼ {TOP_N} SeÃƒÂ§im:"] + [" " + l.replace("- ","").strip() for c, l in top[:TOP_N]]
    
    hi_block = []
    if hi:
        hi.sort(reverse=True)
        hi_block.append("\nÄŸÅ¸â€â€ YÃƒÂ¼ksek GÃƒÂ¼ven SeÃƒÂ§imler:")
        for c, l in hi:
            hi_block.append(" " + l.replace("- ","").strip())
    
    body = "\n".join(lines + [""] + best + hi_block)
    save_state(STATE)
    send_mail(f"GÃƒÂ¼nÃƒÂ¼n Tahminleri | {date_str}", body)

def report_results(date_str):
    results = fetch_results(date_str)
    lines = [f"ÄŸÅ¸â€œÅ  GÃƒÂ¼nÃƒÂ¼n SonuÃƒÂ§larÃ„Â± Ã¢â‚¬â€ {date_str}", ""]
    
    if not results:
        lines.append("BugÃƒÂ¼n iÃƒÂ§in sonuÃƒÂ§ bulunamadÃ„Â±.")
        send_mail(f"GÃƒÂ¼nÃƒÂ¼n SonuÃƒÂ§larÃ„Â± | {date_str}", "\n".join(lines))
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
        
        # GeliÃ…Å¸tirilmiÃ…Å¸ tahmin bulma
        pred = find_prediction_for_result(res)
        fuzzy_used = False
        
        if not pred:
            lines.append(f"Ã¢Ââ€œ {res['home']} {gh}-{ga} {res['away']} (tahmin bulunamadÃ„Â±)")
            continue
        
        # YakÃ„Â±n eÃ…Å¸leÃ…Å¸me kullanÃ„Â±ldÃ„Â±ysa iÃ…Å¸aretle
        if "ÄŸÅ¸â€Â" in str(pred.get("note", "")):
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
        
        # Elo ÃƒÂ¶Ã„Å¸renme - dinamik ev avantajÃ„Â± ile
        result_hw = 1.0 if outcome_idx==0 else 0.0 if outcome_idx==2 else 0.5
        
        # Ãƒâ€“ncelikle kayÃ„Â±tlÃ„Â± home_advantage deÃ„Å¸erini kullan
        home_advantage = pred.get("home_advantage")
        if home_advantage is None:
            # Yedek: maÃƒÂ§ bilgileriyle yeniden hesapla
            home_advantage = home_adv_effective(
                area, res.get("competition", ""), res["home"], res["away"]
            )
        
        elo_update(area, res["home"], res["away"], result_hw, home_advantage)
        
        # goal_scale ÃƒÂ¶Ã„Å¸renme
        goals = gh + ga
        cur_scale = get_goal_scale(area)
        expected_tot = base_total_goals(area)
        err = (goals - expected_tot) / max(1.0, expected_tot)
        new_scale = clamp(cur_scale * (1.0 + GOAL_LR * err), 0.7, 1.4)
        set_goal_scale(area, new_scale)
        
        s, n = goal_stats.get(area, (0,0))
        goal_stats[area] = (s+goals, n+1)
        
        mark = "Ã¢Å“â€¦" if ok else "Ã¢ÂÅ’"
        fuzzy_indicator = " ÄŸÅ¸â€Â" if fuzzy_used else ""
        lines.append(f"{mark}{fuzzy_indicator} {res['home']} {gh}-{ga} {res['away']} | Tahmin: {pred['pick']} ({pred['conf_pct']}%)")
    
    # w_mkt ÃƒÂ¶Ã„Å¸renme (model vs market performansÃ„Â±na gÃƒÂ¶re)
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
        lines.append(f"ÄŸÅ¸ÂÂ¯ DoÃ„Å¸ruluk: {acc:.1f}% | Brier (model/market/blend): "
                    f"{bm_avg:.3f}/{(bk_avg if bk_avg is not None else float('nan')):.3f}/{bb_avg:.3f}")
        lines.append(f"Ã¢Å¡â€“Ã¯Â¸Â w_mkt: {old_w:.2f} Ã¢â€ â€™ {get_w_mkt():.2f}")
        
        if matched_with_fuzzy > 0:
            lines.append(f"ÄŸÅ¸â€Â {matched_with_fuzzy} maÃƒÂ§ yakÃ„Â±n eÃ…Å¸leÃ…Å¸tirme ile bulundu")
    
    if goal_stats:
        lines.append("")
        lines.append("ÄŸÅ¸â€œË† Goal-scale gÃƒÂ¼ncellemeleri:")
        for area, (s, n) in goal_stats.items():
            lines.append(f" - {area}: avg_goals={s/max(1,n):.2f} | goal_scale={get_goal_scale(area):.3f}")
    
    save_state(STATE)
    send_mail(f"GÃƒÂ¼nÃƒÂ¼n SonuÃƒÂ§larÃ„Â± | {date_str}", "\n".join(lines))

# --- SERVICE (otomatik zamanlayÃ„Â±cÃ„Â±) ------------------------------------------
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
    """SÃƒÂ¼rekli ÃƒÂ§alÃ„Â±Ã…Å¸Ã„Â±r; TR 10:00'da bugÃƒÂ¼nÃƒÂ¼n tahmini, ertesi gÃƒÂ¼n TR 04:00'da DÃƒÅ“NÃƒÅ“N sonuÃƒÂ§larÃ„Â±nÃ„Â± gÃƒÂ¶nderir. AynÃ„Â± gÃƒÂ¼n iÃƒÂ§inde tekrarÃ„Â± engellemek iÃƒÂ§in STATE iÃƒÂ§inde tarih izler."""
    log(f"SERVICE baÃ…Å¸latÃ„Â±ldÃ„Â± (TR hedefleri: {PREDICTION_HOUR:02d}:00 ve ertesi gÃƒÂ¼n {RESULTS_HOUR:02d}:{RESULTS_MINUTE:02d} [dÃƒÂ¼ne ait])")
    while True:
        try:
            now_tr = datetime.now(TR_TZ)
            today = _today_str_tr(now_tr)
            
            # Tahmin: bugÃƒÂ¼n 10:00 veya sonrasÃ„Â± ve bugÃƒÂ¼n henÃƒÂ¼z gÃƒÂ¶nderilmemiÃ…Å¸se
            if (STATE.get("last_pred_date") != today) and _time_reached_tr(PREDICTION_HOUR, 0):
                log("SERVICE: Tahmin zamanÃ„Â± geldi Ã¢â€ â€™ rapor hazÃ„Â±rlanÃ„Â±yorÃ¢â‚¬Â¦")
                report_predictions(today)
                STATE["last_pred_date"] = today
                save_state(STATE)
            
            # SonuÃƒÂ§: ertesi gÃƒÂ¼n 04:00'te, dÃƒÂ¼nkÃƒÂ¼ tarihe gÃƒÂ¶re
            if (STATE.get("last_res_date") != today) and _time_reached_tr(RESULTS_HOUR, RESULTS_MINUTE):
                res_date = _yesterday_str_tr(now_tr)  # her zaman DÃƒÅ“N
                log(f"SERVICE: SonuÃƒÂ§ zamanÃ„Â± geldi (dÃƒÂ¼n={res_date}) Ã¢â€ â€™ rapor hazÃ„Â±rlanÃ„Â±yorÃ¢â‚¬Â¦")
                report_results(res_date)
                STATE["last_res_date"] = today  # bugÃƒÂ¼nÃƒÂ¼ iÃ…Å¸aretle, tekrarÃ„Â± engelle
                save_state(STATE)
                
        except Exception:
            tb = traceback.format_exc()
            log(tb)
            try:
                send_mail("Tahmin Botu | SERVICE Hata", tb)
            except Exception:
                pass
        
        # Ã„Â°nce adÃ„Â±mlÃ„Â± uyku: 20 saniye
        time.sleep(20)

# --- Ãƒâ€¡alÃ„Â±Ã…Å¸tÃ„Â±rÃ„Â±cÃ„Â± -------------------------------------------------------------
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
            # OnaylÃ„Â± politika: RESULTS her zaman DÃƒÅ“N'e bakar
            report_results(_yesterday_str_tr(tr_now))
        else:
            send_mail("Tahmin Botu | Bilgi", "AUTO/SERVICE dÃ„Â±Ã…Å¸Ã„Â± ÃƒÂ§alÃ„Â±Ã…Å¸tÃ„Â±rma. MODE=PREDICT veya MODE=RESULTS veya MODE=SERVICE bekleniyor.")
            
    except Exception:
        tb = traceback.format_exc(); log(tb)
        try:
            send_mail("Tahmin Botu | Hata", tb)
        except Exception:
            pass

if __name__ == "__main__":
    main()


# --- Dinlenme (Rest) Etkisi ---
def calculate_rest_effect(days_home, days_away):
    """
    Pozitif deÃ„Å¸er = avantaj, negatif = dezavantaj.
    Basit sezgisel:
      <2 gÃƒÂ¼n: -0.15   |   2-3 gÃƒÂ¼n: -0.10   |   4-6 gÃƒÂ¼n: 0.00   |   >6 gÃƒÂ¼n: +0.05
    """
    def f(d):
        try:
            d = float(d)
        except Exception:
            return 0.0
        if d < 2:
            return -0.15
        if d < 3:
            return -0.10
        if d > 6:
            return 0.05
        return 0.0
    return {"home": f(days_home), "away": f(days_away)}
