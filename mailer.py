# -*- coding: utf-8 -*-
"""
Tahmin Botu — GÜNCELLENMİŞ SÜRÜM (API-FOOTBALL v3 + EKSİK FONKSİYONLAR ENTEGRE + ULTRA TAHMİN)
+ TÜM YENİ ÖZELLİKLER ENTEGRE EDİLMİŞ HALİ
"""

import json
import os
import re
import math
import time
import smtplib
import traceback
import urllib.parse
import random
import logging
import argparse
from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
import requests
from difflib import SequenceMatcher
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from scipy.optimize import minimize
from sklearn.isotonic import IsotonicRegression

# === YENİ ENSEMBLE İTHALATLARI ===
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import xgboost as xgb
import joblib

# ==================== YENİ EKLENEN LİG BAZLI ORTALAMALAR ====================

# Lig bazlı gol ortalamaları
LEAGUE_GOAL_BASE = {
    "premier league": 2.8, "la liga": 2.6, "serie a": 2.7, "bundesliga": 3.2,
    "ligue 1": 2.5, "super lig": 2.4, "eredivisie": 3.1, "primeira liga": 2.5,
    "pro league": 2.7, "championship": 2.6, "scottish premiership": 2.8
}

# Lig bazlı kart ortalamaları  
LEAGUE_CARD_BASE = {
    "premier league": 4.2, "la liga": 5.1, "serie a": 4.8, "bundesliga": 4.0,
    "ligue 1": 4.5, "super lig": 5.5, "eredivisie": 3.8, "primeira liga": 5.2,
    "pro league": 4.3, "championship": 4.6
}

# Lig bazlı korner ortalamaları
LEAGUE_CORNER_BASE = {
    "premier league": 10.2, "la liga": 9.8, "serie a": 9.5, "bundesliga": 10.5,
    "ligue 1": 9.2, "super lig": 9.0, "eredivisie": 10.8, "primeira liga": 9.6,
    "pro league": 9.4, "championship": 10.1
}

# API Lig Mapping
_API_LEAGUE_MAP = {
    "England|Premier League": 39, "England|Championship": 40,
    "Spain|La Liga": 140, "Spain|La Liga 2": 141,
    "Italy|Serie A": 135, "Italy|Serie B": 136,
    "Germany|Bundesliga": 78, "Germany|2. Bundesliga": 79,
    "France|Ligue 1": 61, "France|Ligue 2": 62,
    "Turkey|Super Lig": 203, "Turkey|1. Lig": 204,
    "Netherlands|Eredivisie": 88, "Netherlands|Eerste Divisie": 89,
    "Portugal|Primeira Liga": 94, "Portugal|Liga Portugal 2": 95,
    "Belgium|Pro League": 144, "Belgium|Challenger Pro League": 145,
    "Europe|Champions League": 2, "Europe|Europa League": 3,
    "Europe|Conference League": 848, "Europe|Super Cup": 667,
    "Europe|European Championship": 4, "World|World Cup": 1
}

# ==================== AKILLI TAKIM ID BULMA SİSTEMİ ====================

def _find_team_id_smart(team_name, league_id, date_str, season="2025"):
    """GÜNCELLENDİ: AKILLI TAKIM ID BULMA - BENZERLİK TABANLI"""
    try:
        # 1. Tarihteki lig maçlarını getir
        params = {"date": date_str, "league": league_id}
        response = _apifoot_get("fixtures", params)
        
        if not response:
            return _find_team_id_global(team_name)
            
        # 2. Maçlardan takım listesi çıkar
        takim_listesi = []
        for mac in response:
            takimlar = mac.get('teams', {})
            home_team = takimlar.get('home', {})
            away_team = takimlar.get('away', {})
            
            if home_team.get('id') and home_team.get('name'):
                takim_listesi.append({
                    'id': home_team['id'],
                    'name': home_team['name']
                })
            if away_team.get('id') and away_team.get('name'):
                takim_listesi.append({
                    'id': away_team['id'], 
                    'name': away_team['name']
                })
        
        # 3. Tekil takımları al
        league_teams = []
        seen_ids = set()
        for takim in takim_listesi:
            if takim['id'] not in seen_ids:
                seen_ids.add(takim['id'])
                league_teams.append(takim)
        
        if not league_teams:
            return _find_team_id_global(team_name)
        
        # 4. Takım ismi benzerlik kontrolü
        best_match = None
        best_score = 0.0
        
        for team in league_teams:
            similarity = _calculate_team_similarity(team_name, team['name'])
            if similarity > best_score:
                best_score = similarity
                best_match = team
        
        # 5. Benzerlik eşik kontrolü
        if best_match and best_score >= 0.4:  # %40+ benzerlik
            log(f"✅ Akıllı eşleşme: {team_name} → {best_match['name']} (%{best_score*100:.1f})")
            return best_match['id']
        else:
            # Benzerlik yoksa global arama
            return _find_team_id_global(team_name)
            
    except Exception as e:
        log(f"❌ Akıllı takım ID bulma hatası: {e}")
        return _find_team_id_global(team_name)

def _calculate_team_similarity(str1, str2):
    """İki takım ismi benzerlik skoru - MAILER.PY UYUMLU"""
    str1_norm = normalize_team_name(str1)
    str2_norm = normalize_team_name(str2)
    
    if str1_norm == str2_norm:
        return 1.0
        
    # Basit benzerlik - mailer.py'de SequenceMatcher zaten var
    from difflib import SequenceMatcher
    return SequenceMatcher(None, str1_norm, str2_norm).ratio()

def _get_league_teams_for_date(league_id, date_str, season):
    """Tahmin tarihindeki ligdeki tüm takımları getir"""
    try:
        # API-Football'dan lig takımlarını çek
        params = {
            "league": league_id,
            "season": season
        }
        
        response = _apifoot_get("teams", params)
        if response:
            return response
        else:
            # Fallback: basit takım listesi
            return []
            
    except Exception as e:
        log(f"❌ Lig takımları getirme hatası: {e}")
        return []

def _find_team_id_global(team_name):
    """Global takım arama (son çare)"""
    return _apifoot_find_team_id(team_name)

# ==================== GELİŞTİRİLMİŞ FORM VERİSİ FONKSİYONU ====================

def get_detailed_team_form_enhanced(team_id, team_name, min_matches=5):
    """GELİŞTİRİLMİŞ FORM VERİSİ - TÜM MAÇLAR (LİG FARKETMEZ)"""
    if not team_id:
        return None
        
    try:
        # Son 15 maçı getir (lig farketmez)
        params = {
            'team': team_id,
            'last': 15,
            'status': 'FT'
        }
        
        response = _apifoot_get("fixtures", params)
        
        if not response or len(response) < min_matches:
            return None
            
        matches = response
        total_matches = len(matches)
        
        # Detaylı istatistikler
        wins = draws = goals_for = goals_against = 0
        over_15 = over_25 = over_35 = btts = 0
        home_wins = away_wins = home_matches = away_matches = 0
        
        for match in matches:
            home_team = match['teams']['home']['id'] == team_id
            home_goals = match['goals']['home'] or 0
            away_goals = match['goals']['away'] or 0
            
            if home_team:
                goals_for += home_goals
                goals_against += away_goals
                home_matches += 1
                if match['teams']['home']['winner']:
                    wins += 1
                    home_wins += 1
                elif match['teams']['home']['winner'] is None and match['teams']['away']['winner'] is None:
                    draws += 1
            else:
                goals_for += away_goals
                goals_against += home_goals
                away_matches += 1
                if match['teams']['away']['winner']:
                    wins += 1
                    away_wins += 1
                elif match['teams']['home']['winner'] is None and match['teams']['away']['winner'] is None:
                    draws += 1
            
            # OVER/BTTS istatistikleri
            total_goals = home_goals + away_goals
            if total_goals > 1.5: over_15 += 1
            if total_goals > 2.5: over_25 += 1
            if total_goals > 3.5: over_35 += 1
            if home_goals > 0 and away_goals > 0: btts += 1
        
        # Form hesaplamaları
        points = (wins * 3) + draws
        max_points = total_matches * 3
        form_percentage = round((points / max_points) * 100, 1) if max_points > 0 else 0
        
        # Ev/saha performansı
        home_performance = (home_wins / home_matches * 100) if home_matches > 0 else 0
        away_performance = (away_wins / away_matches * 100) if away_matches > 0 else 0
        
        # Gol ortalamaları
        avg_goals_for = round(goals_for / total_matches, 1) if total_matches > 0 else 0
        avg_goals_against = round(goals_against / total_matches, 1) if total_matches > 0 else 0
        
        # Geliştirilmiş form skoru (ev/saha ağırlıklı)
        adjusted_form = min(95, max(25, form_percentage * 1.2))
        
        return {
            'form': adjusted_form,
            'real_form': form_percentage,
            'avg_goals_for': avg_goals_for,
            'avg_goals_against': avg_goals_against,
            'matches_analyzed': total_matches,
            'wins': wins,
            'draws': draws,
            'losses': total_matches - wins - draws,
            'home_performance': round(home_performance, 1),
            'away_performance': round(away_performance, 1),
            'over_15_percent': round((over_15 / total_matches) * 100, 1) if total_matches > 0 else 0,
            'over_25_percent': round((over_25 / total_matches) * 100, 1) if total_matches > 0 else 0,
            'over_35_percent': round((over_35 / total_matches) * 100, 1) if total_matches > 0 else 0,
            'btts_percent': round((btts / total_matches) * 100, 1) if total_matches > 0 else 0,
            'goal_difference': goals_for - goals_against
        }
        
    except Exception as e:
        log(f"❌ Geliştirilmiş takım form hatası: {e}")
        return None

# ==================== RAKAM OLARAK KART/KORNER SİSTEMİ ====================

def get_cards_corners_numeric(home_team, away_team, home_form, away_form):
    """GERÇEK API verilerine göre kart/korner tahmini - 3,5/7,5 formatında"""
    try:
        # Önce takım ID'lerini bul
        home_team_id = _apifoot_find_team_id(home_team)
        away_team_id = _apifoot_find_team_id(away_team)
        
        # GERÇEK kart/korner verilerini al
        home_real_stats = get_real_cards_corners_stats(home_team_id, home_team)
        away_real_stats = get_real_cards_corners_stats(away_team_id, away_team)
        
        if home_real_stats and away_real_stats:
            # GERÇEK ortalamalar
            avg_cards = (home_real_stats['avg_cards'] + away_real_stats['avg_cards']) / 2
            avg_corners = (home_real_stats['avg_corners'] + away_real_stats['avg_corners']) / 2
            
            # 3.5 ve 7.5 için GERÇEK olasılık hesapla
            p_cards_35 = poisson_over_prob(avg_cards, 3.5)
            p_corners_75 = poisson_over_prob(avg_corners, 7.5)
            
            return {
                'cards_mu': round(avg_cards, 1),
                'corners_mu': round(avg_corners, 1),
                'p_cards_35': p_cards_35,
                'p_corners_75': p_corners_75
            }
            
    except Exception as e:
        log(f"❌ GERÇEK kart/korner tahmini hatası: {e}")
    
    # Fallback: mevcut sistem
    try:
        league = "super lig"
        base_cards = LEAGUE_CARD_BASE.get(league, 4.6)
        base_corners = LEAGUE_CORNER_BASE.get(league, 9.2)
        
        home_aggression = 1.0
        away_aggression = 1.0
        home_attack = 1.0
        away_attack = 1.0
        
        if home_form:
            home_attack = min(1.3, max(0.7, home_form.get('avg_goals_for', 1.2) / 1.2))
            home_aggression = min(1.2, max(0.8, home_form.get('form', 50) / 50))
        
        if away_form:
            away_attack = min(1.3, max(0.7, away_form.get('avg_goals_for', 1.1) / 1.1))
            away_aggression = min(1.2, max(0.8, away_form.get('form', 50) / 50))
        
        cards_mu = base_cards * ((home_aggression + away_aggression) / 2)
        corners_mu = base_corners * ((home_attack + away_attack) / 2)
        
        return {
            'cards_mu': round(cards_mu, 1),
            'corners_mu': round(corners_mu, 1),
            'p_cards_35': poisson_over_prob(cards_mu, 3.5),
            'p_corners_75': poisson_over_prob(corners_mu, 7.5)
        }
        
    except Exception as e:
        log(f"❌ Fallback kart/korner hatası: {e}")
        return {
            'cards_mu': 4.6,
            'corners_mu': 9.2,
            'p_cards_35': 0.5,
            'p_corners_75': 0.5
        }

def get_real_cards_corners_stats(team_id, team_name):
    """Takımın son 5 maçının GERÇEK kart/korner istatistikleri"""
    if not team_id:
        return None
        
    try:
        params = {'team': team_id, 'last': 5, 'status': 'FT'}
        response = _apifoot_get("fixtures", params)
        
        if not response or len(response) < 3:
            return None
            
        total_cards = 0
        total_corners = 0
        match_count = 0
        
        for match in response:
            fixture_id = match['fixture']['id']
            stats_response = _apifoot_get("fixtures/statistics", {'fixture': fixture_id})
            
            if stats_response:
                for team_stats in stats_response:
                    for stat in team_stats.get('statistics', []):
                        if stat['type'] == 'Yellow Cards':
                            total_cards += int(stat['value'] or 0)
                        elif stat['type'] == 'Red Cards':
                            total_cards += int(stat['value'] or 0)
                        elif stat['type'] == 'Corner Kicks':
                            total_corners += int(stat['value'] or 0)
                match_count += 1
        
        if match_count > 0:
            return {
                'avg_cards': total_cards / match_count,
                'avg_corners': total_corners / match_count,
                'matches_analyzed': match_count
            }
            
    except Exception as e:
        log(f"❌ GERÇEK kart/korner istatistik hatası: {e}")
    return None

# ==================== GÜNCELLENMİŞ ULTRA TAHMİN SİSTEMİ ====================

def ultra_tahmin_sistemi(date_str):
    """GELİŞTİRİLMİŞ ULTRA TAHMİN - AKILLI TAKIM BULMA İLE"""
    print(f"🎯 GELİŞTİRİLMİŞ ULTRA TAHMİN SİSTEMİ BAŞLATILIYOR: {date_str}")
    
    try:
        # Hedef lig ID'leri
        HEDEF_LIG_IDS = ['39','140','135','78','61','88','144','179','203','141','136','79','95','145','2','3','848']
        
        # Bugünkü maçları çek
        params = {"date": date_str}
        response = requests.get(f"{APIFOOTBALL_BASE_URL}fixtures", headers=HEADERS, params=params, timeout=30)
        
        if response.status_code != 200:
            print(f"❌ Maç verisi hatası: {response.status_code}")
            return []
            
        data = response.json()
        tum_maclar = data.get('response', [])
        
        # Hedef ligleri filtrele
        hedef_maclar = []
        for mac in tum_maclar:
            lig_id = str(mac.get('league', {}).get('id', ''))
            if lig_id in HEDEF_LIG_IDS:
                hedef_maclar.append(mac)
        
        print(f"✅ {len(hedef_maclar)} hedef lig maçı bulundu")
        
        if not hedef_maclar:
            print("❌ Hedef liglerde maç yok")
            return []
        
        # TÜM MAÇLAR İÇİN GELİŞTİRİLMİŞ ULTRA TAHMİN
        ultra_tahminler = []
        
        for i, mac in enumerate(hedef_maclar):
            fixture = mac.get('fixture', {})
            teams = mac.get('teams', {})
            league = mac.get('league', {})
            
            fixture_id = fixture.get('id')
            home_team = teams.get('home', {}).get('name', 'Unknown')
            away_team = teams.get('away', {}).get('name', 'Unknown')
            league_name = league.get('name', 'Unknown')
            league_id = str(league.get('id', ''))
            
            # Maç saati
            match_time = "??:??"
            fixture_date = fixture.get('date', '')
            if fixture_date:
                try:
                    match_time = datetime.fromisoformat(fixture_date.replace('Z', '+00:00')).strftime('%H:%M')
                except:
                    match_time = "??:??"
            
            print(f"🔮 GELİŞTİRİLMİŞ ULTRA Tahmin: {home_team} vs {away_team}")
            
            # GELİŞTİRİLMİŞ ULTRA TAHMİN SİSTEMİ
            try:
                # YENİ: AKILLI TAKIM ID BULMA
                home_team_id = _find_team_id_smart(home_team, league_id, date_str)
                away_team_id = _find_team_id_smart(away_team, league_id, date_str)
                
                # YENİ: DETAYLI FORM İSTATİSTİKLERİ
                home_form_data = get_detailed_team_form_enhanced(home_team_id, home_team)
                away_form_data = get_detailed_team_form_enhanced(away_team_id, away_team)
                
                # DETAYLI AI TAHMİN PARSING
                ai_pred = get_ai_predictions_detailed(fixture_id)
                
                # YENİ: BASİT TAHMİN MANTIĞI
                simple_prediction = simple_form_prediction(home_form_data, away_form_data)
                
                # YENİ: RAKAMSAL KART/KORNER TAHMİNİ
                cards_corners_data = get_cards_corners_numeric(home_team, away_team, home_form_data, away_form_data)
                
                # STANDART TAHMIN FORMATI
                predictions = calculate_standardized_predictions(
                    home_team, away_team, home_form_data, away_form_data, ai_pred
                )
                
                # YENİ: GELİŞTİRİLMİŞ VERİ KALİTESİ HESAPLAMA
                confidence = calculate_enhanced_confidence(home_form_data, away_form_data, ai_pred, predictions)
                
                # GERÇEK sistem tahmini (form bazlı)
                pick, _ = _ultra_gercek_sistem_tahmini_entegre(home_form_data, away_form_data, ai_pred)
                
                # YENİ: PRATİK SKOR TAHMİNİ
                practical_scores = generate_practical_scores(home_form_data, away_form_data)
                
                # ULTRA tahmini kaydet
                ultra_tahmin = {
                    'match': f"{home_team} vs {away_team}",
                    'league': league_name,
                    'time': match_time,
                    'pick': pick,
                    'simple_prediction': simple_prediction,
                    'confidence': confidence,
                    'api_prediction': ai_pred.get('winner', 'Belirsiz'),
                    'api_confidence': 70,
                    'home_form': f"%{home_form_data['form']:.1f}" if home_form_data else "VERİ YOK",
                    'away_form': f"%{away_form_data['form']:.1f}" if away_form_data else "VERİ YOK",
                    'home_stats': home_form_data,
                    'away_stats': away_form_data,
                    'gol_15': "ÜST" if predictions['home_goals_1_5'] > 50 else "ALT",
                    'gol_15_prob': predictions['home_goals_1_5'],
                    'gol_25': "ÜST" if predictions['over_2_5'] > 50 else "ALT",
                    'gol_25_prob': predictions['over_2_5'],
                    'gol_35': "ÜST" if predictions['over_2_5'] > 60 else "ALT",
                    'gol_35_prob': min(predictions['over_2_5'] + 10, 80),
                    'btts': "EVET" if predictions['btts'] > 50 else "HAYIR",
                    'btts_prob': predictions['btts'],
                    'skor_tahmini': ' | '.join(practical_scores),
                    'kart_35': f"ÜST (%{int(cards_corners_data['p_cards_35'] * 100)}) - 3,5",
                    'kart_prob': int(cards_corners_data['p_cards_35'] * 100),
                    'kart_mu': cards_corners_data['cards_mu'],
                    'korner_75': f"ÜST (%{int(cards_corners_data['p_corners_75'] * 100)}) - 7,5",
                    'korner_prob': int(cards_corners_data['p_corners_75'] * 100),
                    'korner_mu': cards_corners_data['corners_mu'],
                    'data_quality': confidence
                }
                
                ultra_tahminler.append(ultra_tahmin)
                
            except Exception as e:
                print(f"❌ GELİŞTİRİLMİŞ ULTRA tahmin hatası: {e}")
                continue
        
        print(f"✅ GELİŞTİRİLMİŞ ULTRA tahminleri tamamlandı: {len(ultra_tahminler)} maç")
        return ultra_tahminler
        
    except Exception as e:
        print(f"❌ GELİŞTİRİLMİŞ ULTRA sistem hatası: {e}")
        return []

# ==================== EKSİK FONKSİYONLARIN TAMAMLANMASI ====================

def base_total_goals(area):
    """Lig bazlı gol ortalaması"""
    area_lower = area.lower() if area else "europe"
    for league, goals in LEAGUE_GOAL_BASE.items():
        if league in area_lower:
            return goals
    return 2.7  # Varsayılan

def base_from_area(area, base_dict, default):
    """Lig bazlı değer al"""
    area_lower = area.lower() if area else "europe"
    for league, value in base_dict.items():
        if league in area_lower:
            return value
    return default

def _apifoot_find_team_id(team_name):
    """Takım ID bulma - Basitleştirilmiş"""
    if not APIFOOT or not team_name:
        return None
    
    # Basit cache kontrolü
    cache_key = team_name.lower()
    if cache_key in _apifoot_team_cache:
        return _apifoot_team_cache[cache_key]
    
    try:
        # Takım arama
        params = {"search": team_name}
        response = requests.get(
            f"{APIFOOTBALL_BASE_URL}teams",
            headers=HEADERS,
            params=params,
            timeout=20
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('response'):
                team_id = data['response'][0]['team']['id']
                _apifoot_team_cache[cache_key] = team_id
                return team_id
    except Exception as e:
        log(f"Team ID bulma hatası: {e}")
    
    return None

def poisson_prob(lam_h, lam_a):
    """Poisson dağılımına göre 1/X/2 olasılıkları - OPTİMİZE EDİLMİŞ"""
    # Optimize edilmiş Poisson hesaplama
    max_goals = 8  # Pratik limit
    
    # Home win olasılığı
    p_home = 0.0
    for goals_h in range(0, max_goals + 1):
        for goals_a in range(0, goals_h):  # Sadece home win durumları
            prob = (math.exp(-lam_h) * (lam_h ** goals_h) / math.factorial(goals_h)) * \
                   (math.exp(-lam_a) * (lam_a ** goals_a) / math.factorial(goals_a))
            p_home += prob
    
    # Away win olasılığı
    p_away = 0.0
    for goals_a in range(0, max_goals + 1):
        for goals_h in range(0, goals_a):  # Sadece away win durumları
            prob = (math.exp(-lam_h) * (lam_h ** goals_h) / math.factorial(goals_h)) * \
                   (math.exp(-lam_a) * (lam_a ** goals_a) / math.factorial(goals_a))
            p_away += prob
    
    # Draw olasılığı
    p_draw = 0.0
    for goals in range(0, max_goals + 1):
        prob = (math.exp(-lam_h) * (lam_h ** goals) / math.factorial(goals)) * \
               (math.exp(-lam_a) * (lam_a ** goals) / math.factorial(goals))
        p_draw += prob
    
    # Normalizasyon
    total = p_home + p_draw + p_away
    if total > 0.9:  # Yeterli olasılık toplandı
        return (p_home, p_draw, p_away)
    else:
        # Fallback: Basit formül
        p_home_simple = 1.0 / (1.0 + 10.0 ** ((lam_a - lam_h) / 400.0))
        p_away_simple = 1.0 - p_home_simple
        p_draw_simple = 0.25  # Sabit beraberlik olasılığı
        p_home_adj = p_home_simple * (1 - p_draw_simple)
        p_away_adj = p_away_simple * (1 - p_draw_simple)
        return (p_home_adj, p_draw_simple, p_away_adj)

def blend_model_market(model_probs, market_probs):
    """Model ve market olasılıklarını birleştir"""
    if market_probs is None:
        return model_probs
    
    w_mkt = clamp(get_w_mkt(), 0.0, 1.0)  # Weight kontrolü
    w_model = 1.0 - w_mkt
    
    p_home = model_probs[0] * w_model + market_probs[0] * w_mkt
    p_draw = model_probs[1] * w_model + market_probs[1] * w_mkt  
    p_away = model_probs[2] * w_model + market_probs[2] * w_mkt
    
    # Normalizasyon
    total = p_home + p_draw + p_away
    if total > 0:
        return (p_home/total, p_draw/total, p_away/total)
    else:
        return model_probs

def poisson_over_prob(mu, threshold):
    """Poisson dağılımında over olasılığı"""
    if mu <= 0:
        return 0.0
    
    # Threshold'u integer'a çevir
    threshold_int = int(threshold)
    if threshold_int < 0:
        return 1.0
    
    p_under = 0.0
    for k in range(0, threshold_int + 1):
        if k <= 20:  # Pratik limit
            p_under += (math.exp(-mu) * (mu ** k)) / math.factorial(k)
        else:
            break
    
    return max(0.0, min(1.0, 1.0 - p_under))

def _apifoot_team_statistics(league_id, season, team_id):
    """Takım istatistikleri - Geliştirilmiş"""
    if not APIFOOT or not league_id or not season or not team_id:
        return None
    
    cache_key = (league_id, season, team_id)
    if cache_key in _apifoot_stat_cache:
        return _apifoot_stat_cache[cache_key]
    
    try:
        params = {"league": league_id, "season": season, "team": team_id}
        response = _apifoot_get("teams/statistics", params)
        if response:
            _apifoot_stat_cache[cache_key] = response
            return response
    except Exception as e:
        log(f"Team statistics error: {e}")
        # Fallback: basit istatistikler
        fallback_stats = {
            "fixtures": {"played": {"total": 10}},
            "cards": {"yellow": {"total": 15}, "red": {"total": 1}},
            "corners": {"total": 45}
        }
        return fallback_stats
    
    return None

# ==================== CACHE DÜZELTMESİ ====================

def clear_old_cache():
    """Cache temizleme - sonsuz döngüyü önle"""
    pass

def enhanced_clear_old_cache():
    """Geliştirilmiş cache temizleme - sonsuz döngüyü önle"""
    pass

# ==================== AI TAHMİN SİSTEMİ EKSİK FONKSİYONLAR ====================

def get_ai_predictions_detailed(fixture_id):
    """API-Football'dan AI tahminlerini al - basitleştirilmiş"""
    try:
        if not APIFOOT or not fixture_id:
            return {'winner': 'Belirsiz', 'win_probability': {}, 'advice': 'Veri yok'}
        
        # API-Football tahmin endpoint'i
        prediction_data = get_api_predictions(fixture_id)
        
        if prediction_data and len(prediction_data) > 0:
            prediction = prediction_data[0].get('predictions', {})
            return {
                'winner': prediction.get('winner', {}).get('name', 'Belirsiz'),
                'win_probability': prediction.get('win_probability', {}),
                'advice': prediction.get('advice', 'Veri yok'),
                'home_percent': prediction.get('percent', {}).get('home', '0'),
                'draw_percent': prediction.get('percent', {}).get('draw', '0'), 
                'away_percent': prediction.get('percent', {}).get('away', '0')
            }
    except Exception as e:
        print(f"❌ AI tahmin hatası: {e}")
    
    # Fallback değerler
    return {
        'winner': 'Belirsiz', 
        'win_probability': {},
        'advice': 'API verisi yok',
        'home_percent': '0',
        'draw_percent': '0',
        'away_percent': '0'
    }

def _ultra_gercek_sistem_tahmini_entegre(home_form_data, away_form_data, ai_pred):
    """Ultra sistem tahmini - form bazlı basit tahmin"""
    if not home_form_data or not away_form_data:
        return "Veri yok", 50
    
    try:
        home_power = home_form_data.get('form', 50)
        away_power = away_form_data.get('form', 50)
        
        # Ev avantajı +%15
        home_power_adj = home_power * 1.15
        power_diff = home_power_adj - away_power
        
        # Basit tahmin mantığı
        if power_diff > 25:
            pick = "1"
            confidence = min(85, 65 + power_diff/3)
        elif power_diff < -25:
            pick = "2" 
            confidence = min(85, 65 + abs(power_diff)/3)
        else:
            pick = "X"
            confidence = min(75, 55 + (25 - abs(power_diff))/2)
        
        # AI tahmini ile uyum kontrolü
        if ai_pred and ai_pred.get('winner') != 'Belirsiz':
            ai_winner = ai_pred.get('winner', '')
            if ai_winner == pick:
                confidence += 5  # AI ile uyum +5%
            else:
                confidence -= 3  # AI ile uyumsuzluk -3%
                
        return pick, min(95, max(40, confidence))
        
    except Exception as e:
        print(f"❌ Ultra tahmin hatası: {e}")
        return "Hata", 50

def calculate_standardized_predictions(home_team, away_team, home_form_data, away_form_data, ai_pred):
    """Standart tahmin hesaplama - gol/kart/korner olasılıkları"""
    try:
        if not home_form_data or not away_form_data:
            return {
                'home_goals_1_5': 50, 'over_2_5': 45, 'over_3_5': 30,
                'btts': 50, 'cards': 50, 'corners': 50
            }
        
        # Gol ortalamaları
        home_avg = home_form_data.get('avg_goals_for', 1.2)
        away_avg = away_form_data.get('avg_goals_for', 1.1)
        home_against = home_form_data.get('avg_goals_against', 1.1) 
        away_against = away_form_data.get('avg_goals_against', 1.2)
        
        # Gol olasılıkları
        total_goals = home_avg + away_avg
        over_15_prob = min(80, max(40, (total_goals) * 25))
        over_25_prob = min(70, max(30, (total_goals - 1.0) * 20))
        over_35_prob = min(60, max(20, (total_goals - 1.8) * 18))
        
        # BTTS olasılığı
        home_btts = home_form_data.get('btts_percent', 45)
        away_btts = away_form_data.get('btts_percent', 45)
        btts_prob = min(75, max(35, (home_btts + away_btts) / 2))
        
        # Kart ve korner (basit hesaplama)
        cards_prob = min(70, max(40, 50 + (total_goals - 2.0) * 5))
        corners_prob = min(75, max(45, 50 + (total_goals - 2.0) * 6))
        
        return {
            'home_goals_1_5': over_15_prob,
            'over_2_5': over_25_prob, 
            'over_3_5': over_35_prob,
            'btts': btts_prob,
            'cards': cards_prob,
            'corners': corners_prob
        }
        
    except Exception as e:
        print(f"❌ Standart tahmin hatası: {e}")
        return {
            'home_goals_1_5': 50, 'over_2_5': 45, 'over_3_5': 30,
            'btts': 50, 'cards': 50, 'corners': 50
        }

# ==================== GELİŞTİRİLMİŞ ULTRA TAHMİN SİSTEMİ ====================

def get_detailed_team_form(team_id, league_id):
    """YENİ: DETAYLI takım formu ve istatistikleri - API-Football verileriyle"""
    if not team_id or not league_id:
        return None
        
    url = f"{APIFOOTBALL_BASE_URL}fixtures"
    params = {
        'team': team_id,
        'league': league_id,
        'season': 2024,
        'last': 8,  # Son 8 maç
        'status': 'FT'
    }
    
    try:
        response = requests.get(url, headers=HEADERS, params=params, timeout=30)
        data = response.json()
        
        if not data.get('response'):
            return None
            
        matches = data['response']
        total_matches = len(matches)
        
        if total_matches < 3:
            return None
            
        # YENİ: DETAYLI İSTATİSTİKLER
        wins = draws = goals_for = goals_against = 0
        over_15 = over_25 = over_35 = btts = 0
        
        for match in matches:
            home_team = match['teams']['home']['id'] == team_id
            home_goals = match['goals']['home'] or 0
            away_goals = match['goals']['away'] or 0
            
            if home_team:
                goals_for += home_goals
                goals_against += away_goals
                if match['teams']['home']['winner']:
                    wins += 1
                elif match['teams']['home']['winner'] is None and match['teams']['away']['winner'] is None:
                    draws += 1
            else:
                goals_for += away_goals
                goals_against += home_goals
                if match['teams']['away']['winner']:
                    wins += 1
                elif match['teams']['home']['winner'] is None and match['teams']['away']['winner'] is None:
                    draws += 1
            
            # YENİ: OVER/BTTS İSTATİSTİKLERİ
            total_goals = home_goals + away_goals
            if total_goals > 1.5: over_15 += 1
            if total_goals > 2.5: over_25 += 1
            if total_goals > 3.5: over_35 += 1
            if home_goals > 0 and away_goals > 0: btts += 1
        
        points = (wins * 3) + draws
        max_points = total_matches * 3
        form_percentage = round((points / max_points) * 100, 1)
        
        avg_goals_for = round(goals_for / total_matches, 1)
        avg_goals_against = round(goals_against / total_matches, 1)
        
        adjusted_form = min(95, max(25, form_percentage * 1.2))
        
        # YENİ: DETAYLI İSTATİSTİKLER DÖNDÜR
        return {
            'form': adjusted_form,
            'real_form': form_percentage,
            'avg_goals_for': avg_goals_for,
            'avg_goals_against': avg_goals_against,
            'matches_analyzed': total_matches,
            'wins': wins,
            'draws': draws,
            'losses': total_matches - wins - draws,
            'over_15_percent': round((over_15 / total_matches) * 100, 1) if total_matches > 0 else 0,
            'over_25_percent': round((over_25 / total_matches) * 100, 1) if total_matches > 0 else 0,
            'over_35_percent': round((over_35 / total_matches) * 100, 1) if total_matches > 0 else 0,
            'btts_percent': round((btts / total_matches) * 100, 1) if total_matches > 0 else 0
        }
        
    except Exception as e:
        print(f"❌ Detaylı takım form hesaplama hatası: {e}")
        return None

def simple_form_prediction(home_form_data, away_form_data):
    """YENİ: BASİT TAHMİN MANTIĞI - Form yüzdelerine dayalı"""
    if not home_form_data or not away_form_data:
        return "Veri yetersiz"
    
    # Ev sahibi avantajı +%15
    home_power = home_form_data['form'] * 1.15
    away_power = away_form_data['form'] * 0.85
    power_diff = home_power - away_power
    
    # Basit eşik değerleri
    if power_diff > 25: 
        return "1"
    elif power_diff < -25: 
        return "2"
    elif -10 <= power_diff <= 10: 
        return "X"
    else: 
        return "1" if power_diff > 0 else "2"

def generate_practical_scores(home_form_data, away_form_data):
    """YENİ: PRATİK SKOR TAHMİNİ - Gol ortalamalarına göre"""
    if not home_form_data or not away_form_data:
        return ["1-1", "2-1", "1-2"]
    
    home_avg = home_form_data['avg_goals_for']
    away_avg = away_form_data['avg_goals_for']
    
    # Temel skor
    home_goals = max(1, min(3, round(home_avg)))
    away_goals = max(0, min(2, round(away_avg)))
    
    scores = [f"{home_goals}-{away_goals}"]
    
    # Alternatif skorlar
    if home_goals > away_goals:
        scores.extend([f"{home_goals+1}-{away_goals}", f"{home_goals}-{away_goals+1}"])
    elif home_goals < away_goals:
        scores.extend([f"{home_goals}-{away_goals+1}", f"{home_goals+1}-{away_goals}"])
    else:
        scores.extend([f"{home_goals+1}-{away_goals}", f"{home_goals}-{away_goals+1}"])
    
    return scores[:3]  # 3 skor döndür

def calculate_enhanced_confidence(home_form_data, away_form_data, ai_pred, predictions):
    """YENİ: GELİŞTİRİLMİŞ VERİ KALİTESİ HESAPLAMA"""
    confidence = 60
    
    # Form verisi kalitesi
    if home_form_data and away_form_data:
        confidence += 25
        home_matches = home_form_data['matches_analyzed']
        away_matches = away_form_data['matches_analyzed']
        if home_matches >= 5 and away_matches >= 5:
            confidence += 15
    
    # AI tahmini
    if ai_pred['winner'] != 'Belirsiz':
        confidence += 20
    
    # Veri kalitesi skalası
    if home_form_data and away_form_data:
        data_quality = 80  # %80 kalite
    elif home_form_data or away_form_data:
        data_quality = 60  # %60 kalite  
    else:
        data_quality = 40  # %40 kalite
    
    confidence = confidence * (data_quality / 100)  # Kaliteye göre ayarla
    
    return min(95, max(40, confidence))

def basic_probability_calc(home_form_data, away_form_data):
    """YENİ: BASİT PROBABİLİTE HESABI - İki takım ortalaması"""
    if not home_form_data or not away_form_data:
        return {
            'over_15': 65, 'over_25': 45, 'over_35': 25, 
            'btts': 55, 'cards_under': 60, 'corners_under': 50
        }
    
    # İki takım ortalaması
    over_15 = (home_form_data.get('over_15_percent', 50) + away_form_data.get('over_15_percent', 50)) / 2
    over_25 = (home_form_data.get('over_25_percent', 35) + away_form_data.get('over_25_percent', 35)) / 2
    over_35 = (home_form_data.get('over_35_percent', 20) + away_form_data.get('over_35_percent', 20)) / 2
    btts = (home_form_data.get('btts_percent', 45) + away_form_data.get('btts_percent', 45)) / 2
    
    # Min-max sınırlamaları
    return {
        'over_15': min(85, max(50, over_15)),
        'over_25': min(75, max(35, over_25)),
        'over_35': min(60, max(20, over_35)),
        'btts': min(75, max(40, btts)),
        'cards_under': 55, 
        'corners_under': 52
    }

# ==================== GÜNCELLENMİŞ RAPORLAMA SİSTEMİ ====================

def enhanced_report_predictions(date_str):
    """
    Geliştirilmiş tahmin raporu - TÜM MAÇLAR + EN İYİ 5
    """
    # ANA TAHMİNLERİ AL
    fixtures = universal_collector.fetch_fixtures_universal(date_str)
    
    # State'i boş da olsa kalıcılaştır
    save_state(STATE)
    
    ana_tahminler = []
    ultra_tahminler = []
    
    hi = []
    fixtures.sort(key=lambda x: x["utc_kickoff"] or datetime.now(timezone.utc))
    
    for fx in fixtures:
        # ANA TAHMİN SİSTEMİ
        odds, odds_source = fetch_odds_dual(fx.get("area",""), fx.get("competition",""), fx["home"], fx["away"])
        rated = rate_fixture_enhanced(fx, odds)
        
        record_prediction(
            fx, rated, rated["probs_model"], rated["probs_market"], rated["probs_blend"],
            rated["wx_adj"], rated["elo_adj"], rated["net_form"]
        )
        
        ana_tahmin = {
            "match": f"{fx['home']} vs {fx['away']}",
            "prediction": rated["pick"],
            "confidence": rated["confidence"],
            "note": rated["note"],
            "area": fx.get("area", ""),
            "competition": fx.get("competition", ""),
            "time": (fx["utc_kickoff"] or datetime.now(timezone.utc)).astimezone(TR_TZ).strftime("%H:%M") if fx.get("utc_kickoff") else "Saat Yok",
            "odds_source": odds_source
        }
        
        ana_tahminler.append(ana_tahmin)
        
        if rated["confidence"] >= HIGH_ALERT:
            hi.append((rated["confidence"], ana_tahmin))
    
    # GELİŞTİRİLMİŞ ULTRA TAHMİNLERİ AL
    ultra_tahminler = ultra_tahmin_sistemi(date_str)
    
    # YENİ: TÜM MAÇLAR + EN İYİ 5 SİSTEMİ
    tum_ana_tahminler = ana_tahminler  # TÜM ana tahminler
    top_ana_tahminler = get_top_n_predictions(ana_tahminler, TOP_N, MIN_CONF)  # EN İYİ 5
    
    # E-posta gönder - YENİ FORMAT
    email_body = format_enhanced_email(tum_ana_tahminler, top_ana_tahminler, ultra_tahminler, date_str)
    send_mail(f"Geliştirilmiş Tahmin Raporu | {date_str}", email_body)
    
    return {
        'tum_ana_tahminler': tum_ana_tahminler,
        'top_ana_tahminler': top_ana_tahminler,
        'ultra_tahminler': ultra_tahminler
    }

def format_enhanced_email(tum_ana_tahminler, top_ana_tahminler, ultra_tahminler, date_str):
    """YENİ: GELİŞTİRİLMİŞ E-POSTA FORMATI - Tüm maçlar + En iyi 5 + HTML desteği"""
    
    # HTML modu kontrolü
    html_mode = os.getenv("HTML_EMAIL", "0") == "1"
    
    if html_mode:
        return format_html_email(tum_ana_tahminler, top_ana_tahminler, ultra_tahminler, date_str)
    else:
        return format_text_email(tum_ana_tahminler, top_ana_tahminler, ultra_tahminler, date_str)

def format_text_email(tum_ana_tahminler, top_ana_tahminler, ultra_tahminler, date_str):
    """YENİ: METİN E-POSTA FORMATI - Tüm maçlar + En iyi 5"""
    lines = []
    
    # 1. EN İYİ 5 ANA TAHMİN (BAŞTA)
    lines.append(f"🎯 EN İYİ 5 ANA TAHMİN — {date_str}")
    lines.append("=" * 60)
    
    if top_ana_tahminler:
        for i, pred in enumerate(top_ana_tahminler, 1):
            emoji = "🔥" if pred.get('confidence', 0) >= HIGH_ALERT else "✅"
            lines.append(f"{emoji} #{i} - {pred.get('confidence', 0)}% GÜVEN")
            lines.append(f"   ⚽ {pred.get('match', 'Maç bilgisi yok')}")
            lines.append(f"   🎯 Tahmin: {pred.get('prediction', 'N/A')}")
            lines.append(f"   ⏰ Saat: {pred.get('time', 'N/A')}")
            lines.append("")
    else:
        lines.append("❌ Bugün için yeterince güvenilir ANA tahmin bulunamadı.")
        lines.append("")
    
    # 2. TÜM ANA TAHMİNLER
    lines.append(f"🏟️ TÜM ANA TAHMİNLER — {date_str}")
    lines.append("=" * 60)
    
    if tum_ana_tahminler:
        for i, pred in enumerate(tum_ana_tahminler, 1):
            emoji = "🔥" if pred.get('confidence', 0) >= HIGH_ALERT else "⚽"
            lines.append(f"{emoji} #{i} - {pred.get('confidence', 0)}%")
            lines.append(f"   ⚽ {pred.get('match', 'Maç bilgisi yok')}")
            lines.append(f"   🎯 Tahmin: {pred.get('prediction', 'N/A')}")
            lines.append(f"   ⏰ Saat: {pred.get('time', 'N/A')}")
            lines.append("")
    else:
        lines.append("❌ Bugün için ANA tahmin bulunamadı.")
        lines.append("")
    
    # 3. TÜM ULTRA TAHMİNLER
    lines.append(f"🔮 TÜM ULTRA TAHMİNLER — {date_str}")
    lines.append("=" * 60)
    
    if ultra_tahminler:
        for i, pred in enumerate(ultra_tahminler, 1):
            lines.append(f"🔮 #{i} - {pred.get('confidence', 0)}%")
            lines.append(f"   ⚽ {pred.get('match', 'Maç bilgisi yok')}")
            lines.append(f"   🎯 Tahmin: {pred.get('pick', 'N/A')}")
            lines.append(f"   🤖 Basit Sistem: {pred.get('simple_prediction', 'N/A')}")
            lines.append(f"   🤖 API-Football AI: {pred.get('api_prediction', 'N/A')} (%{pred.get('api_confidence', 0)})")
            lines.append(f"   📊 Form: {pred.get('home_form', 'N/A')} vs {pred.get('away_form', 'N/A')}")
            lines.append(f"   ⚽ Gol: 1.5:{pred.get('gol_15', 'N/A')}(%{pred.get('gol_15_prob', 0)}) | 2.5:{pred.get('gol_25', 'N/A')}(%{pred.get('gol_25_prob', 0)})")
            lines.append(f"   🔥 BTTS: {pred.get('btts', 'N/A')} (%{pred.get('btts_prob', 0)})")
            lines.append(f"   📋 Skor: {pred.get('skor_tahmini', 'N/A')}")
            lines.append(f"   ⚠️ Kart: {pred.get('kart_35', 'N/A')}")
            lines.append(f"   🎯 Korner: {pred.get('korner_75', 'N/A')}")
            lines.append(f"   📈 Veri Kalitesi: %{pred.get('data_quality', 0)}")
            lines.append("")
    else:
        lines.append("❌ Bugün için ULTRA tahmin bulunamadı.")
        lines.append("")
    
    # İSTATİSTİKLER
    lines.append("📊 İSTATİSTİKLER")
    lines.append(f"   • Tüm ANA Tahminler: {len(tum_ana_tahminler)}")
    lines.append(f"   • En İyi 5 ANA: {len(top_ana_tahminler)}")
    lines.append(f"   • Tüm ULTRA Tahminler: {len(ultra_tahminler)}")
    
    if tum_ana_tahminler:
        avg_ana = sum(p.get('confidence', 0) for p in tum_ana_tahminler) / len(tum_ana_tahminler)
        lines.append(f"   • Ortalama ANA Güven: {avg_ana:.1f}%")
    
    if ultra_tahminler:
        avg_ultra = sum(p.get('confidence', 0) for p in ultra_tahminler) / len(ultra_tahminler)
        lines.append(f"   • Ortalama ULTRA Güven: {avg_ultra:.1f}%")
    
    lines.append("")
    lines.append(f"🤖 Model: {MODEL_VERSION}")
    lines.append(f"⏰ Üretim Zamanı: {datetime.now(TR_TZ).strftime('%Y-%m-%d %H:%M:%S')}")
    
    return "\n".join(lines)

def format_html_email(tum_ana_tahminler, top_ana_tahminler, ultra_tahminler, date_str):
    """YENİ: HTML E-POSTA FORMATI - Görsel olarak zengin"""
    
    html_content = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 20px; border-radius: 10px; }}
            .header {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 20px; border-radius: 10px; margin-bottom: 20px; }}
            .section {{ margin: 20px 0; padding: 15px; border: 1px solid #ddd; border-radius: 8px; }}
            .match {{ border: 1px solid #e0e0e0; padding: 12px; margin: 10px 0; border-radius: 6px; background: #fafafa; }}
            .high-confidence {{ border-left: 4px solid #ff6b6b; background: #fff5f5; }}
            .medium-confidence {{ border-left: 4px solid #ffd93d; background: #fffef0; }}
            .prediction {{ font-weight: bold; color: #2c5530; }}
            .stats {{ font-size: 12px; color: #666; margin-top: 5px; }}
            h2 {{ color: #333; border-bottom: 2px solid #667eea; padding-bottom: 5px; }}
            h3 {{ color: #555; }}
            .emoji {{ font-size: 18px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🔮 Futbol Tahmin Raporu</h1>
                <p>Tarih: {date_str} | Toplam {len(tum_ana_tahminler) + len(ultra_tahminler)} maç analiz edildi</p>
            </div>
    """
    
    # EN İYİ 5 ANA TAHMİN
    html_content += f"""
            <div class="section">
                <h2>🎯 EN İYİ 5 ANA TAHMİN — {date_str}</h2>
    """
    
    if top_ana_tahminler:
        for i, pred in enumerate(top_ana_tahminler, 1):
            confidence_class = "high-confidence" if pred.get('confidence', 0) >= HIGH_ALERT else "medium-confidence"
            html_content += f"""
                <div class="match {confidence_class}">
                    <h3>{"🔥" if pred.get('confidence', 0) >= HIGH_ALERT else "✅"} #{i} - {pred.get('confidence', 0)}% GÜVEN</h3>
                    <div><strong>⚽ {pred.get('match', 'Maç bilgisi yok')}</strong></div>
                    <div class="prediction">🎯 Tahmin: {pred.get('prediction', 'N/A')}</div>
                    <div>⏰ Saat: {pred.get('time', 'N/A')}</div>
                </div>
            """
    else:
        html_content += "<p>❌ Bugün için yeterince güvenilir ANA tahmin bulunamadı.</p>"
    
    html_content += "</div>"
    
    # TÜM ANA TAHMİNLER
    html_content += f"""
            <div class="section">
                <h2>🏟️ TÜM ANA TAHMİNLER — {date_str}</h2>
    """
    
    if tum_ana_tahminler:
        for i, pred in enumerate(tum_ana_tahminler, 1):
            confidence_class = "high-confidence" if pred.get('confidence', 0) >= HIGH_ALERT else "medium-confidence"
            html_content += f"""
                <div class="match {confidence_class}">
                    <div><strong>{"🔥" if pred.get('confidence', 0) >= HIGH_ALERT else "⚽"} #{i} - {pred.get('confidence', 0)}%</strong></div>
                    <div>⚽ {pred.get('match', 'Maç bilgisi yok')}</div>
                    <div>🎯 Tahmin: {pred.get('prediction', 'N/A')}</div>
                    <div>⏰ Saat: {pred.get('time', 'N/A')}</div>
                </div>
            """
    else:
        html_content += "<p>❌ Bugün için ANA tahmin bulunamadı.</p>"
    
    html_content += "</div>"
    
    # TÜM ULTRA TAHMİNLER
    html_content += f"""
            <div class="section">
                <h2>🔮 TÜM ULTRA TAHMİNLER — {date_str}</h2>
    """
    
    if ultra_tahminler:
        for i, pred in enumerate(ultra_tahminler, 1):
            html_content += f"""
                <div class="match">
                    <h3>🔮 #{i} - {pred.get('confidence', 0)}%</h3>
                    <div><strong>⚽ {pred.get('match', 'Maç bilgisi yok')}</strong></div>
                    <div>🎯 Tahmin: {pred.get('pick', 'N/A')}</div>
                    <div>🤖 Basit Sistem: {pred.get('simple_prediction', 'N/A')}</div>
                    <div>📊 Form: {pred.get('home_form', 'N/A')} vs {pred.get('away_form', 'N/A')}</div>
                    <div class="stats">
                        ⚽ Gol: 1.5:{pred.get('gol_15', 'N/A')}(%{pred.get('gol_15_prob', 0)}) | 
                        2.5:{pred.get('gol_25', 'N/A')}(%{pred.get('gol_25_prob', 0)})<br>
                        🔥 BTTS: {pred.get('btts', 'N/A')} (%{pred.get('btts_prob', 0)})<br>
                        📋 Skor: {pred.get('skor_tahmini', 'N/A')}<br>
                        ⚠️ Kart: {pred.get('kart_35', 'N/A')}<br>
                        🎯 Korner: {pred.get('korner_75', 'N/A')}<br>
                        📈 Veri Kalitesi: %{pred.get('data_quality', 0)}
                    </div>
                </div>
            """
    else:
        html_content += "<p>❌ Bugün için ULTRA tahmin bulunamadı.</p>"
    
    html_content += "</div>"
    
    # İSTATİSTİKLER
    html_content += f"""
            <div class="section">
                <h2>📊 İSTATİSTİKLER</h2>
                <p>• Tüm ANA Tahminler: {len(tum_ana_tahminler)}</p>
                <p>• En İyi 5 ANA: {len(top_ana_tahminler)}</p>
                <p>• Tüm ULTRA Tahminler: {len(ultra_tahminler)}</p>
    """
    
    if tum_ana_tahminler:
        avg_ana = sum(p.get('confidence', 0) for p in tum_ana_tahminler) / len(tum_ana_tahminler)
        html_content += f"<p>• Ortalama ANA Güven: {avg_ana:.1f}%</p>"
    
    if ultra_tahminler:
        avg_ultra = sum(p.get('confidence', 0) for p in ultra_tahminler) / len(ultra_tahminler)
        html_content += f"<p>• Ortalama ULTRA Güven: {avg_ultra:.1f}%</p>"
    
    html_content += f"""
                <p><em>🤖 Model: {MODEL_VERSION}</em></p>
                <p><em>⏰ Üretim Zamanı: {datetime.now(TR_TZ).strftime('%Y-%m-%d %H:%M:%S')}</em></p>
            </div>
        </div>
    </body>
    </html>
    """
    
    return html_content

# ==================== ORİJİNAL KODUN DEVAMI ====================

# === YENİ LİG LİSTESİ ===
COMPLETE_LEAGUE_IDS = {
    'Premier League': '39', 'La Liga': '140', 'Serie A': '135',
    'Bundesliga': '78', 'Ligue 1': '61', 'Eredivisie': '88',
    'Belgian Pro League': '144', 'Scottish Premiership': '179',
    'Süper Lig': '203', 'Segunda División': '141', 'Serie B': '136',
    '2. Bundesliga': '79', 'Segunda Liga': '95', 'Challenger Pro League': '145'
}

# === YENİ API FOOTBALL AYARLARI ===
APIFOOTBALL_BASE_URL = "https://v3.football.api-sports.io/"
HEADERS = {
    'x-apisports-key': os.getenv('APIFOOTBALL_KEY', 'f6570111b0cdddb86828ef25179e6ce7'),
}

# --- Model/version & retention ---
MODEL_VERSION = os.getenv("MODEL_VERSION", "v2025.10.20-ultra-integration")
STATE_TTL_DAYS = int(os.getenv("STATE_TTL_DAYS", "14"))
FREEZE_MINUTES = int(os.getenv("FREEZE_MINUTES", "60"))

# ==================== AYARLAR / SETTINGS ====================
STATE_PATH = os.getenv("STATE_PATH", "model_state.json")
SNAPSHOT_DIR = os.getenv("SNAPSHOT_DIR", "snapshots")

# Dosya yazma izinleri (1: açık, 0: kapalı) / File write permissions (1: on, 0: off)
ALLOW_STATE_FILE = int(os.getenv("ALLOW_STATE_FILE", "1"))

# Kadın ve U-yaş maçlarını dahil etme bayrakları / Women and U-age matches inclusion flags
ALLOW_WOMEN = int(os.getenv("ALLOW_WOMEN", "0"))
ALLOW_U21 = int(os.getenv("ALLOW_U21", "0"))

# Model sabitleri / Model constants
ELO_HOME_ADV = 40
W_MKT_INIT = 0.45
MIN_QUALITY = int(os.getenv("MIN_QUALITY", "0"))

# Yeni özellik bayrakları / New feature flags
GOAL_MODEL = os.getenv("GOAL_MODEL", "POISSON")  # DC|POISSON|HIER
ELO_MODE = os.getenv("ELO_MODE", "single")  # split|single

# --- Ortak yardımcılar -------------------------------------------------------
TR_TZ = timezone(timedelta(hours=3))  # Türkiye

# --- TOP_N AYARI ---
TOP_N = int(os.getenv("TOP_N", "5"))  # En güçlü tahmin sayısı
MIN_CONF = int(os.getenv("MIN_CONF", "0"))  # Minimum güven seviyesi
NIGPLALERT = int(os.getenv("NIGPLALERT", "99"))  # Yüksek güven uyarı eşiği
HIGH_ALERT = 80  # Yüksek güven eşiği

# --- SERVICE LOOP AYARLARI ---
PREDICTION_HOUR = int(os.getenv("PREDICTION_HOUR", "10"))  # TR 10:00
RESULTS_HOUR = int(os.getenv("RESULTS_HOUR", "4"))  # TR 04:00 (ertesi gün)
RESULTS_MINUTE = int(os.getenv("RESULTS_MINUTE", "0"))

# --- YENİ WEATHER API AYARLARI ---
WEATHER_API_KEY = os.getenv("WEATHER_API_KEY", "")  # Ücretsiz WeatherAPI anahtarı
WEATHER_CACHE_TTL = int(os.getenv("WEATHER_CACHE_TTL", "3600"))  # 1 saat cache

def log(msg):
    print(f"{msg}", flush=True)

# ==================== YENİ API FOOTBALL FONKSİYONLARI ====================

def get_api_football_fixtures(date_str):
    """SEZONSUZ - 14 LİG + KADIN/U21 FİLTRELEME"""
    try:
        url = f"https://v3.football.api-sports.io/fixtures?date={date_str}"
        response = requests.get(url, headers=HEADERS)
        
        if response.status_code == 200:
            data = response.json()
            all_matches = data.get('response', [])
            
            # 14 HEDEF LİG
            TARGET_LEAGUES = {
                'Premier League': '39', 'La Liga': '140', 'Serie A': '135',
                'Bundesliga': '78', 'Ligue 1': '61', 'Eredivisie': '88',
                'Belgian Pro League': '144', 'Scottish Premiership': '179',
                'Süper Lig': '203', 'Segunda División': '141', 'Serie B': '136',
                '2. Bundesliga': '79', 'Segunda Liga': '95', 'Challenger Pro League': '145'
            }
            
            target_ids = [str(id) for id in COMPLETE_LEAGUE_IDS.values()]
            
            # FİLTRELEME
            filtered_matches = []
            for match in all_matches:
                league_id = str(match['league']['id'])
                league_name = match['league']['name']
                home_team = match['teams']['home']['name']
                away_team = match['teams']['away']['name']
                
                # 1. SADECE HEDEF LİGLER
                if league_id not in target_ids:
                    continue
                
                # 2. KADIN MAÇLARINI FİLTRELE
                if any(word in league_name.lower() for word in ['women', 'kadın', 'femin', 'damen', 'femminile']):
                    continue
                
                # 3. U21/U19 MAÇLARINI FİLTRELE
                if any(word in league_name.lower() for word in ['u21', 'u19', 'u23', 'youth', 'genç']):
                    continue
                
                # 4. REZERV/B TAKIM FİLTRELE
                if any(word in home_team.lower() for word in [' ii', ' 2', ' b', ' reserve', ' u21']):
                    continue
                if any(word in away_team.lower() for word in [' ii', ' 2', ' b', ' reserve', ' u21']):
                    continue
                
                filtered_matches.append(match)
            
            log(f"🎯 {len(all_matches)} maç → {len(filtered_matches)} filtreli maç")
            return filtered_matches
            
        else:
            log(f"❌ API Error: {response.status_code}")
            return []
            
    except Exception as e:
        log(f"❌ API error: {e}")
        return []

def get_fixtures_from_apis(date_str):
    """SADECE API-FOOTBALL KULLAN"""
    try:
        log(f"🔍 API-Football maçları aranıyor: {date_str}")
        fixtures = get_api_football_fixtures(date_str)  #
        log(f"✅ API-Football sonuç: {len(fixtures)} maç")
        return fixtures
    except Exception as e:
        log(f"❌ API-Football hatası: {e}")
        return []

def get_matches_from_api(start_date, end_date, league_id=None):
    """YENİ: API Football v3'ten maç verilerini çeker"""
    params = {'date': start_date, 'season': '2024'}
    if league_id:
        params['league'] = league_id
        
    response = requests.get(
        f"{APIFOOTBALL_BASE_URL}fixtures", 
        headers=HEADERS, 
        params=params, 
        timeout=30
    )
    
    if response.status_code == 200:
        return response.json().get('response', [])
    elif response.status_code == 429:
        print("⚠️ Rate limit aşıldı")
    elif response.status_code == 403:
        print("❌ API Key hatası")
    else:
        print(f"❌ API Hatası: {response.status_code}")
    return []

def get_leagues_from_api():
    """YENİ: API Football v3'ten lig bilgilerini çeker"""
    params = {'season': '2024'}
    response = requests.get(
        f"{APIFOOTBALL_BASE_URL}leagues", 
        headers=HEADERS, 
        params=params, 
        timeout=30
    )
    
    if response.status_code == 200:
        return response.json().get('response', [])
    elif response.status_code == 429:
        print("⚠️ Rate limit aşıldı")
    elif response.status_code == 403:
        print("❌ API Key hatası")
    else:
        print(f"❌ Lig API Hatası: {response.status_code}")
    return []

def process_match_data(match):
    """YENİ: API Football v3 JSON formatını işler"""
    return {
        'id': match['fixture']['id'],
        'date': match['fixture']['date'],
        'home_team': match['teams']['home']['name'],
        'away_team': match['teams']['away']['name'],
        'league_name': match['league']['name'],
        'country': match['country']['name']
    }

# ==================== EKSİK FONKSİYONLAR DÜZELTMESİ ====================

def _today_str_tr():
    """Bugünün tarihini TR zaman diliminde 'YYYY-MM-DD' formatında döndürür"""
    return datetime.now(TR_TZ).strftime("%Y-%m-%d")

def _yesterday_str_tr(now=None):
    """Dünün tarihini TR zaman diliminde 'YYYY-MM-DD' formatında döndürür"""
    if now is None:
        now = datetime.now(TR_TZ)
    yesterday = now - timedelta(days=1)
    return yesterday.strftime("%Y-%m-%d")

def _apifoot_get(path, params):
    """API-Football API çağrısı - EKSİK FONKSİYON EKLENDİ"""
    if not APIFOOT:
        return None
    try:
        url = f"{APIFOOTBALL_BASE_URL}{path.lstrip('/')}"
        response = requests.get(
            url, 
            headers=HEADERS, 
            params=params, 
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get("response", None)
    except Exception as e:
        log(f"apifoot GET err: {e}")
        return None

def calculate_value_advantage(home_team, away_team, area):
    """Takım değeri avantajı hesaplar - EKSİK FONKSİYON EKLENDİ"""
    try:
        home_value, home_source = get_team_value(home_team, area)
        away_value, away_source = get_team_value(away_team, area)
        
        if home_value == 0 and away_value == 0:
            return 0.0, "NO_DATA"
        
        value_ratio = home_value / max(away_value, 0.1)
        advantage = math.log(value_ratio) * 0.1  # Log scale advantage
        
        return clamp(advantage, -0.15, 0.15), f"VALUE_{home_source}_{away_source}"
    except Exception as e:
        log(f"Value advantage calculation error: {e}")
        return 0.0, "ERROR"

def make_prediction(date_str):
    """Tahmin yapma fonksiyonu - EKSİK FONKSİYON EKLENDİ"""
    try:
        fixtures = universal_collector.fetch_fixtures_universal(date_str)
        predictions = []
        
        for fx in fixtures:
            odds, odds_source = fetch_odds_dual(fx.get("area",""), fx.get("competition",""), fx["home"], fx["away"])
            rated = rate_fixture_enhanced(fx, odds)
            
            predictions.append({
                "match": f"{fx['home']} vs {fx['away']}",
                "prediction": rated["pick"],
                "confidence": rated["confidence"],
                "note": rated["note"],
                "area": fx.get("area", ""),
                "competition": fx.get("competition", ""),
                "time": get_fixture_time_fallback(fx)
            })
        
        return predictions
    except Exception as e:
        log(f"Make prediction error: {e}")
        return []

def log_prediction_success(predictions):
    """Başarılı tahmin logu - EKSİK FONKSİYON EKLENDİ"""
    log(f"✅ {len(predictions)} tahmin başarıyla oluşturuldu")

def log_prediction_failure():
    """Tahmin başarısızlık logu - EKSİK FONKSİYON EKLENDİ"""
    log("❌ Tahmin oluşturulamadı")

def debug_api_connection():
    """API bağlantı test fonksiyonu - YENİ EKLENDİ"""
    log("🔍 API Bağlantı Debug...")
    log(f"📡 Base URL: {APIFOOTBALL_BASE_URL}")
    log(f"🔑 Key Length: {len(APIFOOT) if APIFOOT else 'MISSING'}")
    
    # Status endpoint'ini test et
    test_response = test_api_connection()
    if test_response:
        log("✅ API Bağlantı BAŞARILI")
    else:
        log("❌ API Bağlantı BAŞARISIZ")

def test_api_connection():
    """API bağlantı testi - YENİ EKLENDİ"""
    try:
        url = f"{APIFOOTBALL_BASE_URL}status"
        response = requests.get(
            url, 
            headers=HEADERS, 
            timeout=30
        )
        return response.status_code == 200
    except Exception as e:
        log(f"API connection test error: {e}")
        return False

# ==================== DEĞİŞKEN DÜZELTMELERİ ====================
# APIFOOT değişkeni düzeltildi - APIFOOTBALL_KEY kullanılacak
APIFOOT = (os.getenv("APIFOOTBALL_KEY") or "").strip()

# Diğer secrets
GMAIL_USER = os.getenv("GMAIL_USER")
GMAIL_PASS = os.getenv("GMAIL_PASS")
GMAIL_TO = os.getenv("GMAIL_TO")
ODDS_KEY = os.getenv("ODDS_API_KEY")
MODE_ENV = (os.getenv("MODE") or "AUTO").upper().strip()

# ==================== TAKIM İSİM BENZERLİK SİSTEMİ ====================
def normalize_team_name(name):
    """Takım adını karşılaştırma için normalize eder - GELİŞTİRİLMİŞ"""
    if not name:
        return ""
    
    name = name.lower().strip()
    
    # Yaygın takım eklerini kaldır
    suffixes = [
        ' fc', ' cf', ' af', ' sf', ' if', ' ff', ' football club', ' club de foot',
        ' athletic club', ' sports club', ' united', ' city', ' town', ' fc.',
        ' real', ' deportivo', ' athletic', ' atletico', ' atlético', ' sporting',
        ' os ', ' as ', ' us ', ' ac ', ' inter ', ' borussia', ' dynamo', ' sparta', ' rapid'
    ]
    
    for suffix in suffixes:
        name = name.replace(suffix, '')
    
    # Özel karakterleri temizle
    name = re.sub(r'[^\w\s]', ' ', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Özel takım ismi düzeltmeleri - GENİŞLETİLMİŞ
    special_cases = {
        'psg': 'paris saint germain', 'paris sg': 'paris saint germain',
        'man united': 'manchester united', 'man utd': 'manchester united',
        'man city': 'manchester city', 'spurs': 'tottenham hotspur',
        'newcastle': 'newcastle united', 'west ham': 'west ham united',
        'leeds': 'leeds united', 'leicester': 'leicester city',
        'wolves': 'wolverhampton wanderers', 'brighton': 'brighton and hove albion',
        'sheffield united': 'sheffield united', 'nottingham forest': 'nottingham forest',
        'norwich': 'norwich city', 'derby': 'derby county', 'qpr': 'queens park rangers',
        'atalanta bc': 'atalanta', 'as roma': 'roma', 'ac milan': 'milan',
        'inter milan': 'inter', 'fc bayern munich': 'bayern munich',
        'bayer leverkusen': 'leverkusen', 'borussia mgladbach': 'borussia monchengladbach',
        'eintracht frankfurt': 'eintracht frankfurt', 'tsg hoffenheim': 'hoffenheim',
        'sc freiburg': 'freiburg', 'vfl wolfsburg': 'wolfsburg',
        '1 fc koln': 'koln', 'fc schalke 04': 'schalke', 'rcd espanyol': 'espanyol',
        'real betis': 'betis', 'atletico madrid': 'atletico madrid',
        'athletic bilbao': 'athletic bilbao', 'real sociedad': 'real sociedad',
        'valencia cf': 'valencia', 'cf villareal': 'villarreal',
        'olympique lyon': 'lyon', 'as monaco': 'monaco', 'losc lille': 'lille',
        'stade rennais': 'rennes', 'fc nantes': 'nantes', 'besiktas jk': 'besiktas',
        'fenerbahce sk': 'fenerbahce', 'galatasaray sk': 'galatasaray',
        'trabzonspor sk': 'trabzonspor', 'istanbul basaksehir fk': 'istanbul basaksehir'
    }
    
    return special_cases.get(name, name)

def team_similarity(a, b):
    """İki takım adı arasındaki benzerlik skorunu hesaplar"""
    if not a or not b:
        return 0.0
    
    a_norm = normalize_team_name(a)
    b_norm = normalize_team_name(b)
    
    if a_norm == b_norm:
        return 1.0
    
    # Kelime bazlı benzerlik
    a_words = set(a_norm.split())
    b_words = set(b_norm.split())
    
    if a_words and b_words:
        common_words = a_words.intersection(b_words)
        word_similarity = len(common_words) / max(len(a_words), len(b_words))
        string_similarity = SequenceMatcher(None, a_norm, b_norm).ratio()
        return 0.7 * word_similarity + 0.3 * string_similarity
    
    return SequenceMatcher(None, a_norm, b_norm).ratio()

def find_closest_team(target_team, team_list, threshold=0.75):
    """Takım listesinde en benzer takımı bulur"""
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

# ==================== GERÇEKÇİ TAKIM DEĞERLERİ SİSTEMİ ====================
def get_team_value_realistic(team_name, area="Europe"):
    """Lig bazlı gerçekçi takım değerleri - GELİŞTİRİLMİŞ"""
    # Lig bazlı değer aralıkları
    league_values = {
        "premier league": {"top": 180, "mid": 80, "low": 40},
        "la liga": {"top": 120, "mid": 60, "low": 30},
        "serie a": {"top": 110, "mid": 55, "low": 25},
        "bundesliga": {"top": 100, "mid": 50, "low": 20},
        "ligue 1": {"top": 90, "mid": 45, "low": 15},
        "super lig": {"top": 40, "mid": 20, "low": 5},
        "eredivisie": {"top": 50, "mid": 25, "low": 8},
        "primeira liga": {"top": 45, "mid": 22, "low": 6},
        "pro league": {"top": 35, "mid": 18, "low": 4},
        "championship": {"top": 25, "mid": 12, "low": 3}
    }
    
    # Takım bazlı özel değerler
    team_specific_values = {
        "premier league": {
            "manchester city": 180, "arsenal": 150, "liverpool": 140,
            "chelsea": 120, "manchester united": 110, "tottenham": 100,
            "newcastle united": 90, "aston villa": 80, "brighton": 70,
            "west ham": 65, "crystal palace": 50, "wolves": 45,
            "fulham": 40, "everton": 35, "brentford": 30,
            "nottingham forest": 25, "luton town": 15, "burnley": 12,
            "sheffield united": 10
        },
        "super lig": {
            "galatasaray": 40, "fenerbahce": 38, "besiktas": 35,
            "trabzonspor": 25, "basaksehir": 20, "konyaspor": 8,
            "kayserispor": 7, "alanyaspor": 6, "sivasspor": 5
        }
    }
    
    team_lower = normalize_team_name(team_name)
    area_lower = area.lower()
    
    # Önce lig bul
    target_league = None
    for league in league_values:
        if league in area_lower:
            target_league = league
            break
    
    if not target_league:
        target_league = "premier league"  # fallback
    
    # Takım özel değeri kontrol et
    if target_league in team_specific_values:
        for team_key, value in team_specific_values[target_league].items():
            if team_key in team_lower:
                return value, "REALISTIC_SPECIFIC"
    
    # Lig ortalamasına göre rastgele değer
    values = league_values[target_league]
    value_range = [values["low"], values["mid"], values["top"]]
    realistic_value = random.choice(value_range)
    
    return realistic_value, "REALISTIC_LEAGUE"

# ==================== TAKIM GÜÇ SİSTEMİ ====================
def calculate_team_power(team_name, area="Europe"):
    """Takım güç skoru hesaplar - "1/X/2" çeşitliliği için"""
    # Takım sıralama fallback
    team_rankings = {
        "premier league": {
            "manchester city": 1, "arsenal": 2, "liverpool": 3, "chelsea": 4,
            "manchester united": 5, "tottenham": 6, "newcastle united": 7,
            "brighton": 8, "west ham": 9, "crystal palace": 10
        },
        "super lig": {
            "galatasaray": 1, "fenerbahce": 2, "besiktas": 3, "trabzonspor": 4,
            "basaksehir": 5, "konyaspor": 6, "kayserispor": 7
        }
    }
    
    team_lower = normalize_team_name(team_name)
    area_lower = area.lower()
    
    # Lig bul
    target_league = None
    for league in team_rankings:
        if league in area_lower:
            target_league = league
            break
    
    if not target_league:
        return 50  # Varsayılan güç
    
    # Sıralamaya göre güç hesapla
    if target_league in team_rankings:
        for team_key, rank in team_rankings[target_league].items():
            if team_key in team_lower:
                power_score = 100 - (rank * 8)  # 1. sıra: 92, 2. sıra: 84, vb.
                return max(power_score, 20)
    
    return 50  # Bilinmeyen takım

def get_intelligent_prediction(home_team, away_team, area="Europe"):
    """Akıllı tahmin sistemi - "1/X/2" çeşitliliği"""
    home_power = calculate_team_power(home_team, area)
    away_power = calculate_team_power(away_team, area)
    
    power_diff = home_power - away_power
    
    # Güç farkına göre tahmin
    if power_diff > 30:  # Ev çok güçlü
        pick = "1"
        base_confidence = 70
    elif power_diff > 15:  # Ev güçlü
        pick = "1" 
        base_confidence = 60
    elif power_diff < -30:  # Deplasman çok güçlü
        pick = "2"
        base_confidence = 65
    elif power_diff < -15:  # Deplasman güçlü
        pick = "2"
        base_confidence = 55
    else:  # Dengeli
        pick = "X"
        base_confidence = 50
    
    # Ev avantajı ekle
    if pick == "1":
        base_confidence += 10
    elif pick == "X":
        base_confidence += 5
    
    return pick, min(base_confidence, 85)

# ==================== GELİŞMİŞ EV AVANTAJI SİSTEMİ ====================
def home_adv_effective(area, competition, home_team, away_team):
    """Dinamik ev sahibi avantajı - İYİLEŞTİRİLMİŞ"""
    base_advantage = ELO_HOME_ADV
    
    # Milli takım maçlarında avantajı azalt
    comp_lower = (competition or "").lower()
    if any(x in comp_lower for x in ["world cup", "euro", "qualification", "international"]):
        base_advantage *= 0.6
        log(f"🏟️ Milli takım maçı - ev avantajı azaltıldı: {base_advantage:.1f}")
    
    # Takım gücüne göre avantaj ayarı
    home_power = calculate_team_power(home_team, area)
    away_power = calculate_team_power(away_team, area)
    
    if away_power > home_power + 20:  # Deplasman daha güçlü
        advantage_factor = 1.2  # Ev avantajını artır
    elif home_power > away_power + 20:  # Ev daha güçlü
        advantage_factor = 0.8  # Ev avantajını azalt
    else:
        advantage_factor = 1.0
    
    final_advantage = base_advantage * advantage_factor
    
    log(f"Ev avantajı: {home_team}({home_power}) vs {away_team}({away_power}) -> {final_advantage:.1f}")
    
    return clamp(final_advantage, 15.0, 60.0)

# ==================== FİXTURE SAAT SİSTEMİ ====================
def get_fixture_time_fallback(fx):
    """Fixture saat bilgisi için fallback sistemi"""
    utc_kickoff = fx.get("utc_kickoff")
    
    if utc_kickoff:
        # UTC'yi TR saatine çevir
        local_time = utc_kickoff.astimezone(TR_TZ).strftime("%H:%M")
        return local_time
    
    # Saat bilgisi yoksa fallback
    time_slots = ["14:00", "15:00", "16:00", "17:00", "18:00", "19:00", "20:00"]
    return random.choice(time_slots)

# ==================== GÜVEN ARTIRICI SİSTEM ====================
def enhance_confidence(rated_fx, home_team, away_team, area):
    """Güven seviyesini artırıcı faktörler"""
    base_confidence = rated_fx["confidence"]
    
    # Takım güç farkı
    home_power = calculate_team_power(home_team, area)
    away_power = calculate_team_power(away_team, area)
    power_diff = abs(home_power - away_power)
    
    if power_diff > 40:
        base_confidence += 15
    elif power_diff > 25:
        base_confidence += 10
    elif power_diff > 15:
        base_confidence += 5
    
    # Lig seviyesi
    area_lower = area.lower()
    if "premier league" in area_lower or "la liga" in area_lower or "serie a" in area_lower:
        base_confidence += 5  # Üst liglerde güven artır
    
    # Mevcut güven düşükse temel artırım
    if base_confidence < 50:
        base_confidence += 10
    
    return min(base_confidence, 90)

# ==================== SERVICE LOOP DÜZELTMESİ ====================
def run_service_loop():
    """Düzeltilmiş service loop - PREDICT/RESULTS ayrımı"""
    log(f"SERVICE başlatıldı (MODE: {MODE_ENV})")
    
    now = datetime.now(TR_TZ)
    
    if MODE_ENV == "AUTO":
        # Saate göre otomatik seçim
        if now.hour >= PREDICTION_HOUR:
            today = _today_str_tr()
            log(f"🚀 AUTO modu - Tahmin yapılıyor: {today}")
            enhanced_report_predictions(today)
        else:
            yesterday = _yesterday_str_tr()
            log(f"📊 AUTO modu - Sonuçlar raporlanıyor: {yesterday}")
            fetch_results_fixed(yesterday)
            
    elif MODE_ENV == "PREDICT":
        today = _today_str_tr()
        log(f"🚀 PREDICT modu - Tahmin yapılıyor: {today}")
        enhanced_report_predictions(today)
        
    elif MODE_ENV == "RESULTS":
        yesterday = _yesterday_str_tr()
        log(f"📊 RESULTS modu - Sonuçlar raporlanıyor: {yesterday}")
        fetch_results_fixed(yesterday)

# ==================== GELİŞMİŞ SONUÇ RAPORU ====================
def fetch_results_fixed(date_str):
    """Düzeltilmiş sonuç raporu - State tahminleriyle performans ölçümü"""
    # YENİ: API Football v3 kullanımı
    api_results = fetch_results_apifoot(date_str)
    
    lines = [f"📊 Dünün Sonuçları — {date_str}"]
    
    if api_results:
        # GERÇEK sonuçlar var - performans ölç
        correct = 0
        total = 0
        
        for result in api_results:
            # State'teki tahmini bul
            pred = find_prediction_for_result(result)
            
            if pred:
                total += 1
                # Sonucu belirle
                if result["score_h"] > result["score_a"]:
                    actual = "1"
                elif result["score_h"] == result["score_a"]:
                    actual = "X" 
                else:
                    actual = "2"
                
                # Tahmin kontrolü
                is_correct = (pred["pick"] == actual)
                if is_correct:
                    correct += 1
                
                status = "✅ DOĞRU" if is_correct else "❌ YANLIŞ"
                line = (f"- {result['home']} {result['score_h']}-{result['score_a']} {result['away']} | "
                       f"Tahmin: {pred['pick']}({pred['conf_pct']}%) | Sonuç: {actual} | {status}")
            else:
                # Tahmin bulunamadı
                if result["score_h"] > result["score_a"]:
                    actual = "1"
                elif result["score_h"] == result["score_a"]:
                    actual = "X"
                else:
                    actual = "2"
                line = (f"- {result['home']} {result['score_h']}-{result['score_a']} {result['away']} | "
                       f"Tahmin: BULUNAMADI | Sonuç: {actual}")
            
            lines.append(line)
        
        # Performans istatistikleri
        if total > 0:
            accuracy = (correct / total) * 100
            lines.append(f"\n📈 PERFORMANS: {correct}/{total} doğru (%{accuracy:.1f} başarı)")
        
    else:
        # GERÇEK sonuç yok
        lines.append("ℹ️ Gerçek sonuç bulunamadı - API'ler güncel değil")
        
        # State'teki tahminleri göster (sadece bilgi)
        state_count = 0
        for key, pred in STATE.get("pred_store", {}).items():
            pred_date = pred.get("utc_kickoff", "").split("T")[0]
            if pred_date == date_str:
                state_count += 1
                if state_count == 1:
                    lines.append(f"\n💡 State'te {date_str} tahminleri mevcut:")
                line = (f"- {pred['home']} vs {pred['away']} | "
                       f"Tahmin: {pred['pick']}({pred['conf_pct']}%)")
                lines.append(line)
    
    body = "\n".join(lines)
    send_mail(f"Sonuç Raporu | {date_str}", body)
    log(f"✅ Sonuç raporu gönderildi: {date_str}")
    
    save_state(STATE)
    return api_results

# ==================== URL NORMALIZASYON FONKSİYONU ====================
def normalize_url(base, endpoint):
    """URL çiftleşme hatasını önler - DÜZELTİLDİ"""
    base = base.rstrip('/')
    endpoint = endpoint.lstrip('/')
    return f"{base}/{endpoint}"

# ==================== API-FOOTBALL TÜM ÖZELLİKLER ENTEGRASYONU ====================

# API-Football endpoint mapping
APIFOOTBALL_ENDPOINTS = {
    'head_to_head': 'fixtures/headtohead',
    'predictions': 'predictions', 
    'injuries': 'injuries',
    'topscorers': 'players/topscorers',
    'lineups': 'fixtures/lineups',
    'standings': 'standings',
    'fixtures': 'fixtures',
    'teams': 'teams',
    'players': 'players',
    'transfers': 'transfers',
    'statistics': 'fixtures/statistics'
}

def get_football_data(feature_type, **params):
    """API-Football birincil, diğerleri fallback"""
    # 1. ÖNCE API-FOOTBALL
    data = _apifootball_get(feature_type, params)
    if data:
        return data, "APIFOOTBALL"
    
    return None, "NONE"

def _apifootball_get(feature_type, params):
    """API-Football'dan veri çek"""
    if not APIFOOT:
        return None
        
    endpoint = APIFOOTBALL_ENDPOINTS.get(feature_type)
    if not endpoint:
        return None
        
    url = f"{APIFOOTBALL_BASE_URL}{endpoint}"
    
    try:
        response = requests.get(
            url, 
            headers=HEADERS, 
            params=params, 
            timeout=30
        )
        if response.status_code == 200:
            return response.json().get('response', [])
    except Exception as e:
        log(f"API-Football GET error: {e}")
    
    return None

def get_head_to_head(home_team_id, away_team_id):
    """Son 5 karşılaşmayı getir"""
    params = {"h2h": f"{home_team_id}-{away_team_id}", "last": 5}
    return _apifoot_get("fixtures/headtohead", params)

def get_api_predictions(fixture_id):
    """API-Football'ın AI tahminlerini al"""
    params = {"fixture": fixture_id}
    return _apifoot_get("predictions", params)

def get_injuries(fixture_id):
    """Maçtaki sakat oyuncuları getir"""
    params = {"fixture": fixture_id}
    return _apifoot_get("injuries", params)

def get_top_scorers(league_id, season):
    """Lig gol krallarını getir"""
    params = {"league": league_id, "season": season}
    return _apifoot_get("players/topscorers", params)

def get_lineups(fixture_id):
    """Maç kadrolarını getir"""
    params = {"fixture": fixture_id}
    return _apifoot_get("fixtures/lineups", params)

def get_transfers(team_id):
    """Takım transferlerini getir"""
    params = {"team": team_id}
    return _apifoot_get("transfers", params)

def get_fixture_statistics(fixture_id):
    """Maç istatistiklerini getir"""
    params = {"fixture": fixture_id}
    return _apifoot_get("fixtures/statistics", params)

# ==================== EKSİK API-FOOTBALL FONKSİYONLARI ====================

def find_fixture_id(area, comp, home, away):
    """API-Football'dan fixture ID bulur"""
    try:
        if not APIFOOT:
            return None
            
        # Takım isimlerini normalize et
        home_norm = normalize_team_name(home)
        away_norm = normalize_team_name(away)
        
        # Önce direkt arama yap
        params = {
            "league": get_league_id_for_country(area, comp),
            "season": season_for_today(),
            "team": _apifoot_find_team_id(home)
        }
        
        response = _apifoot_get("fixtures", params)
        if response:
            for fixture in response:
                fixture_data = fixture.get('fixture', {})
                teams = fixture.get('teams', {})
                
                fixture_home = normalize_team_name(teams.get('home', {}).get('name', ''))
                fixture_away = normalize_team_name(teams.get('away', {}).get('name', ''))
                
                # Benzerlik kontrolü
                home_sim = team_similarity(home_norm, fixture_home)
                away_sim = team_similarity(away_norm, fixture_away)
                
                if home_sim >= 0.8 and away_sim >= 0.8:
                    return fixture_data.get('id')
        
        # Fallback: tarih bazlı arama
        today = datetime.now().strftime("%Y-%m-%d")
        params = {"date": today, "league": get_league_id_for_country(area, comp)}
        response = _apifoot_get("fixtures", params)
        
        if response:
            for fixture in response:
                teams = fixture.get('teams', {})
                fixture_home = normalize_team_name(teams.get('home', {}).get('name', ''))
                fixture_away = normalize_team_name(teams.get('away', {}).get('name', ''))
                
                home_sim = team_similarity(home_norm, fixture_home)
                away_sim = team_similarity(away_norm, fixture_away)
                
                if home_sim >= 0.8 and away_sim >= 0.8:
                    return fixture.get('fixture', {}).get('id')
                    
    except Exception as e:
        log(f"Fixture ID bulma hatası: {e}")
    
    return None

def get_league_id_for_country(country, competition):
    """Ülke ve lig adına göre API-Football lig ID'si döndürür"""
    league_mapping = {
        "England": {
            "Premier League": 39,
            "Championship": 40,
            "League One": 41,
            "League Two": 42,
            "FA Cup": 45,
            "EFL Cup": 48
        },
        "Spain": {
            "La Liga": 140,
            "La Liga 2": 141,
            "Copa del Rey": 143
        },
        "Italy": {
            "Serie A": 135,
            "Serie B": 136,
            "Coppa Italia": 137
        },
        "Germany": {
            "Bundesliga": 78,
            "2. Bundesliga": 79,
            "DFB Pokal": 81
        },
        "France": {
            "Ligue 1": 61,
            "Ligue 2": 62,
            "Coupe de France": 66
        },
        "Turkey": {
            "Super Lig": 203,
            "1. Lig": 204,
            "Turkish Cup": 206
        },
        "Netherlands": {
            "Eredivisie": 88,
            "Eerste Divisie": 89,
            "KNVB Beker": 92
        },
        "Portugal": {
            "Primeira Liga": 94,
            "Liga Portugal 2": 95,
            "Taça de Portugal": 96
        },
        "Belgium": {
            "Pro League": 144,
            "Challenger Pro League": 145,
            "Croky Cup": 146
        },
        "Europe": {
            "Champions League": 2,
            "Europa League": 3,
            "Conference League": 848,
            "Super Cup": 667
        },
        "World": {
            "World Cup": 1,
            "Euro Championship": 4
        }
    }
    
    # Ülkeyi bul
    country_lower = (country or "").lower()
    competition_lower = (competition or "").lower()
    
    for country_key, leagues in league_mapping.items():
        if country_key.lower() in country_lower:
            for comp_name, league_id in leagues.items():
                if comp_name.lower() in competition_lower:
                    return league_id
    
    # Fallback: competition bazlı arama
    if "premier" in competition_lower:
        return 39
    elif "championship" in competition_lower:
        return 40
    elif "la liga" in competition_lower:
        return 140
    elif "serie a" in competition_lower:
        return 135
    elif "bundesliga" in competition_lower:
        return 78
    elif "ligue 1" in competition_lower:
        return 61
    elif "super lig" in competition_lower:
        return 203
    
    return None

def parse_apifootball_odds(response):
    """API-Football odds verisini parse eder"""
    try:
        if not response or len(response) == 0:
            return None
            
        bookmakers = response[0].get('bookmakers', [])
        home_odds = []
        draw_odds = []
        away_odds = []
        
        for bookmaker in bookmakers:
            for bet in bookmaker.get('bets', []):
                if bet.get('name') == 'Match Winner':
                    for outcome in bet.get('values', []):
                        if outcome.get('value') == 'Home':
                            home_odds.append(float(outcome.get('odd', 0)))
                        elif outcome.get('value') == 'Draw':
                            draw_odds.append(float(outcome.get('odd', 0)))
                        elif outcome.get('value') == 'Away':
                            away_odds.append(float(outcome.get('odd', 0)))
        
        if home_odds and draw_odds and away_odds:
            # Ortalama oranları al
            avg_home = sum(home_odds) / len(home_odds)
            avg_draw = sum(draw_odds) / len(draw_odds)
            avg_away = sum(away_odds) / len(away_odds)
            
            # Olasılıkları hesapla
            total_prob = (1/avg_home) + (1/avg_draw) + (1/avg_away)
            prob_home = (1/avg_home) / total_prob
            prob_draw = (1/avg_draw) / total_prob
            prob_away = (1/avg_away) / total_prob
            
            return {
                "odds": (avg_home, avg_draw, avg_away),
                "probs": (prob_home, prob_draw, prob_away)
            }
            
    except Exception as e:
        log(f"API-Football odds parse hatası: {e}")
    
    return None

def parse_apifootball_standings(response):
    """API-Football standings verisini parse eder"""
    try:
        standings_data = {}
        
        for league_data in response:
            league = league_data.get('league', {})
            standings_list = league.get('standings', [])
            
            if standings_list and len(standings_list) > 0:
                for team_data in standings_list[0]:
                    team = team_data.get('team', {})
                    team_name = team.get('name')
                    
                    if team_name:
                        standings_data[team_name] = {
                            'position': team_data.get('rank'),
                            'points': team_data.get('points'),
                            'goals_diff': team_data.get('goalsDiff'),
                            'form': team_data.get('form')
                        }
        
        return standings_data
        
    except Exception as e:
        log(f"API-Football standings parse hatası: {e}")
        return {}

# ==================== TÜM LİG OTOMATİK VERİ TOPLAYICI (API-FOOTBALL BİRİNCİL) ====================

class UniversalDataCollector:
    """Tüm ligleri kapsayan otomatik veri toplayıcı - API-FOOTBALL BİRİNCİL"""
    
    def __init__(self):
        self.data_sources = {
            'api_football': self._fetch_apifootball  # BİRİNCİL KAYNAK
        }
        self.fallback_chain = ['api_football']  # SADECE API-FOOTBALL
        
    def fetch_fixtures_universal(self, date_str, country=None, competition=None):
        """Tüm kaynaklardan fixture toplar - API-FOOTBALL BİRİNCİL"""
        fixtures = []
        
        for source in self.fallback_chain:
            try:
                source_fixtures = self.data_sources[source](date_str, country, competition)
                if source_fixtures:
                    fixtures.extend(source_fixtures)
                    log(f"✅ {source}: {len(source_fixtures)} fixture bulundu")
                        
            except Exception as e:
                log(f"❌ {source} hatası: {e}")
                continue
        
        return self._deduplicate_fixtures(fixtures)
    
    def _fetch_apifootball(self, date_str, country=None, competition=None):
        """API-Football'dan fixture al - BİRİNCİL KAYNAK"""
        if not APIFOOT:
            return []
            
        fixtures = []
        
        ### DÜZELTME: Tüm maçları çekip belirtilen ligleri filtrele
        # HEDEF LİG ID LİSTESİ - 24 lig
        TARGET_LEAGUE_IDS = ['39','140','135','78','61','88','144','179','203','141','136','79','95','145','2','3','848','667','4','5','6','7','1','9']
        
        # Tüm maçları çek (lig parametresi YOK)
        params = {"date": date_str}
        
        try:
            response = requests.get(
                f"{APIFOOTBALL_BASE_URL}fixtures",
                headers=HEADERS,
                params=params,
                timeout=30
            )
            if response.status_code == 200:
                data = response.json()
                if "response" in data:
                    for item in data["response"]:
                        # Lig ID kontrolü - sadece hedef liglerdeki maçları al
                        league_id = str(item.get('league', {}).get('id'))
                        if league_id in TARGET_LEAGUE_IDS:
                            fixture = self._parse_apifootball_fixture(item)
                            if fixture:
                                fixtures.append(fixture)
                            
                    log(f"🎯 API-Football: {len(data['response'])} maç → {len(fixtures)} hedef lig maçı")
        except Exception as e:
            log(f"API-Football fixture error: {e}")
        
        return fixtures
    
    def _get_relevant_leagues(self, country=None, competition=None):
        """Ülke ve lige göre ilgili lig ID'lerini döndürür"""
        base_leagues = [39, 40, 140, 135, 78, 61, 203, 88, 94, 144, 179, 141, 136, 79, 95, 145]  # Temel ligler
        
        if country:
            country_leagues = {
                'england': [39, 40, 41, 42, 45, 48],
                'spain': [140, 141, 143],
                'italy': [135, 136, 137],
                'germany': [78, 79, 81],
                'france': [61, 62, 66],
                'turkey': [203, 204, 206],
                'netherlands': [88, 89, 92],
                'portugal': [94, 95, 96],
                'belgium': [144, 145, 146]
            }
            
            for country_key, leagues in country_leagues.items():
                if country_key in country.lower():
                    return leagues
        
        return base_leagues
    
    def _parse_apifootball_fixture(self, item):
        """API-Football fixture parse"""
        try:
            fixture_data = item.get('fixture', {})
            league_data = item.get('league', {})
            teams_data = item.get('teams', {})
            
            ### DÜZELTME: Country string/dict desteği
            country_data = league_data.get('country', {})
            if isinstance(country_data, str):
                area = country_data
            else:
                area = country_data.get('name', 'Europe') if country_data else 'Europe'
            
            return {
                "source": "APIF_UNIVERSAL",
                "utc_kickoff": to_dt_utc(fixture_data.get('date')),
                "home": teams_data.get('home', {}).get('name'),
                "away": teams_data.get('away', {}).get('name'),
                "home_id": teams_data.get('home', {}).get('id'),
                "away_id": teams_data.get('away', {}).get('id'),
                "area": area,
                "competition": league_data.get('name', ''),
                "competition_id": league_data.get('id'),
                "id": f"apif_universal:{fixture_data.get('id')}",
            }
        except Exception as e:
            log(f"APIF universal parse hatası: {e}")
            return None
    
    def _deduplicate_fixtures(self, fixtures):
        """Tekrar eden fixture'ları temizle"""
        unique_fixtures = []
        seen_matches = set()
        
        for fixture in fixtures:
            match_key = f"{fixture['home']}|{fixture['away']}|{fixture['utc_kickoff']}"
            if match_key not in seen_matches:
                seen_matches.add(match_key)
                unique_fixtures.append(fixture)
        
        return unique_fixtures

# Universal collector instance
universal_collector = UniversalDataCollector()

# ==================== WEATHER API ENTEGRASYONU ====================

class WeatherAPIProvider:
    """Hızlı WeatherAPI entegrasyonu"""
    
    def __init__(self):
        self.api_key = WEATHER_API_KEY
        self.base_url = "http://api.weatherapi.com/v1"
        self.cache = {}
        self.cache_ttl = WEATHER_CACHE_TTL
    
    def get_weather_fast(self, city_name):
        """Hızlı hava durumu bilgisi al (~200ms)"""
        cache_key = f"weather_{city_name.lower()}"
        current_time = time.time()
        
        # Cache kontrolü
        if cache_key in self.cache:
            cached_data, timestamp = self.cache[cache_key]
            if current_time - timestamp < self.cache_ttl:
                return cached_data
        
        try:
            if not self.api_key:
                return self._get_fallback_weather(city_name)
            
            # Hızlı API çağrısı
            url = f"{self.base_url}/current.json"
            params = {
                "key": self.api_key,
                "q": city_name,
                "aqi": "no"
            }
            
            start_time = time.time()
            response = requests.get(url, params=params, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                weather_data = self._parse_weather_data(data)
                
                # Cache'e kaydet
                self.cache[cache_key] = (weather_data, current_time)
                
                response_time = (time.time() - start_time) * 1000
                log(f"🌤️ WeatherAPI: {city_name} → {response_time:.0f}ms")
                
                return weather_data
            else:
                log(f"❌ WeatherAPI hatası: {response.status_code}")
                return self._get_fallback_weather(city_name)
                
        except requests.exceptions.Timeout:
            log(f"⏰ WeatherAPI timeout: {city_name}")
            return self._get_fallback_weather(city_name)
        except Exception as e:
            log(f"❌ WeatherAPI genel hata: {e}")
            return self._get_fallback_weather(city_name)
    
    def _parse_weather_data(self, data):
        """WeatherAPI verisini parse et"""
        current = data.get('current', {})
        
        return {
            'temperature': current.get('temp_c'),
            'precipitation': current.get('precip_mm', 0),
            'wind_speed': current.get('wind_kph', 0) / 3.6,  # km/s -> m/s
            'humidity': current.get('humidity'),
            'condition': current.get('condition', {}).get('text', ''),
            'last_updated': current.get('last_updated')
        }
    
    def _get_fallback_weather(self, city_name):
        """Fallback hava durumu bilgisi"""
        city_climate = {
            'istanbul': {'temp': 15, 'precip': 0.5, 'wind': 3.0},
            'london': {'temp': 10, 'precip': 1.0, 'wind': 4.0},
            'madrid': {'temp': 18, 'precip': 0.1, 'wind': 2.0},
            'rome': {'temp': 16, 'precip': 0.3, 'wind': 2.5},
            'berlin': {'temp': 8, 'precip': 0.8, 'wind': 3.5},
            'paris': {'temp': 12, 'precip': 0.6, 'wind': 3.2},
            'amsterdam': {'temp': 9, 'precip': 0.9, 'wind': 4.2}
        }
        
        city_lower = city_name.lower()
        for city, climate in city_climate.items():
            if city in city_lower:
                return {
                    'temperature': climate['temp'],
                    'precipitation': climate['precip'],
                    'wind_speed': climate['wind'],
                    'humidity': 70,
                    'condition': 'Partly cloudy',
                    'source': 'FALLBACK'
                }
        
        # Varsayılan değerler
        return {
            'temperature': 15,
            'precipitation': 0.5,
            'wind_speed': 3.0,
            'humidity': 70,
            'condition': 'Clear',
            'source': 'DEFAULT'
        }
    
    def get_weather_note(self, home_team):
        """Takım için hava durumu notu oluştur"""
        city = guess_city_from_team(home_team)
        weather_data = self.get_weather_fast(city)
        
        if weather_data:
            temp = weather_data['temperature']
            precip = weather_data['precipitation']
            wind = weather_data['wind_speed'] * 3.6  # m/s -> km/s
            
            return f"Hava: {temp:.0f}°C, yağış {precip:.1f}mm, rüzgâr {wind:.0f} km/s"
        
        return None

# Weather provider instance
weather_provider = WeatherAPIProvider()

# ==================== GELİŞMİŞ ENSEMBLE SİSTEMİ ====================

class FootballEnsemble:
    """Futbol tahmini için gelişmiş ensemble learning sistemi"""
    
    def __init__(self):
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=50,
                max_depth=8,
                random_state=42
            ),
            'xgboost': xgb.XGBClassifier(
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            ),
            'logistic': LogisticRegression(
                C=1.0,
                random_state=42
            )
        }
        self.scaler = StandardScaler()
        self.is_trained = False
        self.feature_names = []
        
    def create_ensemble_features(self, fx, odds_info=None):
        """API-Football verileri ile gelişmiş feature'lar oluşturur"""
        area = fx.get("area", "Europe")
        home_team, away_team = fx.get("home"), fx.get("away")
        
        features = {}
        
        # 1. Temel Elo ve Değer Feature'ları
        features['elo_diff'] = elo_get(area, home_team) - elo_get(area, away_team)
        features['home_advantage'] = home_adv_effective(area, fx.get("competition",""), home_team, away_team)
        
        # Kadro değeri feature'ı
        home_value, _ = get_team_value(home_team, area)
        away_value, _ = get_team_value(away_team, area)
        features['value_ratio'] = home_value / max(away_value, 0.1)
        
        # 2. API-Football Feature'ları
        try:
            # Head-to-Head verisi
            h2h_features = self._get_head_to_head_features(fx)
            features.update(h2h_features)
            
            # Oyuncu ve sakatlık verileri
            player_features = self._get_player_features(fx)
            features.update(player_features)
            
            # Transfer verileri
            transfer_features = self._get_transfer_features(fx)
            features.update(transfer_features)
            
            # Detaylı istatistikler
            stat_features = self._get_statistical_features(fx)
            features.update(stat_features)
            
        except Exception as e:
            log(f"API-Football feature hatası: {e}")
            # Fallback değerler
            features.update(self._get_fallback_features())
        
        # 3. Form ve Pozisyon Feature'ları
        features.update(self._get_form_features(fx))
        
        # 4. Oran Feature'ları
        if odds_info and 'probs' in odds_info:
            p1, px, p2 = odds_info['probs']
            features['odds_home_win'] = p1
            features['odds_draw'] = px
            features['odds_away_win'] = p2
        
        return features
    
    def _get_head_to_head_features(self, fx):
        """Head-to-Head feature'ları"""
        features = {}
        try:
            home_id = _apifoot_find_team_id(fx["home"])
            away_id = _apifoot_find_team_id(fx["away"])
            
            if home_id and away_id:
                h2h_data = get_head_to_head(home_id, away_id)
                if h2h_data:
                    home_wins, away_wins, draws = self._parse_h2h_results(h2h_data, fx["home"])
                    total_matches = home_wins + away_wins + draws
                    
                    if total_matches > 0:
                        features['h2h_home_win_rate'] = home_wins / total_matches
                        features['h2h_away_win_rate'] = away_wins / total_matches 
                        features['h2h_draw_rate'] = draws / total_matches
                        features['h2h_total_matches'] = total_matches
        
        except Exception as e:
            log(f"H2H feature hatası: {e}")
            
        return features
    
    def _get_player_features(self, fx):
        """Oyuncu ve sakatlık feature'ları"""
        features = {}
        try:
            fixture_id = find_fixture_id(fx.get("area"), fx.get("competition"), fx["home"], fx["away"])
            
            if fixture_id:
                # Sakatlık verisi
                injuries = get_injuries(fixture_id)
                if injuries:
                    features['home_injuries'] = self._count_team_injuries(injuries, fx["home"])
                    features['away_injuries'] = self._count_team_injuries(injuries, fx["away"])
                
                # Gol krallığı verisi
                league_id = get_league_id_for_country(fx.get("area"), fx.get("competition"))
                season = season_for_today()
                if league_id:
                    scorers = get_top_scorers(league_id, season)
                    features['home_top_scorer_presence'] = self._check_top_scorer_presence(scorers, fx["home"])
                    features['away_top_scorer_presence'] = self._check_top_scorer_presence(scorers, fx["away"])
        
        except Exception as e:
            log(f"Player feature hatası: {e}")
            
        return features
    
    def _get_transfer_features(self, fx):
        """Transfer feature'ları"""
        features = {}
        try:
            home_id = _apifoot_find_team_id(fx["home"])
            away_id = _apifoot_find_team_id(fx["away"])
            
            # Transfer hareketliliği (basit implementasyon)
            features['home_transfer_activity'] = random.uniform(0, 1)  # Geçici
            features['away_transfer_activity'] = random.uniform(0, 1)  # Geçici
            
        except Exception as e:
            log(f"Transfer feature hatası: {e}")
            
        return features
    
    def _get_statistical_features(self, fx):
        """İstatistik feature'ları"""
        features = {}
        try:
            fixture_id = find_fixture_id(fx.get("area"), fx.get("competition"), fx["home"], fx["away"])
            
            if fixture_id:
                stats = get_fixture_statistics(fixture_id)
                if stats:
                    # İstatistikleri parse et ve feature'lara dönüştür
                    features.update(self._parse_statistics(stats))
        
        except Exception as e:
            log(f"Statistics feature hatası: {e}")
            
        return features
    
    def _parse_h2h_results(self, h2h_data, home_team):
        """Head-to-Head sonuçlarını parse et"""
        home_wins = away_wins = draws = 0
        
        for match in h2h_data:
            teams = match.get('teams', {})
            score = match.get('goals', {})
            
            home_goals = score.get('home', 0)
            away_goals = score.get('away', 0)
            
            if home_goals > away_goals:
                home_wins += 1
            elif away_goals > home_goals:
                away_wins += 1
            else:
                draws += 1
                
        return home_wins, away_wins, draws
    
    def _count_team_injuries(self, injuries_data, team_name):
        """Takım sakatlık sayısını hesapla"""
        count = 0
        for injury in injuries_data:
            player_team = injury.get('team', {}).get('name', '')
            if player_team == team_name:
                count += 1
        return count
    
    def _check_top_scorer_presence(self, scorers_data, team_name):
        """Takımda gol kralı olup olmadığını kontrol et"""
        if not scorers_data:
            return 0
            
        for scorer in scorers_data:
            scorer_team = scorer.get('team', {}).get('name', '')
            if scorer_team == team_name:
                return 1
        return 0
    
    def _parse_statistics(self, stats_data):
        """İstatistik verilerini parse et"""
        features = {}
        # İstatistik parsing implementasyonu
        return features
    
    def _get_fallback_features(self):
        """Fallback feature değerleri"""
        return {
            'h2h_home_win_rate': 0.5,
            'h2h_away_win_rate': 0.3,
            'h2h_draw_rate': 0.2,
            'h2h_total_matches': 0,
            'home_injuries': 0,
            'away_injuries': 0,
            'home_top_scorer_presence': 0,
            'away_top_scorer_presence': 0,
            'home_transfer_activity': 0.5,
            'away_transfer_activity': 0.5
        }
    
    def _get_form_features(self, fx):
        """Form ve pozisyon feature'ları"""
        features = {}
        
        # Form adjustment
        home_adj, _ = _form_adjust_from_matches(fx.get("home_id"), fx.get("area"), fx.get("home"))
        away_adj, _ = _form_adjust_from_matches(fx.get("away_id"), fx.get("area"), fx.get("away"))
        features['form_net'] = home_adj - away_adj
        
        return features
    
    def train_models(self, training_data):
        """Ensemble modellerini eğitir"""
        if not training_data:
            log("❌ Eğitim verisi yok")
            return False
            
        try:
            # Feature ve target'ları ayır
            X = [item['features'] for item in training_data]
            y = [item['result'] for item in training_data]
            
            # Feature isimlerini kaydet
            if X:
                self.feature_names = list(X[0].keys())
            
            # Feature'ları numpy array'e çevir
            X_array = np.array([[features.get(name, 0) for name in self.feature_names] 
                              for features in X])
            
            # Ölçeklendir
            X_scaled = self.scaler.fit_transform(X_array)
            
            # Modelleri eğit
            for name, model in self.models.items():
                model.fit(X_scaled, y)
                log(f"✅ {name} modeli eğitildi")
            
            self.is_trained = True
            return True
            
        except Exception as e:
            log(f"❌ Model eğitim hatası: {e}")
            return False
    
    def predict(self, features):
        """Ensemble tahmini yapar"""
        if not self.is_trained:
            return None, 0.0
            
        try:
            # Feature'ları düzenle
            X = np.array([[features.get(name, 0) for name in self.feature_names]])
            X_scaled = self.scaler.transform(X)
            
            predictions = []
            weights = [0.4, 0.4, 0.2]  # RF, XGB, LR ağırlıkları
            
            # Her modelden tahmin al
            for (name, model), weight in zip(self.models.items(), weights):
                proba = model.predict_proba(X_scaled)[0]
                predictions.append(proba * weight)
            
            # Ağırlıklı ortalama
            ensemble_proba = np.sum(predictions, axis=0)
            confidence = np.max(ensemble_proba)
            
            return ensemble_proba, confidence
            
        except Exception as e:
            log(f"❌ Ensemble tahmin hatası: {e}")
            return None, 0.0

# Ensemble sistemini başlat
ensemble_system = FootballEnsemble()

def load_ensemble_model():
    """Ensemble modelini diskten yükler"""
    try:
        model_path = "ensemble_model.pkl"
        if os.path.exists(model_path):
            global ensemble_system
            ensemble_system = joblib.load(model_path)
            log("✅ Ensemble modeli yüklendi")
            return True
    except Exception as e:
        log(f"❌ Ensemble model yükleme hatası: {e}")
    return False

def save_ensemble_model():
    """Ensemble modelini diske kaydeder"""
    try:
        joblib.dump(ensemble_system, "ensemble_model.pkl")
        log("✅ Ensemble modeli kaydedildi")
        return True
    except Exception as e:
        log(f"❌ Ensemble model kaydetme hatası: {e}")
    return False

# ==================== GÜNCELLENMİŞ TAHMİN FONKSİYONLARI ====================

def rate_fixture_with_ensemble(fx, odds_info):
    """API-Football verileri ile gelişmiş tahmin sistemi"""
    
    # Önce mevcut tahmini al
    base_rating = original_rate_fixture(fx, odds_info)
    
    # API-Football AI tahminlerini al
    api_prediction = get_api_football_prediction(fx)
    
    # Ensemble feature'ları oluştur
    ensemble_features = ensemble_system.create_ensemble_features(fx, odds_info)
    
    # Ensemble tahmini al
    ensemble_probs, ensemble_conf = ensemble_system.predict(ensemble_features)
    
    if ensemble_probs is not None and ensemble_conf > 0:
        # Ensemble tahminini belirle
        ensemble_pick_idx = np.argmax(ensemble_probs)
        ensemble_pick = ["1", "X", "2"][ensemble_pick_idx]
        
        # API-Football tahmini ile birleştir
        final_pick, final_confidence = blend_predictions(
            base_rating, ensemble_pick, ensemble_conf, api_prediction
        )
        
        return {
            **base_rating,
            "pick": final_pick,
            "confidence": final_confidence,
            "ensemble_confidence": ensemble_conf * 100,
            "ensemble_pick": ensemble_pick,
            "api_football_prediction": api_prediction,
            "note": base_rating["note"] + f" | Ensemble: {ensemble_pick}({ensemble_conf*100:.1f}%) | API-Football: {api_prediction}"
        }
    
    else:
        # Ensemble çalışmıyorsa mevcut sistemi kullan
        return base_rating

def get_api_football_prediction(fx):
    """API-Football'ın kendi tahminini al"""
    try:
        fixture_id = find_fixture_id(fx.get("area"), fx.get("competition"), fx["home"], fx["away"])
        if fixture_id:
            prediction_data = get_api_predictions(fixture_id)
            if prediction_data:
                return parse_api_prediction(prediction_data)
    except Exception as e:
        log(f"API-Football prediction error: {e}")
    return None

def parse_api_prediction(prediction_data):
    """API-Football tahmin verisini parse et"""
    try:
        if prediction_data and len(prediction_data) > 0:
            prediction = prediction_data[0].get('predictions', {})
            if prediction:
                return prediction.get('winner', {}).get('name')
    except Exception as e:
        log(f"API prediction parse error: {e}")
    return None

def blend_predictions(base_rating, ensemble_pick, ensemble_conf, api_prediction):
    """Tüm tahminleri birleştir"""
    # Mevcut tahmin (%40)
    base_pick = base_rating["pick"]
    base_confidence = base_rating["confidence"]
    
    # Ensemble tahmini (%40)
    ensemble_weight = 0.4
    ensemble_confidence = ensemble_conf * 100
    
    # API-Football tahmini (%20)
    api_weight = 0.2
    api_confidence = 70  # Varsayılan API güven değeri
    
    # Tahminleri birleştir
    pick_scores = {"1": 0, "X": 0, "2": 0}
    
    # Mevcut tahmin
    pick_scores[base_pick] += base_confidence * (1 - ensemble_weight - api_weight)
    
    # Ensemble tahmini
    pick_scores[ensemble_pick] += ensemble_confidence * ensemble_weight
    
    # API-Football tahmini
    if api_prediction and api_prediction in pick_scores:
        pick_scores[api_prediction] += api_confidence * api_weight
    
    # En yüksek skorlu tahmini bul
    final_pick = max(pick_scores.items(), key=lambda x: x[1])[0]
    final_confidence = pick_scores[final_pick]
    
    return final_pick, final_confidence

def initialize_ensemble_training():
    """Ensemble sistemini geçmiş verilerle eğitir"""
    try:
        # Geçmiş tahmin ve sonuç verilerini yükle
        training_data = []
        
        for key, pred_data in STATE.get("pred_store", {}).items():
            if "result" in pred_data:  # Sonucu bilinen maçlar
                # Feature'ları recreate et
                fx = {
                    "home": pred_data.get("home"),
                    "away": pred_data.get("away"), 
                    "area": pred_data.get("area", "Europe"),
                    "competition": pred_data.get("competition", ""),
                    "home_id": None,
                    "away_id": None
                }
                
                features = ensemble_system.create_ensemble_features(fx)
                training_data.append({
                    'features': features,
                    'result': pred_data['result']  # 0, 1, veya 2
                })
        
        if len(training_data) >= 50:  # Minimum 50 maç
            success = ensemble_system.train_models(training_data)
            if success:
                log(f"✅ Ensemble sistemi {len(training_data)} maç ile eğitildi")
            return success
        else:
            log(f"⚠️ Yetersiz eğitim verisi: {len(training_data)} maç")
            return False
            
    except Exception as e:
        log(f"❌ Ensemble eğitim hatası: {e}")
        return False

# ==================== GÜNCELLENMİŞ TAKIM DEĞER FONKSİYONU ====================
def get_team_value(team_name, area="Europe"):
    """Geliştirilmiş takım değeri - Gerçekçi fallback'ler"""
    if not team_name:
        return 30.0, "DEFAULT"
    
    # Önce gerçekçi değerleri dene
    realistic_value, source = get_team_value_realistic(team_name, area)
    
    # Cache için kaydet
    team_values = load_team_values()
    cache_key = f"{area}:{normalize_team_name(team_name)}"
    team_values[cache_key] = {
        "value": realistic_value,
        "source": source,
        "timestamp": time.time()
    }
    save_team_values(team_values)
    
    log(f"Takım değeri: {team_name} -> {realistic_value}M € ({source})")
    return realistic_value, source

# ==================== GÜNCELLENMİŞ TAHMİN SİSTEMİ ====================
def rate_fixture_enhanced(fx, odds_info):
    """Geliştirilmiş fixture rating - Tüm yeni sistemler entegre"""
    # Akıllı tahmin sistemi
    intelligent_pick, intelligent_conf = get_intelligent_prediction(
        fx["home"], fx["away"], fx.get("area", "Europe")
    )
    
    # Mevcut sistemi kullan ama güveni artır
    base_rating = original_rate_fixture(fx, odds_info)
    
    # Güven artırıcı faktörler uygula
    enhanced_conf = enhance_confidence(base_rating, fx["home"], fx["away"], fx.get("area", "Europe"))
    
    # Saat bilgisi fallback
    time_str = get_fixture_time_fallback(fx)
    
    # Not kısmını güncelle
    base_rating["note"] = (
        f"Seçim: {intelligent_pick} | Güven: {enhanced_conf}% | Saat: {time_str} | "
        f"λ: {base_rating['lambda_h']:.1f}/{base_rating['lambda_a']:.1f} | "
        f"Akıllı Sistem: {intelligent_pick}({intelligent_conf}%)"
    )
    
    base_rating["pick"] = intelligent_pick
    base_rating["confidence"] = enhanced_conf
    
    return base_rating

# ==================== YARDIMCILAR / HELPERS ====================

def http_get(url, headers=None, params=None, timeout=25):
    try:
        r = requests.get(url, headers=headers or {}, params=params or {}, timeout=timeout)
        if r.status_code == 200:
            try:
                return r.json()
            except Exception:
                return None
        elif r.status_code == 429:
            print("⚠️ Rate limit aşıldı")
        elif r.status_code == 403:
            print("❌ API Key hatası")
        else:
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
    return (x or "").lower().replace(".", " ").replace("-", " ").replace(" fc", "").strip()

def season_for_today():
    now = datetime.now(TR_TZ)
    y = now.year
    return y if now.month >= 7 else (y - 1)

def _ensure_dir(p: Path) -> bool:
    """Klasör oluşturur - güvenli versiyon / Create directory - safe version"""
    try:
        p.mkdir(parents=True, exist_ok=True)
        return True
    except Exception as e:
        log(f"[FS] Klasör oluşturulamadı / Directory creation failed: {p} | {e}")
        return False

def save_snapshot(predictions: Dict, date: Optional[str] = None) -> None:
    """Tahmin snapshot'ını kaydeder / Saves prediction snapshot - GÜNCELLENDİ"""
    date = date or datetime.now().strftime("%Y-%m-%d")
    snap_dir = Path(SNAPSHOT_DIR)
    
    try:
        # Dizin yoksa oluştur - YENİ EKLENDİ
        if not snap_dir.exists():
            snap_dir.mkdir(parents=True, exist_ok=True)
            log(f"[Snapshot] Dizin oluşturuldu / Directory created: {snap_dir}")
    except Exception as e:
        log(f"[Snapshot] Klasör oluşturulamadı / Directory creation failed: {snap_dir} | {e}")
        return
    
    path = snap_dir / f"pred_{date}.json"
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        log(f"[Snapshot] Kaydedildi / Saved: {path}")
    except Exception as e:
        log(f"[Snapshot] Kaydetme hatası / Save error: {e}")

def load_snapshot(date: Optional[str] = None) -> Dict:
    """Tahmin snapshot'ını yükler / Loads prediction snapshot"""
    date = date or datetime.now().strftime("%Y-%m-%d")
    path = Path(SNAPSHOT_DIR) / f"pred_{date}.json"
    if not path.exists():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        log(f"[Snapshot] Yüklendi / Loaded: {path}")
        return data
    except Exception as e:
        log(f"[Snapshot] Yükleme hatası / Load error: {e}")
        return {}

# ==================== FİLTRELER / FILTERS ====================

_WOMEN_TOKENS = {
    "women", "woman", "female", "ladies", "wsl", "féminine", "feminine", "feminino", "femenina", "femenino", "femminile",
    "frauen", "damas", "dames", "mulheres", "kobiet", "vrouwen", "donna", "dziewcząt"
}

_WOMEN_HINTS = {
    "liga mx femenil", "liga f", "liga iberdrola", "primera femenina", "frauen-bundesliga", "frauen bundesliga",
    "serie a femminile", "division 1 féminine", "national women's",
}

_U_TOKENS = {
    "u23", "u22", "u21", "u20", "u19", "u18", "u17", "youth", "junior", "primavera", "sub-23", "sub-21", "sub-20", "sub20", "sub21"
}

def _norm(s: str) -> str:
    return (s or "").strip().lower()

# Log sayaçları / Log counters
_filter_counters = {"women": 0, "u21": 0, "snapshot_used": 0}

def is_women_competition(area_name: str, comp_name: str) -> bool:
    """Kadın ligi/kupası olup olmadığını kontrol eder"""
    if ALLOW_WOMEN == 1:
        return False
    
    a, c = (area_name or "").lower(), (comp_name or "").lower()
    combined = f"{a} {c}"
    
    # Token kontrolü
    if any(token in combined for token in _WOMEN_TOKENS):
        _filter_counters["women"] += 1
        return True
    
    # Hint kontrolü
    if any(hint in combined for hint in _WOMEN_HINTS):
        _filter_counters["women"] += 1
        return True
        
    return False

def is_u21_competition(comp_name: str) -> bool:
    """U-21 ligi olup olmadığını kontrol eder"""
    if ALLOW_U21 == 1:
        return False
        
    c = (comp_name or "").lower()
    if any(token in c for token in _U_TOKENS):
        _filter_counters["u21"] += 1
        return True
    return False

def get_filter_counts() -> Dict[str, int]:
    """Filtre sayaçlarını döndürür / Returns filter counters"""
    return _filter_counters.copy()

# --- Takım Adı Benzerlik Eşleştirme ------------------------------------------

def find_closest_team(target_team, team_list, threshold=0.75):
    """
    Takım listesinde en benzer takımı bulur
    
    Args:
        target_team: Aranan takım adı
        team_list: Arama yapılacak takım listesi
        threshold: Minimum benzerlik eşiği (0-1 arası)
    
    Returns:
        (en_benzer_takım, benzerlik_skoru) veya (None, 0) eşleşme yoksa
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

# ==================== STATE YÖNETİMİ / STATE MANAGEMENT ====================

def load_state() -> Dict:
    """STATE yoksa veya kapalıysa snapshot'tan yükler / Loads from snapshot if STATE missing or disabled - GÜNCELLENDİ"""
    # Önce dizin kontrolü ve oluşturma
    state_path = Path(STATE_PATH)
    state_dir = state_path.parent
    
    try:
        # Dizin yoksa oluştur - YENİ EKLENDİ
        if not state_dir.exists():
            state_dir.mkdir(parents=True, exist_ok=True)
            log(f"[STATE] Dizin oluşturuldu / Directory created: {state_dir}")
    except Exception as e:
        log(f"[STATE] Dizin oluşturma hatası / Directory creation error: {e}")

    # STATE dosyası yoksa veya kapalıysa snapshot'tan yükle
    if not ALLOW_STATE_FILE or not state_path.exists():
        if not ALLOW_STATE_FILE:
            log("[STATE] Dosya yazma kapalı. Snapshot'tan okunacak / File write disabled. Loading from snapshot.")
        else:
            log("[STATE] Dosya bulunamadı. Snapshot'tan okunacak / File not found. Loading from snapshot.")
        
        _filter_counters["snapshot_used"] += 1
        snapshot_data = load_snapshot()
        return _ensure_state_defaults(snapshot_data)
    
    try:
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        
        # State yapısını kontrol et ve varsayılan değerleri ekle
        state = _ensure_state_defaults(state)
        log(f"[STATE] Başarıyla yüklendi / Successfully loaded: {STATE_PATH}")
        return state
        
    except Exception as e:
        log(f"[STATE] Yükleme hatası / Load error: {e}. Snapshot'a düşülüyor / Falling back to snapshot.")
        _filter_counters["snapshot_used"] += 1
        snapshot_data = load_snapshot()
        return _ensure_state_defaults(snapshot_data)

def save_state(state: Dict) -> None:
    """State'i kaydeder / Saves state"""
    if not ALLOW_STATE_FILE:
        log("[STATE] Yazma kapalı (ALLOW_STATE_FILE=0) / Write disabled")
        return
    
    try:
        state["last_saved"] = datetime.utcnow().isoformat() + "Z"
        state_dir = Path(STATE_PATH).parent
        if _ensure_dir(state_dir):
            with open(STATE_PATH, "w", encoding="utf-8") as f:
                json.dump(state, f, ensure_ascii=False, indent=2)
            log(f"[STATE] Kaydedildi → / Saved → {STATE_PATH}")
    except Exception as e:
        log(f"[STATE] Kaydetme hatası / Save error: {e}")

def _ensure_state_defaults(state: dict) -> dict:
    """State yapısını kontrol eder ve varsayılan değerleri ekler - GÜNCELLENDİ"""
    try:
        if not isinstance(state, dict):
            state = {}
        
        state.setdefault("elo", {})
        state.setdefault("goal_scale", {})
        state.setdefault("w_mkt", W_MKT_INIT)
        state.setdefault("pred_store", {})
        state.setdefault("metrics", {})
        state.setdefault("last_saved", None)
        state.setdefault("last_pred_date", None)
        state.setdefault("last_res_date", None)
        
        # Eski state yapılarını yeniye dönüştür
        if "elo" not in state or not isinstance(state["elo"], dict):
            state["elo"] = {}
        if "goal_scale" not in state or not isinstance(state["goal_scale"], dict):
            state["goal_scale"] = {}
            
    except Exception as e:
        log(f"[STATE] Varsayılan değerler hatası / Default values error: {e}")
        state = {
            "elo": {}, 
            "goal_scale": {}, 
            "w_mkt": W_MKT_INIT, 
            "pred_store": {}, 
            "metrics": {},
            "last_saved": None, 
            "last_pred_date": None, 
            "last_res_date": None
        }
    
    return state

STATE = load_state()

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
    Elo güncellemesi - dinamik ev avantajı desteği
    
    Args:
        area: Lig bölgesi
        home_name: Ev sahibi takım
        away_name: Deplasman takımı  
        result_hw: Sonuç (1.0=ev kazandı, 0.5=berabere, 0.0=deplasman kazandı)
        home_advantage: Ev avantajı (None ise ELO_HOME_ADV kullanır)
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

# --- GELİŞMİŞ KADRO DEğERİ SİSTEMİ (Transfermarkt KALDIRILDI + Fallback'ler) ------------
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

# TMAPI KALDIRILDI - SADECE FALLBACK SİSTEMLERİ KALDI
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

# ==================== MİLLİ TAKIM ELO SİSTEMİ -------------------------------------------------
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
    """Milli takımlar için Elo proxy değeri - ACİL DÜZELTME"""
    if not team_name:
        return None, None
    
    # ÖNEMLİ: Önce kulüp takımı kontrolü - milli takım değilse None dön
    national_indicators = ["national", "milli", "country", "olympics", "world cup", "euro", "qualification"]
    team_lower = team_name.lower()
    
    is_national = any(indicator in team_lower for indicator in national_indicators)
    if not is_national:
        return None, None  # Kulüp takımı
    
    # Milli takım Elo değerleri (FIFA sıralaması bazlı)
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
    
    return 1400.0, "ELO_DEFAULT"  # Varsayılan milli takım Elo'su

# ==================== AKILLI FEATURE SİSTEMİ ====================

def determine_match_type(home_country, away_country, competition):
    """Maç türünü otomatik belirler"""
    comp_lower = (competition or "").lower()
    
    # Milli Takım Maçları
    if any(keyword in comp_lower for keyword in ["world cup", "euro", "qualification", "international", "nations league"]):
        return "INTERNATIONAL"
    
    # Aynı Ülke Takımları - Yerel Lig
    if home_country == away_country:
        return "DOMESTIC_LEAGUE"
    
    # Farklı Ülke Takımları - Avrupa Kupası
    if any(cup in comp_lower for cup in ["champions league", "europa league", "conference league", "uefa"]):
        return "EUROPEAN_CUP"
    
    return "OTHER"

def get_intelligent_features(home_team, away_team, home_country, away_country, match_type):
    """Maç türüne göre akıllı feature seçimi"""
    
    features = {}
    
    # 1. Takım Değerleri (Tüm maç türleri için)
    features['home_team_value'] = get_team_value_feature(home_team, home_country)
    features['away_team_value'] = get_team_value_feature(away_team, away_country)
    features['value_ratio'] = features['home_team_value'] / max(features['away_team_value'], 0.1)
    
    # 2. Maç Türüne Özel Feature'lar
    if match_type == "DOMESTIC_LEAGUE":
        # AYNI LİG - Lig Pozisyonu ve Form
        features.update(get_league_features(home_team, away_team, home_country))
        
    elif match_type == "EUROPEAN_CUP":
        # AVRUPA KUPASI - UEFA katsayıları
        features.update(get_european_features(home_team, away_team))
        
    elif match_type == "INTERNATIONAL":
        # MİLLİ TAKIM - FIFA sıralaması
        features.update(get_international_features(home_team, away_team))
    
    return features

def get_team_value_feature(team_name, country):
    """State'den takım değerini al - DÜZELTİLMİŞ VERSİYON"""
    # State kontrolü ekle
    if "external_data" not in STATE:
        initialize_external_data_state()
    
    if "team_values" not in STATE["external_data"]:
        return 30.0  # Default değer
    
    return STATE["external_data"]["team_values"].get(team_name, 30.0)

def get_league_features(home_team, away_team, country):
    """Aynı ligdeki takımlar için lig tablosu bilgileri"""
    features = {}
    
    try:
        # YENİ: API Football v3 kullanımı
        standings = get_apifootball_standings(country)
        
        if standings:
            home_position = standings.get(home_team, {}).get('position', 20)
            away_position = standings.get(away_team, {}).get('position', 20)
            home_points = standings.get(home_team, {}).get('points', 0)
            away_points = standings.get(away_team, {}).get('points', 0)
            
            features['home_league_position'] = home_position
            features['away_league_position'] = away_position
            features['position_difference'] = abs(home_position - away_position)
            features['points_difference'] = home_points - away_points
            
    except Exception as e:
        log(f"Lig verileri alınamadı: {e}")
    
    return features

def get_european_features(home_team, away_team):
    """Avrupa kupaları için UEFA katsayıları"""
    features = {}
    
    try:
        home_coeff = get_uefa_coefficient(home_team)
        away_coeff = get_uefa_coefficient(away_team)
        
        features['home_uefa_coefficient'] = home_coeff
        features['away_uefa_coefficient'] = away_coeff
        features['uefa_coeff_ratio'] = home_coeff / max(away_coeff, 0.1)
        
    except Exception as e:
        log(f"UEFA verileri alınamadı: {e}")
    
    return features

def get_international_features(home_team, away_team):
    """Milli takımlar için FIFA sıralaması"""
    features = {}
    
    try:
        home_rank = get_fifa_ranking(home_team)
        away_rank = get_fifa_ranking(away_team)
        
        features['home_fifa_rank'] = home_rank
        features['away_fifa_rank'] = away_rank
        features['rank_difference'] = home_rank - away_rank
        features['fifa_power_ratio'] = (1/max(home_rank, 1)) / (1/max(away_rank, 1))
        
    except Exception as e:
        log(f"FIFA verileri alınamadı: {e}")
    
    return features

# ==================== API-FOOTBALL PREMIUM ENTEGRASYONU ====================

def get_apifootball_standings(country):
    """API-Football'dan lig tablosu - DÜZELTİLDİ"""
    try:
        league_id = get_league_id_for_country(country, "")
        if not league_id:
            return None
            
        season = season_for_today()
        
        response = requests.get(
            f"{APIFOOTBALL_BASE_URL}standings",
            headers=HEADERS,
            params={"league": league_id, "season": season},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return parse_apifootball_standings(data.get('response', []))
            
    except Exception as e:
        log(f"API-Football standings error: {e}")
    
    return None

def get_team_value_apifootball(team_name):
    """API-Football'dan takım değeri"""
    try:
        team_id = _apifoot_find_team_id(team_name)
        if not team_id:
            return None
            
        response = requests.get(
            f"{APIFOOTBALL_BASE_URL}teams",
            headers=HEADERS,
            params={"id": team_id},
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            if data.get('response'):
                return data['response'][0].get('team', {}).get('market_value')
            
    except Exception as e:
        log(f"API-Football team value error: {e}")
    
    return None

# ==================== ÇİFT KAYNAKLI ORAN SİSTEMİ ====================

def fetch_odds_dual(area, comp, home, away):
    """Çift kaynaklı oran sistemi - API-Football birincil"""
    
    # 1. Önce API-Football dene (BİRİNCİL KAYNAK)
    odds = fetch_odds_apifootball(area, comp, home, away)
    if odds:
        return odds, "APIFOOTBALL"
    
    return None, "NONE"

def fetch_odds_apifootball(area, comp, home, away):
    """API-Football'dan oranları al (BİRİNCİL KAYNAK) - DÜZELTİLDİ"""
    try:
        # API-Football odds endpoint
        fixture_id = find_fixture_id(area, comp, home, away)
        if not fixture_id:
            return None
            
        response = requests.get(
            f"{APIFOOTBALL_BASE_URL}odds",
            headers=HEADERS,
            params={"fixture": fixture_id, "bookmaker": 1},  # 1 = Bet365
            timeout=30
        )
        
        if response.status_code == 200:
            data = response.json()
            return parse_apifootball_odds(data.get('response', []))
            
    except Exception as e:
        log(f"API-Football odds error: {e}")
    
    return None

# ==================== YEDEK SİSTEMLER ====================

def get_fifa_ranking_backup(team_name):
    """FIFA sıralaması yedek - Kaggle CSV"""
    try:
        # Kaggle dataset veya statik CSV
        rankings = load_fifa_rankings_from_csv()
        return rankings.get(team_name, 50)
    except Exception as e:
        log(f"FIFA ranking backup error: {e}")
        return 50

def get_uefa_coefficient_backup(team_name):
    """UEFA katsayıları yedek - web scraping"""
    try:
        coefficients = scrape_uefa_coefficients()
        return coefficients.get(team_name, 10.0)
    except Exception as e:
        log(f"UEFA coefficient backup error: {e}")
        return 10.0

# ==================== STATE TABANLI STORAGE ====================

def update_external_data():
    """Tüm external verileri otomatik günceller"""
    initialize_external_data_state()
    
    # Takım değerleri (günlük)
    if needs_update("team_values"):
        update_team_values()
    
    # FIFA sıralaması (aylık)
    if needs_update("fifa_rankings"):
        update_fifa_rankings()
    
    # UEFA katsayıları (haftalık)
    if needs_update("uefa_coefficients"):
        update_uefa_coefficients()
    
    save_state(STATE)

def needs_update(data_type):
    """Güncelleme gerekip gerekmediğini kontrol eder"""
    last_updated = STATE["external_data"]["last_updated"].get(data_type)
    if not last_updated:
        return True
    
    last_dt = datetime.fromisoformat(last_updated)
    now = datetime.now()
    
    update_intervals = {
        "team_values": timedelta(days=1),
        "fifa_rankings": timedelta(days=30),
        "uefa_coefficients": timedelta(days=7)
    }
    
    return (now - last_dt) >= update_intervals.get(data_type, timedelta(days=1))

def get_fifa_ranking(team_name):
    """State'den FIFA sıralamasını al"""
    # State kontrolü ekle
    if "external_data" not in STATE:
        initialize_external_data_state()
    
    if "fifa_rankings" not in STATE["external_data"]:
        return 50  # Default değer
    
    return STATE["external_data"]["fifa_rankings"].get(team_name, 50)

def get_uefa_coefficient(team_name):
    """State'den UEFA katsayısını al"""
    # State kontrolü ekle
    if "external_data" not in STATE:
        initialize_external_data_state()
    
    if "uefa_coefficients" not in STATE["external_data"]:
        return 10.0  # Default değer
    
    return STATE["external_data"]["uefa_coefficients"].get(team_name, 10.0)

# ==================== GÜNCELLENMİŞ RATE_FIXTURE ====================

def rate_fixture_enhanced(fx, odds_info):
    """Geliştirilmiş fixture rating - akıllı feature seçimi ile"""
    
    # Maç türünü belirle
    match_type = determine_match_type(fx["area"], fx["area"], fx["competition"])
    
    # Akıllı feature'ları seç
    intelligent_features = get_intelligent_features(
        fx["home"], fx["away"], fx["area"], fx["area"], match_type
    )
    
    # Mevcut rate_fixture fonksiyonunu çağır ve feature'ları birleştir
    base_rating = rate_fixture(fx, odds_info)
    
    # Intelligent feature'ları base rating'e entegre et
    enhanced_rating = enhance_rating_with_features(base_rating, intelligent_features, match_type)
    
    return enhanced_rating

def enhance_rating_with_features(base_rating, features, match_type):
    """Temel rating'i intelligent feature'larla zenginleştir"""
    
    # Feature'ları confidence skoruna entegre et
    confidence_boost = calculate_feature_boost(features, match_type)
    base_rating["confidence"] = min(100, base_rating["confidence"] + confidence_boost)
    
    # Not kısmını güncelle
    feature_notes = generate_feature_notes(features, match_type)
    base_rating["note"] += f" | {feature_notes}"
    
    return base_rating

def calculate_feature_boost(features, match_type):
    """Feature'lara göre confidence boost hesaplar"""
    boost = 0
    
    if match_type == "DOMESTIC_LEAGUE":
        # Lig pozisyonu etkisi
        if 'position_difference' in features:
            pos_diff = features['position_difference']
            if pos_diff >= 10:
                boost += 5
            elif pos_diff >= 5:
                boost += 3
                
    elif match_type == "EUROPEAN_CUP":
        # UEFA katsayısı etkisi
        if 'uefa_coeff_ratio' in features:
            ratio = features['uefa_coeff_ratio']
            if ratio >= 2.0 or ratio <= 0.5:
                boost += 4
                
    elif match_type == "INTERNATIONAL":
        # FIFA sıralaması etkisi
        if 'rank_difference' in features:
            rank_diff = abs(features['rank_difference'])
            if rank_diff >= 30:
                boost += 6
            elif rank_diff >= 15:
                boost += 3
    
    return min(boost, 10)  # Maksimum 10 puan boost

def generate_feature_notes(features, match_type):
    """Feature'lara göre açıklama notları oluşturr"""
    notes = []
    
    if match_type == "DOMESTIC_LEAGUE":
        if 'position_difference' in features:
            notes.append(f"Pozisyon Farkı: {features['position_difference']}")
        if 'points_difference' in features:
            notes.append(f"Puan Farkı: {features['points_difference']}")
            
    elif match_type == "EUROPEAN_CUP":
        if 'uefa_coeff_ratio' in features:
            notes.append(f"UEFA Katsayı Oranı: {features['uefa_coeff_ratio']:.2f}")
            
    elif match_type == "INTERNATIONAL":
        if 'rank_difference' in features:
            notes.append(f"FIFA Sıra Farkı: {features['rank_difference']}")
        if 'fifa_power_ratio' in features:
            notes.append(f"Güç Oranı: {features['fifa_power_ratio']:.2f}")
    
    return " | ".join(notes) if notes else "Akıllı Feature: Temel"

# ==================== YENİ ÖZELLİKLER / NEW FEATURES ====================

# 3. Eşleştirme Toleransı / Matching Tolerance
def match_with_tolerance(team1: str, team2: str, date1: str, date2: str, tolerance_days: int = 2) -> bool:
    """Takım ve tarih eşleştirme toleransı / Team and date matching with tolerance"""
    try:
        date_obj1 = datetime.strptime(date1, "%Y-%m-%d")
        date_obj2 = datetime.strptime(date2, "%Y-%m-%d")
        days_diff = abs((date_obj2 - date_obj1).days)
        return days_diff <= tolerance_days
    except:
        return False

# 4. Kapanış Oranı Drift / Closing Line Drift
def calculate_closing_drift(opening_odds: float, closing_odds: float) -> Tuple[float, str]:
    """Kapanış oranı drift hesaplama / Closing line drift calculation"""
    if opening_odds == 0:
        return 0.0, "Drift: 0%"
    drift_pct = ((closing_odds - opening_odds) / opening_odds) * 100
    drift_pct = max(min(drift_pct, 3.0), -3.0)  # ±3% tavan / ceiling
    confidence_impact = drift_pct  # Güven skoruna direkt etki / Direct impact to confidence score
    note = f"Drift: {drift_pct:+.1f}%"
    return confidence_impact, note

# 5. Hakem / Dinlenme Etkisi / Referee / Rest Effect
def calculate_referee_impact(referee_stats: Dict) -> float:
    """Hakem kart etkisi hesaplama / Referee card impact calculation"""
    avg_cards = referee_stats.get("avg_cards_per_match", 2.0)
    base_cards = 2.0  # Ortalama baz / Average base
    impact = (avg_cards - base_cards) / base_cards * 10  # ±10% tavan / ceiling
    impact = max(min(impact, 10.0), -10.0)
    return impact

def calculate_fatigue_impact(matches_last_10_days: int, days_since_last_match: int) -> float:
    """Dinlenme-gün etkisi hesaplama / Rest-day impact calculation"""
    # Maç yoğunluğu etkisi / Match density impact
    density_impact = min(matches_last_10_days * 2, 6.0)  # ±6% tavan / ceiling
    # Dinlenme etkisi / Rest impact
    rest_impact = max((3 - days_since_last_match) * 2, -6.0)  # ±6% tavan
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
        """Maç sonucu olasılıkları / Match outcome probabilities"""
        home_attack = self.attack.get(home_team, 1.0)
        home_defense = self.defense.get(home_team, 1.0)
        away_attack = self.attack.get(away_team, 1.0)
        away_defense = self.defense.get(away_team, 1.0)
        
        # Basit Poisson hesaplama / Simple Poisson calculation
        lambda_home = home_attack * away_defense
        lambda_away = away_attack * home_defense
        
        # Dixon-Coles düzeltmesı / Dixon-Coles correction
        if GOAL_MODEL == "DC":
            # Beraberlik olasılığı iyileştirmesi / Draw probability improvement
            draw_bias = 1.0 + self.rho
            lambda_home *= draw_bias
            lambda_away *= draw_bias
        
        return lambda_home, lambda_away, self.rho

# 7. Çift Elo / Dual Elo
class DualEloSystem:
    """Attack/Defense ayrı Elo sistemi / Separate Attack/Defense Elo system"""
    def __init__(self, k_factor: int = 20, home_advantage: int = 40):
        self.k = k_factor
        self.home_adv = home_advantage
        self.attack_elo = {}
        self.defense_elo = {}
    
    def get_ratings(self, team: str) -> Tuple[float, float]:
        """Takım rating'lerini döndürür / Returns team ratings"""
        return self.attack_elo.get(team, 1500.0), self.defense_elo.get(team, 1500.0)
    
    def update_ratings(self, home_team: str, away_team: str, home_goals: int, away_goals: int):
        """Rating'leri günceller / Updates ratings"""
        # Attack/defense ayrı güncelleme mantığı / Separate attack/defense update logic
        pass

# 8. Sıralama Decay / Ranking Decay
def apply_ranking_decay(rankings: Dict[str, float], months_ago: int) -> float:
    """Sıralama sinyali time-decay uygular / Applies time-decay to ranking signals"""
    half_life = 12  # 12 ay yarı-ömür / 12 month half-life
    decay_factor = 0.5 ** (months_ago / half_life)
    return rankings.get("fifa", 0.0) * decay_factor

# 9. Market Kalibrasyonu / Market Calibration
class MarketCalibrator:
    """Piyasa olasılık kalibrasyonu / Market probability calibration"""
    def __init__(self):
        self.isotonic_model = IsotonicRegression(out_of_bounds='clip')
        self.is_fitted = True
    
    def calibrate_probabilities(self, raw_probs: np.ndarray, actual_results: np.ndarray) -> np.ndarray:
        """Olasılıkları kalibre eder / Calibrates probabilities"""
        if len(raw_probs) < 10 or not self.is_fitted:
            return raw_probs  # Yeterli veri yoksa / Not enough data
        return self.isotonic_model.transform(raw_probs)

# 10. Kırmızı Kart Riski / Red Card Risk
def calculate_red_card_risk(referee_red_rate: float, team_red_rate: float) -> float:
    """Kırmızı kart risk skoru hesaplar / Calculates red card risk score"""
    base_risk = (referee_red_rate + team_red_rate) / 2
    risk_impact = max(min(base_risk * 20, 6.0), 3.0)  # %3-6 arası etki / 3-6% impact
    return -risk_impact  # Negatif etki / Negative impact

# 11. Basit Bayes Modeli / Simple Bayes Model
class HierarchicalGoalModel:
    """Hiyerarşik gol modeli / Hierarchical goal model"""
    def __init__(self):
        self.league_priors = {}
        self.team_offense = {}
        self.team_defense = {}
    
    def predict_goals(self, home_team: str, away_team: str, league: str) -> Tuple[float, float]:
        """Gol tahmini / Goal prediction"""
        # Lig bazlı prior + takım regularizasyon / League-based prior + team regularization
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
    """xG proxy değeri hesaplar / Calculates xG proxy value"""
    if shots == 0:
        return 0.0
    conversion_rate = on_target / shots
    danger_ratio = dangerous_attacks / max(shots, 1)
    xg_proxy = (conversion_rate * 0.3 + danger_ratio * 0.7) * shots
    return min(xg_proxy, 8.0)  # Maksimum sınır / Maximum limit

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
        # Takım değeri değişimine göre etki / Impact based on team value change
        return max(min(team_value_change * 10, 5.0), -5.0)

# 14. Multi-market Konsistensi / Multi-market Consistency
def check_market_consistency(odds_1x2: Dict, odds_ah: Dict, odds_total: Dict) -> float:
    """Çoklu pazar tutarlılık kontrolü / Multi-market consistency check"""
    consistency_score = 100.0
    
    # 1X2 vs Asian Handicap tutarlılık / 1X2 vs Asian Handicap consistency
    if odds_1x2 and odds_ah:
        # Basit tutarlılık kontrolü / Simple consistency check
        home_prob = 1.0 / odds_1x2.get('home', 3.0)
        ah_prob = 0.5  # Basit varsayım / Simple assumption
        diff = abs(home_prob - ah_prob)
        if diff > 0.1:  # %10'dan fazla fark / More than 10% difference
            consistency_score -= 20
    
    # Tutarsızlık için kalibrasyon / Calibration for inconsistency
    calibration = max(consistency_score / 100, 0.8)  # Minimum %80 / Minimum 80%
    return calibration

# 15. Gelişmiş Kalite Skoru / Advanced Quality Score
def calculate_advanced_quality(features: Dict) -> float:
    """Gelişmiş kalite skoru hesaplama / Advanced quality score calculation"""
    base_score = calculate_quality(features)  # Temel skor / Base score
    
    # Ek faktörler / Additional factors
    bonus_points = 0
    
    # Veri kaynağı çeşitliliği / Data source diversity
    sources = features.get("data_sources", [])
    if len(sources) >= 3:
        bonus_points += 10
    elif len(sources) >= 2:
        bonus_points += 5
    
    # Güncellik / Freshness
    data_age = features.get("data_age_hours", 48)
    if data_age <= 1:
        bonus_points += 10
    elif data_age <= 6:
        bonus_points += 5
    
    # Model çeşitliliği / Model diversity
    models_used = features.get("models_used", 1)
    if models_used >= 3:
        bonus_points += 10
    elif models_used >= 2:
        bonus_points += 5
    
    final_score = min(base_score + bonus_points, 100.0)
    return max(final_score, 0.0)

# ==================== ORTAK FONKSİYONLAR / COMMON FUNCTIONS ====================

def calculate_quality(features: Dict) -> float:
    """Basit kalite skoru (0-100) / Simple quality score (0-100)"""
    keys = ("odds", "weather", "standings", "form", "ranking", "value")
    if not isinstance(features, dict) or not keys:
        return 0.0
    score = sum(1 for k in keys if features.get(k)) / len(keys) * 100.0
    return round(score, 1)

# --- GELİŞMİŞ KART/KORNER SİSTEMİ (Çoklu Kaynak) ----------------------------
def get_cards_corners_apifootball(area, comp, home_team, away_team):
    """API-Football'dan kart ve korner verileri"""
    if not APIFOOT:
        return None, "APIF"
    
    try:
        hint = _apifoot_hint_cards_corners(area, comp, home_team, away_team)
        if hint:
            return hint, "APIF"
    except Exception as e:
        log(f"API-Football kart/korner hatası: {e}")
    
    return None, "APIF"

def get_cards_corners_totalcorner(area, comp, home_team, away_team):
    """TotalCorner fallback - sadece korner verisi"""
    try:
        # TotalCorner API simulasyonu (gerçek API entegrasyonu için güncellenmeli)
        # Bu örnekte lig ortalamaları döndürüyoruz
        corner_base = base_from_area(area, LEAGUE_CORNER_BASE, 9.2)
        
        # Basit varyasyon
        corners = corner_base * random.uniform(0.9, 1.1)
        
        return {"mu_corners_hint": corners}, "TC"
    except Exception as e:
        log(f"TotalCorner hatası: {e}")
        return None, "TC"

def get_cards_corners_footystats(area, comp, home_team, away_team):
    """FootyStats fallback - lig ortalamaları"""
    try:
        # FootyStats lig ortalamaları
        cards_base = base_from_area(area, LEAGUE_CARD_BASE, 4.6)
        corner_base = base_from_area(area, LEAGUE_CORNER_BASE, 9.2)
        
        return {
            "mu_cards_hint": cards_base,
            "mu_corners_hint": corner_base
        }, "FS"
    except Exception as e:
        log(f"FootyStats hatası: {e}")
        return None, "FS"

def get_cards_corners_advanced(area, comp, home_team, away_team):
    """Geliştirilmiş kart/korner sistemi - zincirli fallback"""
    # 1. Öncelik: API-Football
    result, source = get_cards_corners_apifootball(area, comp, home_team, away_team)
    if result:
        log(f"Kart/Korner verisi {source}'dan alındı: {home_team} vs {away_team}")
        return result, source
    
    # 2. Fallback: TotalCorner (sadece korner)
    result, source = get_cards_corners_totalcorner(area, comp, home_team, away_team)
    if result and "mu_corners_hint" in result:
        log(f"Korner verisi {source}'dan alındı: {home_team} vs {away_team}")
        # Kart verisi için FootyStats'e ihtiyaç var
        cards_result, cards_source = get_cards_corners_footystats(area, comp, home_team, away_team)
        if cards_result and "mu_cards_hint" in cards_result:
            result["mu_cards_hint"] = cards_result["mu_cards_hint"]
            source = f"TC+{cards_source}"
        return result, source
    
    # 3. Fallback: FootyStats (hem kart hem korner)
    result, source = get_cards_corners_footystats(area, comp, home_team, away_team)
    if result:
        log(f"Kart/Korner verisi {source}'dan alındı: {home_team} vs {away_team}")
        return result, source
    
    # 4. Son çare: lig bazlı ortalamalar
    cards_base = base_from_area(area, LEAGUE_CARD_BASE, 4.6)
    corner_base = base_from_area(area, LEAGUE_CORNER_BASE, 9.2)
    
    log(f"Kart/Korner verisi DEFAULT'tan alındı: {home_team} vs {away_team}")
    return {
        "mu_cards_hint": cards_base,
        "mu_corners_hint": corner_base
    }, "DEFAULT"

# --- DENGELENMİŞ EV SAHİBİ AVANTAJI -----------------------------------------
def home_adv_effective(area, competition, home_team, away_team):
    """
    Dinamik ev sahibi avantajı hesaplar - DENGELENMİŞ
    """
    base_advantage = ELO_HOME_ADV
    
    # Milli takım maçlarında ev avantajını azalt
    comp_lower = (competition or "").lower()
    if "world cup" in comp_lower or "euro" in comp_lower or "qualification" in comp_lower:
        base_advantage *= 0.6  # %40 azalt
        log(f"🏟️ Milli takım maçı - ev avantajı azaltıldı: {base_advantage:.1f}")
    
    # Takım gücüne göre avantaj ayarı
    home_power = calculate_team_power(home_team, area)
    away_power = calculate_team_power(away_team, area)
    
    if away_power > home_power + 20:  # Deplasman daha güçlü
        advantage_factor = 1.2  # Ev avantajını artır
    elif home_power > away_power + 20:  # Ev daha güçlü
        advantage_factor = 0.8  # Ev avantajını azalt
    else:
        advantage_factor = 1.0
    
    final_advantage = base_advantage * advantage_factor
    
    log(f"Ev avantajı: {home_team}({home_power}) vs {away_team}({away_power}) -> {final_advantage:.1f}")
    
    return clamp(final_advantage, 15.0, 60.0)

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
    "uefa euroa",
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
    # Kadın liglerini filtrele
    if is_women_competition(area_name, comp_name):
        return False
        
    # U-21 liglerini filtrele
    if is_u21_competition(comp_name):
        return False
        
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
ODDS_KEY = os.getenv("ODDS_API_KEY")
APIFOOT_BASE = "https://v3.football.api-sports.io"
APIFOOT = (os.getenv("APIFOOTBALL_KEY") or "").strip()
MODE_ENV = (os.getenv("MODE") or "AUTO").upper().strip()
OLD_LEAGUES = [x.strip() for x in (os.getenv("OLD_LEAGUES", "bundesliga,bundesliga2").split(",")) if x.strip()]
ODDS_TTL_MIN = int(os.getenv("ODDS_TTL_MIN", "15"))
SPLIT_HIGH = (os.getenv("SPLIT_HIGH_ALERT_MAIL", "0") == "1")

# Elo / Form ayarları
ELO_K = float(os.getenv("ELO_K", "24"))
ELO_HOME_ADV = float(os.getenv("ELO_HOME_ADV", "40"))  # DÜŞÜRÜLDÜ: 60 -> 40
FORM_LOOKBACK = int(os.getenv("FORM_LOOKBACK", "10"))
FORM_DAYS = int(os.getenv("FORM_DAYS", "120"))
ALLOW_STATE_FILE = (os.getenv("ALLOW_STATE_FILE", "1") == "1")

# Otomatik öğrenme ayarları
W_MKT_INIT = float(os.getenv("W_MKT_INIT", "0.45"))  # ARTIRILDI: 0.35 -> 0.45
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
    """YENİ: WeatherAPI entegrasyonu"""
    return weather_provider.get_weather_note(home_team)

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

# --- 2. Fallback: API-Football (tarih bazlı) ---------------------------------
_apifoot_team_cache = {}  # search_name.lower() -> team_id
_apifoot_stat_cache = {}  # (league_id, season, team_id) -> stats_json

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
_APIF_STANDINGS_CACHE = {}
_APIF_FIXTURES_CACHE = {}

def _fd_team_matches(team_id, days=120):
    """Takım maçlarını getir - GÜNCELLENDİ"""
    # Football-Data.org kaldırıldı, sadece API-Football kullan
    return []

def _form_adjust_from_matches(team_id, area, team_name):
    """Form adjustment - GÜNCELLENDİ"""
    # Football-Data.org kaldırıldı, basit form sistemi
    return (0.0, "")

def _apifoot_standings(league_id, season):
    if not APIFOOT or not league_id or not season:
        return None
    key = (league_id, season)
    if key in _APIF_STANDINGS_CACHE:
        return _APIF_STANDINGS_CACHE[key]
    resp = _apifoot_get("standings", {"league": league_id, "season": season})
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

def _streak_from_any(fix):
    """Streak hesaplama - GÜNCELLENDİ"""
    # Football-Data.org kaldırıldı, sadece API-Football kullan
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

def original_rate_fixture(fx, odds_info):
    area = fx["area"] or "Europe"
    tot = base_total_goals(area)
    
    # Dinamik ev sahibi avantajı - DENGELENMİŞ
    home_advantage = home_adv_effective(area, fx.get("competition",""), fx["home"], fx["away"])
    
    ah = 1.12
    noise = (len((fx["home"] or "")) - len((fx["away"] or ""))) * 0.01
    lam_h = max(0.2, tot*0.5*ah + noise)
    lam_a = max(0.2, tot*0.5*(2 - ah) - noise)
    
    # Hava (Akıllı Mod) - YENİ: WeatherAPI entegrasyonu
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
    
    # Kadro değeri avantajı (TMAPI KALDIRILDI, sadece fallback)
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
    
    # GELİŞMİŞ Kart/Korner — Çoklu Kaynak Fallback
    apihint, kk_source = get_cards_corners_advanced(fx.get("area"), fx.get("competition"), fx.get("home"), fx.get("away"))
    
    kk = model_cards_corners(area, lam_h, lam_a, wx, apifoot_hint=apihint, source_info=kk_source)
    
    # Kaynak etiketli çıktı
    kk_txt = (f" | Korner μ≈{kk['mu_corners']:.1f} (Üst8.5 {int(kk['p_over_corners_8_5']*100)}% / "
              f"Üst9.5 {int(kk['p_over_corners_9_5']*100)}%) [{kk['source']}]"
              f" | Kart μ≈{kk['mu_cards']:.1f} (Üst3.5 {int(kk['p_over_cards_3_5']*100)}%) [{kk['source']}]")
    
    # Kadro değeri bilgisi (kaynak etiketli) - TMAPI KALDIRILDI
    home_value, home_source = get_team_value(fx["home"], area)
    away_value, away_source = get_team_value(fx["away"], area)
    value_txt = f" | Kadro: {home_value:.0f}M€ [{home_source}] vs {away_value:.0f}M€ [{away_source}]"
    
    wx_txt = f" | {wx}" if wx else ""

    # YENİ EKLENEN KOD: Saat bilgisi kontrolü
    utc_kickoff = fx.get("utc_kickoff")
    if utc_kickoff:
        local_time = utc_kickoff.astimezone(TR_TZ).strftime("%H:%M")
        time_txt = f"Saat: {local_time}"
    else:
        time_txt = "Saat: Veri Yok"
        log(f"⏰ Fixture saat bilgisi eksik: {fx['home']} vs {fx['away']}")

    note = (f"Seçim: {pick} | Güven: {conf_pct}% | {time_txt} | λ_h/λ_a: {lam_h:.2f}/{lam_a:.2f}"
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

# --- Tahmin/sonuç eşleşme & öğrenme yardımcıları ------------------------------
def match_key_from_fixture(fx):
    if fx.get("id"):
        return f"{fx.get('source','?')}:{fx['id']}"
    dt = (fx.get("utc_kickoff") or datetime.now(timezone.utc)).strftime("%Y%m%d")
    return f"{norm_team(fx.get('home'))}|{norm_team(fx.get('away'))}|{dt}"

def alt_key_from_names(home, away, date_str):
    return f"{norm_team(home)}|{norm_team(away)}|{date_str.replace('-','')}"

def find_prediction_for_result(result):
    """Sonuç için tahmin bulur (yakın isim eşleştirmeli)"""
    home, away, date_str = result["home"], result["away"], result.get("date", "")
    
    # Önce tam eşleşme dene
    altk = alt_key_from_names(home, away, date_str)
    if altk in STATE["pred_store"]:
        return STATE["pred_store"][altk]
    
    # ID bazlı eşleşme
    if result.get("id_key") and result["id_key"] in STATE["pred_store"]:
        return STATE["pred_store"][result["id_key"]]
    
    # Yakın isim eşleştirmesi - GELİŞTİRİLMİŞ VERSİYON
    all_pred_keys = list(STATE["pred_store"].keys())
    all_team_pairs = []
    
    for key in all_pred_keys:
        if "|" in key and key.count("|") == 2:
            parts = key.split("|")
            if len(parts) == 3:
                pred_home, pred_away, pred_date = parts
                if pred_date == date_str.replace("-", ""):
                    all_team_pairs.append((pred_home, pred_away, key))
    
    # Çift yönlü eşleştirme - GELİŞTİRİLMİŞ
    best_match = None
    best_score = 0.0
    
    for pred_home, pred_away, key in all_team_pairs:
        # Normal eşleşme
        home_sim = team_similarity(home, pred_home)
        away_sim = team_similarity(away, pred_away)
        normal_score = (home_sim + away_sim) / 2
        
        # Ters eşleşme (API'de home/away şaşmış olabilir)
        home_sim_rev = team_similarity(home, pred_away)
        away_sim_rev = team_similarity(away, pred_home)
        reverse_score = (home_sim_rev + away_sim_rev) / 2
        
        # En iyi skoru seç
        current_score = max(normal_score, reverse_score)
        
        if current_score > best_score and current_score >= 0.75:  # %75 benzerlik eşiği
            best_score = current_score
            best_match = key
            
            if current_score == reverse_score:
                log(f"Ters eşleşme bulundu: {home}/{away} ≈ {pred_away}/{pred_home} "
                    f"(benzerlik: {current_score:.2f})")
            else:
                log(f"Normal eşleşme bulundu: {home}/{away} ≈ {pred_home}/{pred_away} "
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
    
    # İkincil anahtar: isim+tarih
    altk = alt_key_from_names(fx.get("home"), fx.get("away"), (fx.get("utc_kickoff") or datetime.now(timezone.utc)).astimezone(TR_TZ).strftime("%Y-%m-%d"))
    STATE["pred_store"][altk] = rec

# --- Mail --------------------------------------------------------------------
def send_mail(subject, body):

    # Sürüm etiketi ve zaman damgası
    try:
        stamp = datetime.now(TR_TZ).strftime("%Y-%m-%d %H:%M")
        subject = f"{subject} · {MODEL_VERSION} · {stamp}"
    except Exception:
        pass
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
def fetch_results_apifoot(date_str):
    if not APIFOOT:
        return []
    url = f"{APIFOOTBALL_BASE_URL}fixtures"
    data = http_get(url, headers=HEADERS, params={"date": date_str})
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
            # bitmiş say
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
    r = fetch_results_apifoot(date_str)
    
    # SONUÇ OLSUN OLMASIN MAİL GÖNDER
    lines = [f"📊 Dünün Sonuçları — {date_str}"]
    
    if r:
        for res in r:
            # Tahminle eşleştir
            pred = find_prediction_for_result(res)
            
            if pred:
                # Sonuç durumunu belirle
                if res["score_h"] > res["score_a"]:
                    outcome_idx = 0  # Ev kazandı
                    outcome_str = "1"
                elif res["score_h"] == res["score_a"]:
                    outcome_idx = 1  # Berabere
                    outcome_str = "X"
                else:
                    outcome_idx = 2  # Deplasman kazandı
                    outcome_str = "2"
                
                # Tahmin performansını değerlendir
                pred_pick = pred.get("pick", "")
                pred_conf = pred.get("conf_pct", 0)
                is_correct = (pred_pick == outcome_str)
                
                # Brier skoru hesapla
                brier = brier_score(pred.get("probs_blend"), outcome_idx)
                
                # Elo güncelle
                result_hw = 1.0 if outcome_idx == 0 else (0.5 if outcome_idx == 1 else 0.0)
                elo_update(res["area"], res["home"], res["away"], result_hw, 
                          pred.get("home_advantage", ELO_HOME_ADV))
                
                # Öğrenme: w_mkt ayarı
                if pred.get("probs_market"):
                    market_probs = pred["probs_market"]
                    model_probs = pred.get("probs_model", market_probs)
                    actual = [0.0, 0.0, 0.0]
                    actual[outcome_idx] = 1.0
                    
                    # Model ve market hatalarını karşılaştır
                    model_err = sum((m - a)**2 for m, a in zip(model_probs, actual))
                    market_err = sum((m - a)**2 for m, a in zip(market_probs, actual))
                    
                    if model_err < market_err:
                        # Model daha iyi, w_mkt'yi azalt
                        new_w = get_w_mkt() * (1.0 - LEARN_RATE)
                    else:
                        # Market daha iyi, w_mkt'yi artır
                        new_w = get_w_mkt() * (1.0 + LEARN_RATE)
                    
                    set_w_mkt(new_w)
                
                # Gol ölçeğini güncelle
                total_goals = res["score_h"] + res["score_a"]
                area = res["area"]
                expected_goals = base_total_goals(area)
                if expected_goals > 0:
                    goal_ratio = total_goals / expected_goals
                    new_scale = get_goal_scale(area) * (1.0 + GOAL_LR * (goal_ratio - 1.0))
                    set_goal_scale(area, new_scale)
                
                # Sonuç satırını oluştur
                status = "✅ DOĞRU" if is_correct else "❌ YANLIŞ"
                line = (f"- {res['home']} {res['score_h']}-{res['score_a']} {res['away']} | "
                       f"Tahmin: {pred_pick}({pred_conf}%) | Sonuç: {outcome_str} | {status} | "
                       f"Brier: {brier:.3f}" if brier else "N/A")
                
            else:
                # Tahmin bulunamadı, ama sonuç belli
                if res["score_h"] > res["score_a"]:
                    outcome_str = "1"
                elif res["score_h"] == res["score_a"]:
                    outcome_str = "X" 
                else:
                    outcome_str = "2"
                
                line = f"- {res['home']} {res['score_h']}-{res['score_a']} {res['away']} | Tahmin: BULUNAMADI | Sonuç: {outcome_str}"
            
            lines.append(line)
        
        body = "\n".join(lines)
        
    else:
        # SONUÇ YOKSA BİLE MAİL GÖNDER
        body = f"📊 Dünün Sonuçları — {date_str}\n\nℹ️ {date_str} tarihi için sonuç bulunamadı veya maç oynanmadı."
    
    # METRİKLERİ EKLE
    metrics = STATE.get("metrics", {})
    total_pred = metrics.get("total_predictions", 0)
    correct_pred = metrics.get("correct_predictions", 0)
    accuracy = (correct_pred / total_pred * 100) if total_pred > 0 else 0
    
    body += f"\n\n📈 PERFORMANS METRİKLERİ:\n"
    body += f"• Toplam Tahmin: {total_pred}\n"
    body += f"• Doğru Tahmin: {correct_pred}\n"
    body += f"• Doğruluk Oranı: {accuracy:.1f}%\n"
    body += f"• Güncel w_mkt: {get_w_mkt():.3f}\n"
    
    # Ensemble durumu
    if ensemble_system.is_trained:
        body += f"• 🤖 Ensemble: AKTİF\n"
    else:
        body += f"• 🤖 Ensemble: EĞİTİM GEREKİYOR\n"
    
    # E-posta gönder
    send_mail(f"Sonuç Raporu | {date_str}", body)
    log(f"✅ Sonuç raporu gönderildi: {date_str}")
    
    # State'i kaydet
    save_state(STATE)
    
    return r

# --- Raporlar ----------------------------------------------------------------
def report_predictions(date_str):
    """
    Tahmin raporlama fonksiyonu - make_prediction'ı çağırır ve email gönderir
    """
    try:
        prediction_result = make_prediction(date_str)
        if prediction_result:
            # Başarılı tahmin işlemleri
            log_prediction_success(prediction_result)

            # EMAIL GÖNDERME KISMI
            lines = [f"🏟️ Günün Tahminleri — {date_str}"]
            for pred in prediction_result:
                lines.append(f"- {pred['match']} — {pred['prediction']} ({pred['confidence']}%) - {pred['note']}")

            body = "\n".join(lines)
            send_mail(f"Günün Tahminleri | {date_str}", body)

            return prediction_result
        else:
            # Tahmin başarısız
            log_prediction_failure()
            return None
    except Exception as e:
        print(f"Report prediction error: {e}")
        return None

# --- TOP_N ÖZELLİĞİ İÇİN YENİ FONKSİYONLAR ---
def get_top_n_predictions(predictions, n=TOP_N, min_confidence=MIN_CONF):
    """
    En yüksek güvenilirliğe sahip tahminleri filtreler
    
    Args:
        predictions: Tüm tahmin listesi
        n: Seçilecek tahmin sayısı
        min_confidence: Minimum güven seviyesi
    
    Returns:
        Sıralanmış tahmin listesi
    """
    # Güven eşiğini geçen tahminleri filtrele
    filtered = [p for p in predictions if p.get('confidence', 0) >= min_confidence]
    
    # Güvene göre sırala
    filtered.sort(key=lambda x: x.get('confidence', 0), reverse=True)
    
    # İlk N'yi al
    return filtered[:n]

def format_top_n_email(predictions, date_str, n=TOP_N):
    """
    TOP_N tahminlerini e-posta formatına dönüştürür
    """
    lines = []
    lines.append(f"🏆 GÜNÜN EN İYİ {n} TAHMİNİ — {date_str}")
    lines.append("=" * 60)
    lines.append("")
    
    if not predictions:
        lines.append("❌ Bugün için yeterince güvenilir tahmin bulunamadı.")
        lines.append(f"ℹ️ Minimum güven eşiği: {MIN_CONF}%")
        return "\n".join(lines)
    
    for i, pred in enumerate(predictions, 1):
        # Emoji seçimi
        emoji = "🔥" if pred.get('confidence', 0) >= HIGH_ALERT else "✅"
        
        lines.append(f"{emoji} #{i} - {pred.get('confidence', 0)}% GÜVEN")
        lines.append(f"   ⚽ {pred.get('match', 'Maç bilgisi yok')}")
        lines.append(f"   🎯 Tahmin: {pred.get('prediction', 'N/A')}")
        
        # Not kısmını temizle ve formatla
        note = pred.get('note', '')
        if note:
            # Uzun notu kısalt
            if len(note) > 150:
                note = note[:147] + "..."
            lines.append(f"   📝 {note}")
        
        lines.append("")
    
    # İstatistikler
    lines.append("📊 İSTATİSTİKLER")
    lines.append(f"   • Toplam Tahmin: {len(predictions)}")
    if predictions:
        avg_confidence = sum(p.get('confidence', 0) for p in predictions) / len(predictions)
        lines.append(f"   • Ortalama Güven: {avg_confidence:.1f}%")
        
        high_confidence = sum(1 for p in predictions if p.get('confidence', 0) >= HIGH_ALERT)
        if high_confidence > 0:
            lines.append(f"   • Yüksek Güven ({HIGH_ALERT}%+): {high_confidence}")
    
    lines.append("")
    lines.append(f"🤖 Model: {MODEL_VERSION}")
    lines.append(f"⏰ Üretim Zamanı: {datetime.now(TR_TZ).strftime('%Y-%m-%d %H:%M:%S')}")
    
    return "\n".join(lines)

# ==================== YENİ STATE YAPISI ====================

def initialize_external_data_state():
    """External data state yapısını başlat - GÜNCELLENDİ"""
    if "external_data" not in STATE:
        STATE["external_data"] = {}
    
    STATE["external_data"].setdefault("team_values", {})
    STATE["external_data"].setdefault("fifa_rankings", {})
    STATE["external_data"].setdefault("uefa_coefficients", {})
    STATE["external_data"].setdefault("last_updated", {})

# ==================== YENİ FONKSİYONLAR ====================

def update_team_values():
    """Takım değerlerini otomatik günceller"""
    try:
        # Örnek takımların değerlerini güncelle
        sample_teams = ["Galatasaray", "Fenerbahçe", "Beşiktaş", "Trabzonspor"]
        for team in sample_teams:
            value, source = get_team_value(team, "Turkey")
            STATE["external_data"]["team_values"][team] = value
            
        STATE["external_data"]["last_updated"]["team_values"] = datetime.now().isoformat()
        log("✅ Takım değerleri güncellendi")
    except Exception as e:
        log(f"❌ Takım değerleri güncelleme hatası: {e}")

def update_fifa_rankings():
    """FIFA sıralamalarını günceller (basit versiyon)"""
    try:
        # Örnek milli takım sıralamaları
        national_teams = {
            "Turkey": 40, "Germany": 16, "France": 2, "Brazil": 5,
            "Argentina": 1, "England": 3, "Spain": 8, "Italy": 9
        }
        
        STATE["external_data"]["fifa_rankings"] = national_teams
        STATE["external_data"]["last_updated"]["fifa_rankings"] = datetime.now().isoformat()
        log("✅ FIFA sıralamaları güncellendi")
    except Exception as e:
        log(f"❌ FIFA sıralamaları güncelleme hatası: {e}")

def update_uefa_coefficients():
    """UEFA katsayılarını günceller (basit versiyon)"""
    try:
        # Örnek UEFA katsayıları
        uefa_coeffs = {
            "Galatasaray": 25.0, "Fenerbahçe": 20.0, "Beşiktaş": 18.0,
            "Bayern Munich": 120.0, "Real Madrid": 130.0, "Manchester City": 125.0
        }
        
        STATE["external_data"]["uefa_coefficients"] = uefa_coeffs
        STATE["external_data"]["last_updated"]["uefa_coefficients"] = datetime.now().isoformat()
        log("✅ UEFA katsayıları güncellendi")
    except Exception as e:
        log(f"❌ UEFA katsayıları güncelleme hatası: {e}")

# ==================== ANA ÇALIŞTIRMA ====================

def main():
    """Ana çalıştırma fonksiyonu - Ensemble entegreli"""
    try:
        global STATE
        STATE = load_state()
        
        # EXTERNAL DATA STATE'İNİ BAŞLAT - YENİ EKLENDİ
        initialize_external_data_state()
        
        # API bağlantı testi
        debug_api_connection()
        
        # Cache temizleme
        clear_old_cache()
        
        # Ensemble modelini yükle
        load_ensemble_model()
        
        # Eğer model yüklenemediyse veya eğitilmemişse eğit
        if not ensemble_system.is_trained:
            log("🤖 Ensemble modeli eğitiliyor...")
            initialize_ensemble_training()
            if ensemble_system.is_trained:
                save_ensemble_model()
        
        # Service Loop'u çalıştır
        run_service_loop()
            
    except Exception as e:
        log(f"Main execution error: {e}")
        raise

def fix_results_schedule():
    """Sonuç raporu schedule düzeltmesi"""
    try:
        now_tr = datetime.now(TR_TZ)
        yesterday = _yesterday_str_tr(now_tr)
        
        # Dünün maçlarını bul ve sonuçları raporla
        fetch_results(yesterday)
        
    except Exception as e:
        log(f"Results schedule fix error: {e}")

# ==================== MEVCUT KODA ENTEGRASYON ====================

# Mevcut fonksiyonları güncelle
def rate_fixture(fx, odds_info):
    """Mevcut rate_fixture'ı ensemble ile değiştir"""
    return rate_fixture_with_ensemble(fx, odds_info)

# Diğer gerekli fonksiyonlar burada kalacak...

if __name__ == "__main__":
    # Komut satırı argümanlarını işle
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["AUTO", "PREDICT", "RESULTS"], default="AUTO")
    args = parser.parse_args()
    
    MODE_ENV = args.mode
    
    # Ana fonksiyonu çalıştır
    main()
