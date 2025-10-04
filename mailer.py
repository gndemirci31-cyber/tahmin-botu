# === FIX: Daha sağlam fikstür çekme (geniş zaman penceresi + statü) ==========
import os, requests, datetime as dt
from datetime import timedelta, datetime, timezone
try:
    from zoneinfo import ZoneInfo
except:
    from backports.zoneinfo import ZoneInfo  # py<3.9 için

FD_BASE = "https://api.football-data.org/v4"
FD_TOKEN = os.getenv("FOOTBALL_DATA_TOKEN")

def _fd_get(path, params=None):
    r = requests.get(
        FD_BASE + path,
        headers={"X-Auth-Token": FD_TOKEN} if FD_TOKEN else {},
        params=params or {},
        timeout=30
    )
    r.raise_for_status()
    return r.json()

def _tier_one_comp_ids():
    """Ücretsiz planın izin verdiği lig ID’lerini al (cache’lenebilir)."""
    try:
        data = _fd_get("/competitions", params={"plan":"TIER_ONE"})
        return [str(c["id"]) for c in (data.get("competitions") or [])]
    except Exception:
        # erişim sorununda konkurensiz düşme: boş → tüm maç endpointine dene
        return []

def fetch_fixtures(day: dt.date):
    """
    Gün için fikstürü getir.
    - Zaman penceresi: TR günü etrafında -6h/+6h (UTC sapmalarını yakalar)
    - Statü: SCHEDULED ve TIMED
    - Plan TIER_ONE ligleriyle kısıtla (free planda garanti)
    """
    tz_tr = ZoneInfo("Europe/Istanbul")
    start = datetime(day.year, day.month, day.day, 0, 0, tzinfo=tz_tr) - timedelta(hours=6)
    end   = datetime(day.year, day.month, day.day, 23, 59, tzinfo=tz_tr) + timedelta(hours=6)

    q = {
        "dateFrom": start.astimezone(timezone.utc).date().isoformat(),
        "dateTo":   end.astimezone(timezone.utc).date().isoformat()
    }

    matches = []
    comp_ids = _tier_one_comp_ids()
    try:
        if comp_ids:
            q2 = dict(q)
            q2["competitions"] = ",".join(comp_ids[:50])  # güvenli limit
            data = _fd_get("/matches", params=q2)
        else:
            data = _fd_get("/matches", params=q)

        for m in data.get("matches", []):
            if (m.get("status") in ("SCHEDULED","TIMED")):
                matches.append(m)
    except Exception as e:
        print(f"[fixtures] hata: {e}")

    # Hiç yoksa bir de “dateFrom=bugün, dateTo=bugün+1” ile deneyelim
    if not matches:
        try:
            q_fallback = {
                "dateFrom": day.isoformat(),
                "dateTo":   (day + timedelta(days=1)).isoformat()
            }
            data2 = _fd_get("/matches", params=q_fallback)
            for m in data2.get("matches", []):
                if (m.get("status") in ("SCHEDULED","TIMED")):
                    matches.append(m)
        except Exception as e:
            print(f"[fixtures-fallback] hata: {e}")

    print(f"[fixtures] {day.isoformat()} için bulunan maç sayısı: {len(matches)}")
    return matches
# ============================================================================

