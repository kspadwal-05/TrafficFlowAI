#!/usr/bin/env python3
"""
TrafficFlow ETL - SQLite + VBA JSON integration
Tech: Docker, MS Access, Pandas, Python, SQL, VBA
"""
import os, sys, json, sqlite3, logging, urllib.request, urllib.error
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
from difflib import SequenceMatcher

ACCESS_TABLE = "ALTERED Device Inventory List_back_up_Nov13_18"
DB_PATH = "/app/data/processed/trafficflow.db"
OUT_JSON = "/app/data/processed/transformed_data.json"
PREFER_VBA_EXPORT = os.getenv("PREFER_VBA_EXPORT", "1") == "1"

# logging
os.makedirs("/app/data/processed", exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=[logging.StreamHandler(sys.stdout)])
log = logging.getLogger("etl")

@dataclass
class Config:
    access_db_path: str
    api_base_url: str = "https://httpbin.org/post"
    api_token: str = "dev-token"
    accuracy_threshold: float = 0.7

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            access_db_path=os.getenv("ACCESS_DB_PATH","/app/data/input/traffic_database.accdb"),
            api_base_url=os.getenv("API_BASE_URL","https://httpbin.org/post"),
            api_token=os.getenv("API_TOKEN","dev-token"),
            accuracy_threshold=float(os.getenv("ACCURACY_THRESHOLD","0.7")),
        )

class AccessExtractor:
    def __init__(self, cfg: Config):
        self.cfg = cfg

    def extract(self) -> pd.DataFrame:
        # Prefer VBA-exported JSON if present
        vba_json = "/app/data/input/export_from_vba.json"
        if os.path.exists(vba_json) and PREFER_VBA_EXPORT:
            log.info("Using VBA-exported JSON: %s", vba_json)
            with open(vba_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, dict) and isinstance(data.get("records"), list):
                records = data["records"]
            elif isinstance(data, list):
                records = data
            else:
                records = []
            try:
                df = pd.json_normalize(records)
            except Exception:
                df = pd.DataFrame(records)
            return df

        # Try pyodbc (Windows)
        try:
            import pyodbc
            if os.path.exists(self.cfg.access_db_path):
                conn_str = f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={self.cfg.access_db_path};"
                log.info("Attempting pyodbc Access connection...")
                with pyodbc.connect(conn_str) as conn:
                    df = pd.read_sql(f"SELECT * FROM [{ACCESS_TABLE}]", conn)
                    log.info("Read %d rows via pyodbc", len(df))
                    return df
        except Exception as e:
            log.warning("pyodbc path failed: %s", e)

        # Try UCanAccess (JDBC)
        try:
            import jaydebeapi  # type: ignore
            jdbc_url = f"jdbc:ucanaccess://{self.cfg.access_db_path};ignorecase=true"
            driver = "net.ucanaccess.jdbc.UcanaccessDriver"
            log.info("Attempting UCanAccess JDBC connection...")
            conn = jaydebeapi.connect(driver, jdbc_url, [], os.environ.get("CLASSPATH",""))
            try:
                cur = conn.cursor()
                cur.execute(f"SELECT * FROM [{ACCESS_TABLE}]")
                rows = cur.fetchall()
                cols = [d[0] for d in cur.description]
                df = pd.DataFrame(rows, columns=cols)
                log.info("Read %d rows via UCanAccess", len(df))
                return df
            finally:
                conn.close()
        except Exception as e:
            log.warning("UCanAccess path failed: %s", e)

        # Fallback CSV
        sample = "/app/data/input/sample.csv"
        if os.path.exists(sample):
            log.warning("Falling back to sample CSV: %s", sample)
            return pd.read_csv(sample)
        raise RuntimeError("No Access source / VBA JSON / sample CSV found.")

class AddressComparator:
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self.refs: List[str] = []

    def load_refs(self, path: str):
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    s = line.strip()
                    if s and s.lower() != "n/a":
                        self.refs.append(s)

    def best(self, s: str) -> str:
        if not s or not self.refs:
            return s
        best_s, best_r = s, 0.0
        for ref in self.refs:
            r = SequenceMatcher(None, s.lower(), ref.lower()).ratio()
            if r > best_r:
                best_r, best_s = r, ref
        return best_s if best_r >= self.threshold else s

class Transformer:
    def __init__(self, threshold: float):
        self.comp = AddressComparator(threshold)
        # Optionally load reference names file if present
        for cand in ["/app/legacy/outputReplaceOnly.txt", "/app/legacy/StreetNames.txt"]:
            self.comp.load_refs(cand)

    def to_json_records(self, df: pd.DataFrame) -> List[Dict]:
        # If this looks like VBA JSON (already has recordId), just normalize/validate minimal fields
        if "recordId" in df.columns:
            records = []
            for _, r in df.iterrows():
                rec_id = self._s(r.get("recordId",""))
                if not rec_id:
                    continue
                rec = {
                    "recordId": rec_id,
                    "status": self._s(r.get("status","")),
                    "streetName": self.comp.best(self._s(r.get("streetName",""))),
                    "intersection1": self.comp.best(self._s(r.get("intersection1",""))),
                    "intersection2": self.comp.best(self._s(r.get("intersection2",""))),
                    "roadType": self._s(r.get("roadType","")),
                    "requestedAnalysisInfoDate": self._d(r.get("requestedAnalysisInfoDate")),
                    "receivedAnalysisInfoDate": self._d(r.get("receivedAnalysisInfoDate")),
                    "streetOperation": self._s(r.get("streetOperation","")),
                    "volume": self._i(r.get("volume")),
                    "postedSpeedLimit": self._i(r.get("postedSpeedLimit")),
                    "averageSpeed": self._f(r.get("averageSpeed")),
                    "percentileSpeed85": self._f(r.get("percentileSpeed85")),
                    "analysisRecommended": self._s(r.get("analysisRecommended","")),
                    "planNumber": self._s(r.get("planNumber","")),
                    "estimatedCost": self._f(r.get("estimatedCost")),
                    "comments": self._s(r.get("comments","")),
                    "numSpeedHumps": self._i(r.get("numSpeedHumps")),
                    "numSpeedBumps": self._i(r.get("numSpeedBumps")),
                    "priorityRanking": self._i(r.get("priorityRanking")),
                }
                records.append(rec)
            return records

        # Else assume Access headers and map
        rename = {
            "ID":"ID",
            "Location - Street Name1":"street",
            "From - Street Name2":"int1",
            "To - Street Name3":"int2",
            "Road Classification":"roadType",
            "Status":"status",
            "Date Data Requested":"requested",
            "Date Data Received":"received",
            "Street Operation":"streetOperation",
            "Volume (vpd)":"volume",
            "Posted Speed Limit (km/h)":"posted",
            "Average Speed (km/h)":"avg",
            "85th Percentile Speed (km/h)":"p85",
            "Staff Recommended2":"recommended",
            "Plan/Drawing Number":"plan",
            "Estimated Cost":"cost",
            "Comments":"comments",
            "Speed Humps":"humps",
            "Laneway Speed Bump":"bumps",
            "Ranking":"rank",
        }
        df2 = df.rename(columns={k:v for k,v in rename.items() if k in df.columns}).copy()
        records = []
        for _, r in df2.iterrows():
            rec_id = self._s(r.get("ID",""))
            if not rec_id:
                continue
            rec = {
                "recordId": rec_id,
                "status": self._s(r.get("status","")),
                "streetName": self.comp.best(self._s(r.get("street",""))),
                "intersection1": self.comp.best(self._s(r.get("int1",""))),
                "intersection2": self.comp.best(self._s(r.get("int2",""))),
                "roadType": self._s(r.get("roadType","")),
                "requestedAnalysisInfoDate": self._d(r.get("requested")),
                "receivedAnalysisInfoDate": self._d(r.get("received")),
                "streetOperation": self._s(r.get("streetOperation","")),
                "volume": self._i(r.get("volume")),
                "postedSpeedLimit": self._i(r.get("posted")),
                "averageSpeed": self._f(r.get("avg")),
                "percentileSpeed85": self._f(r.get("p85")),
                "analysisRecommended": self._s(r.get("recommended","")),
                "planNumber": self._s(r.get("plan","")),
                "estimatedCost": self._f(r.get("cost")),
                "comments": self._s(r.get("comments","")),
                "numSpeedHumps": self._i(r.get("humps")),
                "numSpeedBumps": self._i(r.get("bumps")),
                "priorityRanking": self._i(r.get("rank")),
            }
            records.append(rec)
        return records

    def _s(self, v): 
        return "" if pd.isna(v) else str(v).strip()
    def _i(self, v, default=0):
        try: return int(float(v)) if not pd.isna(v) and str(v)!="" else default
        except: return default
    def _f(self, v, default=0.0):
        try: return float(v) if not pd.isna(v) and str(v)!="" else default
        except: return default
    def _d(self, v):
        try:
            if pd.isna(v) or str(v)=="": return ""
            return pd.to_datetime(v).strftime("%Y-%m-%d")
        except: return ""

class SQLiteLoader:
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    def init_schema(self):
        sql = """
        CREATE TABLE IF NOT EXISTS traffic_devices (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            record_id TEXT UNIQUE NOT NULL,
            street_name TEXT,
            intersection1 TEXT,
            intersection2 TEXT,
            road_type TEXT,
            status TEXT,
            requested_analysis_date TEXT,
            received_analysis_date TEXT,
            street_operation TEXT,
            volume_vpd INTEGER,
            posted_speed_limit INTEGER,
            average_speed REAL,
            percentile_speed_85 REAL,
            analysis_recommended TEXT,
            plan_number TEXT,
            estimated_cost REAL,
            comments TEXT,
            num_speed_humps INTEGER DEFAULT 0,
            num_speed_bumps INTEGER DEFAULT 0,
            priority_ranking INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX IF NOT EXISTS idx_td_record_id ON traffic_devices(record_id);
        CREATE INDEX IF NOT EXISTS idx_td_street ON traffic_devices(street_name);
        """
        with sqlite3.connect(self.db_path) as con:
            con.executescript(sql)

    def upsert_records(self, recs: List[Dict]) -> int:
        q = """
        INSERT INTO traffic_devices (
            record_id, street_name, intersection1, intersection2, road_type,
            status, requested_analysis_date, received_analysis_date, street_operation,
            volume_vpd, posted_speed_limit, average_speed, percentile_speed_85,
            analysis_recommended, plan_number, estimated_cost, comments,
            num_speed_humps, num_speed_bumps, priority_ranking, updated_at
        ) VALUES (
            :record_id, :street_name, :intersection1, :intersection2, :road_type,
            :status, :requested_analysis_date, :received_analysis_date, :street_operation,
            :volume_vpd, :posted_speed_limit, :average_speed, :percentile_speed_85,
            :analysis_recommended, :plan_number, :estimated_cost, :comments,
            :num_speed_humps, :num_speed_bumps, :priority_ranking, CURRENT_TIMESTAMP
        )
        ON CONFLICT(record_id) DO UPDATE SET
            street_name=excluded.street_name,
            intersection1=excluded.intersection1,
            intersection2=excluded.intersection2,
            road_type=excluded.road_type,
            status=excluded.status,
            requested_analysis_date=excluded.requested_analysis_date,
            received_analysis_date=excluded.received_analysis_date,
            street_operation=excluded.street_operation,
            volume_vpd=excluded.volume_vpd,
            posted_speed_limit=excluded.posted_speed_limit,
            average_speed=excluded.average_speed,
            percentile_speed_85=excluded.percentile_speed_85,
            analysis_recommended=excluded.analysis_recommended,
            plan_number=excluded.plan_number,
            estimated_cost=excluded.estimated_cost,
            comments=excluded.comments,
            num_speed_humps=excluded.num_speed_humps,
            num_speed_bumps=excluded.num_speed_bumps,
            priority_ranking=excluded.priority_ranking,
            updated_at=CURRENT_TIMESTAMP;
        """
        with sqlite3.connect(self.db_path) as con:
            cur = con.cursor()
            count = 0
            for r in recs:
                cur.execute(q, {
                    "record_id": r["recordId"],
                    "street_name": r["streetName"],
                    "intersection1": r["intersection1"],
                    "intersection2": r["intersection2"],
                    "road_type": r["roadType"],
                    "status": r["status"],
                    "requested_analysis_date": r["requestedAnalysisInfoDate"] or None,
                    "received_analysis_date": r["receivedAnalysisInfoDate"] or None,
                    "street_operation": r["streetOperation"],
                    "volume_vpd": r["volume"],
                    "posted_speed_limit": r["postedSpeedLimit"],
                    "average_speed": r["averageSpeed"],
                    "percentile_speed_85": r["percentileSpeed85"],
                    "analysis_recommended": r["analysisRecommended"],
                    "plan_number": r["planNumber"],
                    "estimated_cost": r["estimatedCost"],
                    "comments": r["comments"],
                    "num_speed_humps": r["numSpeedHumps"],
                    "num_speed_bumps": r["numSpeedBumps"],
                    "priority_ranking": r["priorityRanking"],
                })
                count += 1
            con.commit()
            return count

def post_json(url: str, token: str, payload: dict, timeout: int = 60) -> tuple[int, str]:
    req = urllib.request.Request(url, method="POST")
    req.add_header("Content-Type","application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    data = json.dumps(payload).encode("utf-8")
    try:
        with urllib.request.urlopen(req, data=data, timeout=timeout) as resp:
            status = resp.status
            body = resp.read().decode("utf-8", errors="ignore")
            return status, body
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="ignore")
    except Exception as e:
        return 0, str(e)

def main():
    cfg = Config.from_env()
    log.info("TrafficFlow ETL starting")

    # 1) Extract
    df = AccessExtractor(cfg).extract()

    # 2) Transform (handles both Access-headers and VBA-JSON shapes)
    tx = Transformer(cfg.accuracy_threshold)
    records = tx.to_json_records(df)

    # Save batch JSON
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)

    # 3) Load to SQLite
    loader = SQLiteLoader(DB_PATH)
    loader.init_schema()
    inserted = loader.upsert_records(records)
    log.info("Inserted/updated records in SQLite: %d", inserted)

    # 4) Publish
    status, body = post_json(cfg.api_base_url, cfg.api_token, {"records": records})
    if 200 <= status < 300:
        log.info("API upload OK (%s)", status)
    else:
        log.warning("API upload failed (%s): %s", status, body[:300])

    log.info("TrafficFlow ETL complete")

if __name__ == "__main__":
    main()
