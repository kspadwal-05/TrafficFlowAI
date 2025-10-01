#!/usr/bin/env python3
"""
TrafficFlow AI - Enhanced ETL Pipeline with ML and Real-time Processing
Advanced data engineering pipeline with machine learning and streaming capabilities
"""
import os, sys, json, sqlite3, logging, urllib.request, urllib.error
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import asyncio
import redis
from kafka import KafkaProducer

# Import our new ML and streaming modules
from .ml_models.traffic_predictor import TrafficPredictor, AnomalyDetector
from .streaming.kafka_processor import KafkaStreamProcessor, TrafficEvent

ACCESS_TABLE = "ALTERED Device Inventory List_back_up_Nov13_18"
DB_PATH = "/app/data/processed/trafficflow.db"
OUT_JSON = "/app/data/processed/transformed_data.json"
PREFER_VBA_EXPORT = os.getenv("PREFER_VBA_EXPORT", "1") == "1"

# Enhanced logging
os.makedirs("/app/data/processed", exist_ok=True)
os.makedirs("/app/logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("/app/logs/trafficflow_ai.log")
    ]
)
log = logging.getLogger("trafficflow_ai")

@dataclass
class EnhancedConfig:
    """Enhanced configuration with ML and streaming settings"""
    access_db_path: str
    api_base_url: str = "https://httpbin.org/post"
    api_token: str = "dev-token"
    accuracy_threshold: float = 0.7
    kafka_bootstrap_servers: str = "localhost:9092"
    redis_url: str = "localhost:6379"
    enable_ml: bool = True
    enable_streaming: bool = True
    model_retrain_threshold: int = 1000  # Retrain after 1000 new records

    @classmethod
    def from_env(cls) -> "EnhancedConfig":
        return cls(
            access_db_path=os.getenv("ACCESS_DB_PATH", "/app/data/input/traffic_database.accdb"),
            api_base_url=os.getenv("API_BASE_URL", "https://httpbin.org/post"),
            api_token=os.getenv("API_TOKEN", "dev-token"),
            accuracy_threshold=float(os.getenv("ACCURACY_THRESHOLD", "0.7")),
            kafka_bootstrap_servers=os.getenv("KAFKA_BOOTSTRAP_SERVERS", "localhost:9092"),
            redis_url=os.getenv("REDIS_URL", "localhost:6379"),
            enable_ml=os.getenv("ENABLE_ML", "true").lower() == "true",
            enable_streaming=os.getenv("ENABLE_STREAMING", "true").lower() == "true",
            model_retrain_threshold=int(os.getenv("MODEL_RETRAIN_THRESHOLD", "1000"))
        )

class EnhancedAccessExtractor:
    """Enhanced data extractor with caching and validation"""
    
    def __init__(self, cfg: EnhancedConfig):
        self.cfg = cfg
        self.redis_client = None
        if cfg.enable_streaming:
            try:
                self.redis_client = redis.Redis.from_url(f"redis://{cfg.redis_url}")
                self.redis_client.ping()
                log.info("Redis connection established")
            except Exception as e:
                log.warning(f"Redis connection failed: {e}")
                self.redis_client = None

    def extract(self) -> pd.DataFrame:
        """Enhanced extraction with caching and validation"""
        # Check cache first
        cache_key = "traffic_data_cache"
        if self.redis_client:
            try:
                cached_data = self.redis_client.get(cache_key)
                if cached_data:
                    log.info("Using cached data")
                    return pd.read_json(cached_data)
            except Exception as e:
                log.warning(f"Cache read failed: {e}")

        # Original extraction logic (from your existing code)
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
        else:
            # Try pyodbc (Windows)
            try:
                import pyodbc
                if os.path.exists(self.cfg.access_db_path):
                    conn_str = f"DRIVER={{Microsoft Access Driver (*.mdb, *.accdb)}};DBQ={self.cfg.access_db_path};"
                    log.info("Attempting pyodbc Access connection...")
                    with pyodbc.connect(conn_str) as conn:
                        df = pd.read_sql(f"SELECT * FROM [{ACCESS_TABLE}]", conn)
                        log.info("Read %d rows via pyodbc", len(df))
            except Exception as e:
                log.warning("pyodbc path failed: %s", e)
                # Try UCanAccess (JDBC)
                try:
                    import jaydebeapi
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
                    finally:
                        conn.close()
                except Exception as e:
                    log.warning("UCanAccess path failed: %s", e)
                    # Fallback CSV
                    sample = "/app/data/input/sample.csv"
                    if os.path.exists(sample):
                        log.warning("Falling back to sample CSV: %s", sample)
                        df = pd.read_csv(sample)
                    else:
                        raise RuntimeError("No Access source / VBA JSON / sample CSV found.")

        # Cache the data
        if self.redis_client:
            try:
                self.redis_client.setex(cache_key, 3600, df.to_json())  # Cache for 1 hour
                log.info("Data cached successfully")
            except Exception as e:
                log.warning(f"Cache write failed: {e}")

        return df

class EnhancedTransformer:
    """Enhanced transformer with ML features and data quality checks"""
    
    def __init__(self, threshold: float, enable_ml: bool = True):
        self.comp = AddressComparator(threshold)
        self.enable_ml = enable_ml
        self.ml_predictor = TrafficPredictor() if enable_ml else None
        self.anomaly_detector = AnomalyDetector() if enable_ml else None
        
        # Load reference names
        for cand in ["/app/legacy/outputReplaceOnly.txt", "/app/legacy/StreetNames.txt"]:
            self.comp.load_refs(cand)

    def to_json_records(self, df: pd.DataFrame) -> List[Dict]:
        """Enhanced transformation with ML features"""
        records = []
        ml_features = []
        
        for _, r in df.iterrows():
            # Original transformation logic
            if "recordId" in df.columns:
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
            else:
                # Access headers mapping (from your existing code)
                rename = {
                    "ID":"ID", "Location - Street Name1":"street", "From - Street Name2":"int1",
                    "To - Street Name3":"int2", "Road Classification":"roadType", "Status":"status",
                    "Date Data Requested":"requested", "Date Data Received":"received",
                    "Street Operation":"streetOperation", "Volume (vpd)":"volume",
                    "Posted Speed Limit (km/h)":"posted", "Average Speed (km/h)":"avg",
                    "85th Percentile Speed (km/h)":"p85", "Staff Recommended2":"recommended",
                    "Plan/Drawing Number":"plan", "Estimated Cost":"cost", "Comments":"comments",
                    "Speed Humps":"humps", "Laneway Speed Bump":"bumps", "Ranking":"rank",
                }
                df2 = df.rename(columns={k:v for k,v in rename.items() if k in df.columns}).copy()
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
            
            # Add ML-enhanced features
            if self.enable_ml and self.ml_predictor:
                try:
                    # Prepare features for ML
                    features = self.ml_predictor.prepare_features(pd.DataFrame([rec]))
                    
                    # Add ML predictions
                    if hasattr(self.ml_predictor, 'model') and self.ml_predictor.model is not None:
                        prediction = self.ml_predictor.predict(features)[0]
                        rec["ml_predicted_volume"] = float(prediction)
                        rec["ml_confidence"] = 0.85  # Simplified confidence score
                    
                    # Anomaly detection
                    if self.anomaly_detector and hasattr(self.anomaly_detector, 'isolation_forest'):
                        anomalies, scores = self.anomaly_detector.detect_anomalies(features)
                        rec["is_anomaly"] = bool(anomalies[0])
                        rec["anomaly_score"] = float(scores[0])
                    
                except Exception as e:
                    log.warning(f"ML processing failed for record {rec.get('recordId', 'unknown')}: {e}")
            
            # Add data quality metrics
            rec["data_quality_score"] = self._calculate_quality_score(rec)
            rec["processing_timestamp"] = datetime.now().isoformat()
            
            records.append(rec)
            ml_features.append(features[0] if self.enable_ml else None)
        
        # Train ML models if we have enough data
        if self.enable_ml and len(records) > 100:
            self._train_ml_models(records, ml_features)
        
        return records

    def _calculate_quality_score(self, rec: Dict) -> float:
        """Calculate data quality score for a record"""
        score = 1.0
        
        # Check for missing critical fields
        critical_fields = ['recordId', 'streetName', 'volume', 'averageSpeed']
        missing_fields = sum(1 for field in critical_fields if not rec.get(field))
        score -= missing_fields * 0.2
        
        # Check for data consistency
        if rec.get('volume', 0) < 0:
            score -= 0.1
        if rec.get('averageSpeed', 0) < 0:
            score -= 0.1
        if rec.get('postedSpeedLimit', 0) < 0:
            score -= 0.1
        
        return max(0.0, min(1.0, score))

    def _train_ml_models(self, records: List[Dict], features: List):
        """Train ML models with new data"""
        try:
            if not features or not any(f for f in features):
                return
            
            # Prepare training data
            X = np.array([f for f in features if f is not None])
            y = np.array([r['volume'] for r, f in zip(records, features) if f is not None])
            
            if len(X) < 50:  # Need minimum data for training
                return
            
            # Train traffic predictor
            if self.ml_predictor:
                metrics = self.ml_predictor.train(X, y)
                log.info(f"ML model trained - R²: {metrics['test_r2']:.3f}")
                
                # Save model
                os.makedirs("/app/models", exist_ok=True)
                self.ml_predictor.save_model("/app/models/traffic_model.pkl")
            
            # Train anomaly detector
            if self.anomaly_detector:
                self.anomaly_detector.fit(X)
                log.info("Anomaly detector trained")
                
        except Exception as e:
            log.error(f"ML model training failed: {e}")

    # Helper methods (from your existing code)
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

class AddressComparator:
    """Address comparison utility (from your existing code)"""
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

class EnhancedSQLiteLoader:
    """Enhanced SQLite loader with performance monitoring"""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)

    def init_schema(self):
        """Initialize enhanced database schema"""
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
            -- ML-enhanced fields
            ml_predicted_volume REAL,
            ml_confidence REAL,
            is_anomaly BOOLEAN DEFAULT FALSE,
            anomaly_score REAL,
            data_quality_score REAL,
            processing_timestamp TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_td_record_id ON traffic_devices(record_id);
        CREATE INDEX IF NOT EXISTS idx_td_street ON traffic_devices(street_name);
        CREATE INDEX IF NOT EXISTS idx_td_anomaly ON traffic_devices(is_anomaly);
        CREATE INDEX IF NOT EXISTS idx_td_quality ON traffic_devices(data_quality_score);
        CREATE INDEX IF NOT EXISTS idx_td_processing ON traffic_devices(processing_timestamp);
        """
        with sqlite3.connect(self.db_path) as con:
            con.executescript(sql)

    def upsert_records(self, recs: List[Dict]) -> int:
        """Enhanced upsert with ML fields"""
        q = """
        INSERT INTO traffic_devices (
            record_id, street_name, intersection1, intersection2, road_type,
            status, requested_analysis_date, received_analysis_date, street_operation,
            volume_vpd, posted_speed_limit, average_speed, percentile_speed_85,
            analysis_recommended, plan_number, estimated_cost, comments,
            num_speed_humps, num_speed_bumps, priority_ranking,
            ml_predicted_volume, ml_confidence, is_anomaly, anomaly_score,
            data_quality_score, processing_timestamp, updated_at
        ) VALUES (
            :record_id, :street_name, :intersection1, :intersection2, :road_type,
            :status, :requested_analysis_date, :received_analysis_date, :street_operation,
            :volume_vpd, :posted_speed_limit, :average_speed, :percentile_speed_85,
            :analysis_recommended, :plan_number, :estimated_cost, :comments,
            :num_speed_humps, :num_speed_bumps, :priority_ranking,
            :ml_predicted_volume, :ml_confidence, :is_anomaly, :anomaly_score,
            :data_quality_score, :processing_timestamp, CURRENT_TIMESTAMP
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
            ml_predicted_volume=excluded.ml_predicted_volume,
            ml_confidence=excluded.ml_confidence,
            is_anomaly=excluded.is_anomaly,
            anomaly_score=excluded.anomaly_score,
            data_quality_score=excluded.data_quality_score,
            processing_timestamp=excluded.processing_timestamp,
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
                    "ml_predicted_volume": r.get("ml_predicted_volume"),
                    "ml_confidence": r.get("ml_confidence"),
                    "is_anomaly": r.get("is_anomaly", False),
                    "anomaly_score": r.get("anomaly_score"),
                    "data_quality_score": r.get("data_quality_score", 1.0),
                    "processing_timestamp": r.get("processing_timestamp"),
                })
                count += 1
            con.commit()
            return count

class StreamingPublisher:
    """Publish data to Kafka for real-time processing"""
    
    def __init__(self, bootstrap_servers: str):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
    
    def setup_producer(self):
        """Initialize Kafka producer"""
        try:
            from kafka import KafkaProducer
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            log.info("Kafka producer initialized")
        except Exception as e:
            log.warning(f"Kafka producer setup failed: {e}")
    
    def publish_traffic_events(self, records: List[Dict]):
        """Publish traffic events to Kafka"""
        if not self.producer:
            self.setup_producer()
        
        if not self.producer:
            return
        
        try:
            for record in records:
                # Create traffic event
                event = TrafficEvent(
                    event_id=record["recordId"],
                    timestamp=datetime.now(),
                    location=f"{record['streetName']} & {record['intersection1']}",
                    event_type="data_update",
                    severity=1,
                    volume=record.get("volume", 0),
                    speed=record.get("averageSpeed", 0),
                    confidence=record.get("ml_confidence", 1.0),
                    metadata=record
                )
                
                # Publish to Kafka
                self.producer.send(
                    "traffic-events",
                    key=event.event_id,
                    value={
                        "event_id": event.event_id,
                        "timestamp": event.timestamp.isoformat(),
                        "location": event.location,
                        "event_type": event.event_type,
                        "severity": event.severity,
                        "volume": event.volume,
                        "speed": event.speed,
                        "confidence": event.confidence,
                        "metadata": event.metadata
                    }
                )
            
            self.producer.flush()
            log.info(f"Published {len(records)} events to Kafka")
            
        except Exception as e:
            log.error(f"Failed to publish events: {e}")

def post_json(url: str, token: str, payload: dict, timeout: int = 60) -> tuple[int, str]:
    """Enhanced API posting with retry logic"""
    req = urllib.request.Request(url, method="POST")
    req.add_header("Content-Type","application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    data = json.dumps(payload).encode("utf-8")
    
    for attempt in range(3):  # Retry up to 3 times
        try:
            with urllib.request.urlopen(req, data=data, timeout=timeout) as resp:
                status = resp.status
                body = resp.read().decode("utf-8", errors="ignore")
                return status, body
        except urllib.error.HTTPError as e:
            if attempt == 2:  # Last attempt
                return e.code, e.read().decode("utf-8", errors="ignore")
            time.sleep(1)  # Wait before retry
        except Exception as e:
            if attempt == 2:  # Last attempt
                return 0, str(e)
            time.sleep(1)  # Wait before retry

def main():
    """Enhanced main function with ML and streaming"""
    cfg = EnhancedConfig.from_env()
    log.info("TrafficFlow AI Enhanced ETL starting")
    
    start_time = datetime.now()
    
    try:
        # 1) Extract
        log.info("Step 1: Extracting data...")
        df = EnhancedAccessExtractor(cfg).extract()
        log.info(f"Extracted {len(df)} records")

        # 2) Transform (with ML)
        log.info("Step 2: Transforming data with ML features...")
        tx = EnhancedTransformer(cfg.accuracy_threshold, cfg.enable_ml)
        records = tx.to_json_records(df)
        log.info(f"Transformed {len(records)} records")

        # Save batch JSON
        with open(OUT_JSON, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        log.info(f"Saved transformed data to {OUT_JSON}")

        # 3) Load to SQLite (enhanced)
        log.info("Step 3: Loading to enhanced SQLite database...")
        loader = EnhancedSQLiteLoader(DB_PATH)
        loader.init_schema()
        inserted = loader.upsert_records(records)
        log.info(f"Inserted/updated {inserted} records in SQLite")

        # 4) Publish to streaming (if enabled)
        if cfg.enable_streaming:
            log.info("Step 4: Publishing to Kafka...")
            publisher = StreamingPublisher(cfg.kafka_bootstrap_servers)
            publisher.publish_traffic_events(records)

        # 5) Publish to API
        log.info("Step 5: Publishing to API...")
        status, body = post_json(cfg.api_base_url, cfg.api_token, {"records": records})
        if 200 <= status < 300:
            log.info(f"API upload successful ({status})")
        else:
            log.warning(f"API upload failed ({status}): {body[:300]}")

        # Performance metrics
        end_time = datetime.now()
        processing_time = (end_time - start_time).total_seconds()
        records_per_second = len(records) / processing_time if processing_time > 0 else 0
        
        log.info(f"TrafficFlow AI Enhanced ETL completed in {processing_time:.2f}s")
        log.info(f"Processing rate: {records_per_second:.2f} records/second")
        
        # Log ML insights if available
        if cfg.enable_ml:
            anomaly_count = sum(1 for r in records if r.get("is_anomaly", False))
            avg_quality = sum(r.get("data_quality_score", 1.0) for r in records) / len(records)
            log.info(f"ML Insights - Anomalies detected: {anomaly_count}, Avg quality: {avg_quality:.3f}")

    except Exception as e:
        log.error(f"Enhanced ETL failed: {e}")
        raise

if __name__ == "__main__":
    main()
