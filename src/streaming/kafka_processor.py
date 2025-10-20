#!/usr/bin/env python3
"""
Apache Kafka Stream Processor for TrafficFlow AI
Real-time traffic data processing with Kafka
"""
try:
    from kafka import KafkaProducer, KafkaConsumer
    from kafka.errors import KafkaError
except ImportError:
    KafkaProducer = None
    KafkaConsumer = None
    KafkaError = Exception
import json
import logging
from datetime import datetime
from typing import Dict, Any
try:
    import redis
except ImportError:
    redis = None

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class KafkaStreamProcessor:
    """Kafka stream processor for traffic data"""
    
    def __init__(self, bootstrap_servers='localhost:9092'):
        self.bootstrap_servers = bootstrap_servers
        self.producer = None
        self.consumer = None
        if redis:
            self.redis_client = redis.Redis(host='localhost', port=6379, db=0)
        else:
            self.redis_client = None
        
    def create_producer(self):
        """Create Kafka producer"""
        if not KafkaProducer:
            logger.warning("Kafka not available, skipping producer creation")
            return
        try:
            self.producer = KafkaProducer(
                bootstrap_servers=self.bootstrap_servers,
                value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                key_serializer=lambda k: k.encode('utf-8') if k else None
            )
            logger.info("Kafka producer created successfully")
        except KafkaError as e:
            logger.error(f"Failed to create Kafka producer: {e}")
            
    def create_consumer(self, topic='traffic-data'):
        """Create Kafka consumer"""
        if not KafkaConsumer:
            logger.warning("Kafka not available, skipping consumer creation")
            return
        try:
            self.consumer = KafkaConsumer(
                topic,
                bootstrap_servers=self.bootstrap_servers,
                value_deserializer=lambda m: json.loads(m.decode('utf-8')),
                auto_offset_reset='latest',
                group_id='trafficflow-ai-group'
            )
            logger.info(f"Kafka consumer created for topic: {topic}")
        except KafkaError as e:
            logger.error(f"Failed to create Kafka consumer: {e}")
    
    def publish_traffic_event(self, event_data: Dict[str, Any]):
        """Publish traffic event to Kafka"""
        if not self.producer:
            self.create_producer()
            
        try:
            event_data['timestamp'] = datetime.now().isoformat()
            self.producer.send('traffic-data', value=event_data, key=event_data.get('location'))
            self.producer.flush()
            logger.info(f"Published traffic event: {event_data['location']}")
        except KafkaError as e:
            logger.error(f"Failed to publish event: {e}")
    
    def cache_metrics(self, location: str, metrics: Dict[str, Any]):
        """Cache metrics in Redis"""
        if not self.redis_client:
            return
        try:
            key = f"traffic_metrics:{location}"
            self.redis_client.setex(key, 300, json.dumps(metrics))  # 5 minute TTL
            logger.info(f"Cached metrics for {location}")
        except Exception as e:
            logger.error(f"Failed to cache metrics: {e}")
    
    def get_cached_metrics(self, location: str) -> Dict[str, Any]:
        """Get cached metrics from Redis"""
        if not self.redis_client:
            return {}
        try:
            key = f"traffic_metrics:{location}"
            data = self.redis_client.get(key)
            if data:
                return json.loads(data)
        except Exception as e:
            logger.error(f"Failed to get cached metrics: {e}")
        return {}
    
    def process_stream(self):
        """Process Kafka stream"""
        if not self.consumer:
            self.create_consumer()
            
        try:
            for message in self.consumer:
                data = message.value
                location = data.get('location', 'unknown')
                
                # Process traffic data
                processed_data = self.process_traffic_data(data)
                
                # Cache in Redis
                self.cache_metrics(location, processed_data)
                
                logger.info(f"Processed traffic data for {location}")
                
        except Exception as e:
            logger.error(f"Stream processing error: {e}")
    
    def process_traffic_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Process and enrich traffic data"""
        return {
            'location': data.get('location'),
            'volume': data.get('volume', 0),
            'average_speed': data.get('average_speed', 0),
            'congestion_level': self.calculate_congestion_level(data),
            'processed_at': datetime.now().isoformat()
        }
    
    def calculate_congestion_level(self, data: Dict[str, Any]) -> str:
        """Calculate congestion level based on speed and volume"""
        speed = data.get('average_speed', 0)
        speed_limit = data.get('speed_limit', 50)
        
        if speed < speed_limit * 0.7:
            return 'high'
        elif speed < speed_limit * 0.9:
            return 'medium'
        else:
            return 'low'

if __name__ == "__main__":
    processor = KafkaStreamProcessor()
    processor.process_stream()