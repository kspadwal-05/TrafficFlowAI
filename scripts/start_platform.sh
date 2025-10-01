#!/bin/bash

# TrafficFlow AI Platform Startup Script

echo "🚀 Starting TrafficFlow AI Platform..."

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker and try again."
    exit 1
fi

# Create necessary directories
mkdir -p data/processed models logs monitoring/grafana/dashboards monitoring/grafana/datasources

# Start the platform
echo "📦 Starting Docker containers..."
docker-compose up -d

# Wait for services to be ready
echo "⏳ Waiting for services to start..."
sleep 30

# Check service health
echo "🔍 Checking service health..."

# Check API
if curl -f http://localhost:8000/health > /dev/null 2>&1; then
    echo "✅ API is healthy"
else
    echo "❌ API is not responding"
fi

# Check Dashboard
if curl -f http://localhost:8501 > /dev/null 2>&1; then
    echo "✅ Dashboard is running"
else
    echo "❌ Dashboard is not responding"
fi

# Check Kafka
if docker-compose exec kafka kafka-topics --bootstrap-server localhost:9092 --list > /dev/null 2>&1; then
    echo "✅ Kafka is running"
else
    echo "❌ Kafka is not responding"
fi

# Check Redis
if docker-compose exec redis redis-cli ping > /dev/null 2>&1; then
    echo "✅ Redis is running"
else
    echo "❌ Redis is not responding"
fi

echo ""
echo "🎉 TrafficFlow AI Platform is ready!"
echo ""
echo "📊 Access your services:"
echo "   Dashboard: http://localhost:8501"
echo "   API: http://localhost:8000"
echo "   Grafana: http://localhost:3000 (admin/admin123)"
echo "   Prometheus: http://localhost:9090"
echo ""
echo "🚀 Run the ETL pipeline:"
echo "   docker-compose exec trafficflow-ai python src/etl_main_enhanced.py"
echo ""
echo "🧠 Train ML models:"
echo "   docker-compose exec trafficflow-ai python src/ml_models/train_models.py"
echo ""
echo "📈 View logs:"
echo "   docker-compose logs -f trafficflow-ai"
