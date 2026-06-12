#!/usr/bin/env bash
set -e

echo "============================================================"
echo " 莫高窟壁画监测系统 - 启动脚本"
echo "============================================================"

echo "[1/3] 安装依赖..."
pip3 install -r requirements.txt

echo ""
echo "[2/3] 数据库初始化（手动执行）:"
echo "  createdb mogao_monitor"
echo "  psql -d mogao_monitor -f database/init_timescaledb.sql"

echo ""
echo "[3/3] 启动后端..."
echo ""
echo "访问地址:"
echo "  API文档:      http://localhost:8000/docs"
echo "  三维前端:     http://localhost:8000/frontend/index.html"
echo "  5G模拟器:     python3 simulator/fiveg_simulator.py --mode fast"
echo ""

cd backend && python3 main.py
