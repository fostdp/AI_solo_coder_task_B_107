@echo off
chcp 65001 >nul
echo ============================================================
echo  莫高窟壁画监测系统 - 一键启动脚本
echo ============================================================
echo.

echo [1/4] 检查Python环境...
python --version >nul 2>&1
if errorlevel 1 (
    echo [错误] 未检测到Python，请先安装Python 3.10+
    pause
    exit /b 1
)
python --version

echo.
echo [2/4] 安装依赖包...
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
if errorlevel 1 (
    echo [警告] 部分依赖安装失败，尝试继续...
)

echo.
echo [3/4] 检查数据库初始化...
if exist database\init_timescaledb.sql (
    echo 数据库初始化脚本已就位: database\init_timescaledb.sql
    echo 请先手动执行: psql -U postgres -d mogao_monitor -f database\init_timescaledb.sql
)

echo.
echo [4/4] 启动后端服务 (端口 8000)...
echo.
echo ============================================================
echo  启动完成后，请访问:
echo  - API文档:      http://localhost:8000/docs
echo  - 三维监控前端: http://localhost:8000/frontend/index.html
echo ============================================================
echo.
echo 启动5G模拟器请新开终端执行: python simulator\fiveg_simulator.py --mode fast
echo.

cd backend && python main.py
