-- ============================================================
-- TimescaleDB 自动压缩配置
-- 超表超过一定时间后自动使用 zstd 压缩，节省80%存储
-- ============================================================
-- 压缩算法: zstd (压缩比 3.5~10×)
-- 压缩段: orderby 主键时间维度，segmentby 设备/墙面维度

SELECT set_column_type('vibration_raw_data', 'x_axis_accel', 'float8[]');
SELECT set_column_type('vibration_raw_data', 'y_axis_accel', 'float8[]');
SELECT set_column_type('vibration_raw_data', 'z_axis_accel', 'float8[]');

ALTER TABLE vibration_raw_data SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'sensor_id',
    timescaledb.compress_orderby = 'time DESC',
    timescaledb.compress_chunk_time_interval = '1 day'
);

ALTER TABLE thermal_images SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'camera_id',
    timescaledb.compress_orderby = 'time DESC',
    timescaledb.compress_chunk_time_interval = '1 day'
);

ALTER TABLE modal_analysis_results SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'surface_id',
    timescaledb.compress_orderby = 'time DESC',
    timescaledb.compress_chunk_time_interval = '7 days'
);

ALTER TABLE delamination_regions SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'surface_id',
    timescaledb.compress_orderby = 'time DESC',
    timescaledb.compress_chunk_time_interval = '7 days'
);

ALTER TABLE grouting_diffusion SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'task_id, injection_point_id',
    timescaledb.compress_orderby = 'time DESC',
    timescaledb.compress_chunk_time_interval = '1 day'
);

ALTER TABLE reinforcement_effectiveness SET (
    timescaledb.compress,
    timescaledb.compress_segmentby = 'task_id, surface_id',
    timescaledb.compress_orderby = 'time DESC',
    timescaledb.compress_chunk_time_interval = '30 days'
);

ALTER TABLE alerts SET (
    timescaledb.compress = false
);

-- ============================================================
-- 自动压缩策略: 数据超过 7 天的 chunk 自动压缩
-- ============================================================
SELECT add_compression_policy('vibration_raw_data',
    compress_after => INTERVAL '7 days',
    if_not_exists => TRUE
);

SELECT add_compression_policy('thermal_images',
    compress_after => INTERVAL '7 days',
    if_not_exists => TRUE
);

SELECT add_compression_policy('modal_analysis_results',
    compress_after => INTERVAL '30 days',
    if_not_exists => TRUE
);

SELECT add_compression_policy('delamination_regions',
    compress_after => INTERVAL '30 days',
    if_not_exists => TRUE
);

SELECT add_compression_policy('grouting_diffusion',
    compress_after => INTERVAL '7 days',
    if_not_exists => TRUE
);

SELECT add_compression_policy('reinforcement_effectiveness',
    compress_after => INTERVAL '90 days',
    if_not_exists => TRUE
);

-- ============================================================
-- 数据保留策略: 超表自动清理过期 chunk
-- vibration_raw_data: 保留90天
-- thermal_images: 保留180天
-- delamination_regions: 永久保留
-- ============================================================
SELECT add_retention_policy('vibration_raw_data',
    drop_after => INTERVAL '90 days',
    if_not_exists => TRUE
);

SELECT add_retention_policy('thermal_images',
    drop_after => INTERVAL '180 days',
    if_not_exists => TRUE
);

SELECT add_retention_policy('grouting_diffusion',
    drop_after => INTERVAL '730 days',
    if_not_exists => TRUE
);

-- ============================================================
-- 持续聚合刷新策略 (30分钟，已在 refresh.sql 中定义)
-- ============================================================
