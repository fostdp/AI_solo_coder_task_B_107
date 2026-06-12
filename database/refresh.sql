-- ============================================================
-- 连续聚合刷新策略修复
-- 问题: 默认自动刷新策略滞后严重，查询到过期数据
-- 修复: 显式设置 30 分钟刷新间隔
-- ============================================================

-- 删除旧连续聚合视图 (需先删除再重建以应用刷新策略)
DROP MATERIALIZED VIEW IF EXISTS delamination_daily_summary;
DROP MATERIALIZED VIEW IF EXISTS vibration_modal_hourly;

-- 重建剥离面积日汇总 (30分钟刷新)
CREATE MATERIALIZED VIEW delamination_daily_summary
WITH (timescaledb.continuous, timescaledb.materialized_only = true) AS
SELECT
    time_bucket('1 day', time) AS bucket_day,
    surface_id,
    COUNT(DISTINCT region_id) AS active_region_count,
    SUM(area_sqm) AS total_delamination_area,
    AVG(severity_score) AS avg_severity,
    MAX(frequency_drop_pct) AS max_frequency_drop
FROM delamination_regions
WHERE is_active = TRUE
GROUP BY bucket_day, surface_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('delamination_daily_summary',
    start_offset        => INTERVAL '7 days',
    end_offset          => INTERVAL '1 hour',
    schedule_interval   => INTERVAL '30 minutes',
    if_not_exists       => TRUE
);

-- 重建模态频率小时汇总 (30分钟刷新)
CREATE MATERIALIZED VIEW vibration_modal_hourly
WITH (timescaledb.continuous, timescaledb.materialized_only = true) AS
SELECT
    time_bucket('1 hour', time) AS bucket_hour,
    surface_id,
    AVG(natural_frequencies[1]) AS avg_fundamental_freq,
    AVG(damping_ratios[1]) AS avg_damping_ratio
FROM modal_analysis_results
GROUP BY bucket_hour, surface_id
WITH NO DATA;

SELECT add_continuous_aggregate_policy('vibration_modal_hourly',
    start_offset        => INTERVAL '3 days',
    end_offset          => INTERVAL '30 minutes',
    schedule_interval   => INTERVAL '30 minutes',
    if_not_exists       => TRUE
);

-- 为告警检测添加手动刷新入口 (可用于告警检查前强制刷新)
-- CALL refresh_continuous_aggregate('delamination_daily_summary', NULL, NULL);
-- CALL refresh_continuous_aggregate('vibration_modal_hourly', NULL, NULL);

-- 首次全量填充历史数据
CALL refresh_continuous_aggregate('delamination_daily_summary', NULL, NULL);
CALL refresh_continuous_aggregate('vibration_modal_hourly', NULL, NULL);
