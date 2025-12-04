package com.myoungho.backend.migration

import org.slf4j.LoggerFactory
import org.springframework.boot.ApplicationArguments
import org.springframework.boot.ApplicationRunner
import org.springframework.core.Ordered
import org.springframework.core.annotation.Order
import org.springframework.jdbc.core.JdbcTemplate
import org.springframework.stereotype.Component

/**
 * Ensures the current schema exists for SQLite deployments where Hibernate's
 * ddl-auto=update may not create new tables if a legacy DB file already exists.
 * This initializer creates tables if they do not exist and adds minimal indexes.
 */
@Component
@Order(Ordered.HIGHEST_PRECEDENCE)
class SchemaInitializer(
    private val jdbcTemplate: JdbcTemplate,
) : ApplicationRunner {

    private val log = LoggerFactory.getLogger(javaClass)

    override fun run(args: ApplicationArguments?) {
        try {
            // Create metrics table (matches MetricEntity)
            jdbcTemplate.execute(
                """
                CREATE TABLE IF NOT EXISTS metrics (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  timestamp TEXT NOT NULL,
                  received_at TEXT NOT NULL,
                  user_id INTEGER,
                  type TEXT,
                  pid INTEGER,
                  app TEXT,
                  name TEXT,
                  tag TEXT,
                  ts_ns INTEGER,
                  ts_start_ns INTEGER,
                  ts_end_ns INTEGER,
                  duration_ns INTEGER,
                  extra TEXT NOT NULL DEFAULT '{}'
                )
                """
            )

            // Create metric_devices table (matches MetricDeviceEntity)
            jdbcTemplate.execute(
                """
                CREATE TABLE IF NOT EXISTS metric_devices (
                  id INTEGER PRIMARY KEY AUTOINCREMENT,
                  metric_id INTEGER NOT NULL,
                  idx INTEGER,
                  device_id INTEGER,
                  gpu_uuid TEXT,
                  gpu_name TEXT,
                  pci_bus INTEGER,
                  used_mib INTEGER,
                  free_mib INTEGER,
                  total_mib INTEGER,
                  FOREIGN KEY(metric_id) REFERENCES metrics(id) ON DELETE CASCADE
                )
                """
            )

            // Indexes for query performance (IF NOT EXISTS via SQLite name uniqueness)
            jdbcTemplate.execute("CREATE INDEX IF NOT EXISTS idx_pm_timestamp ON metrics(timestamp)")
            jdbcTemplate.execute("CREATE INDEX IF NOT EXISTS idx_pm_user ON metrics(user_id)")
            jdbcTemplate.execute("CREATE INDEX IF NOT EXISTS idx_pm_app ON metrics(app)")
            jdbcTemplate.execute("CREATE INDEX IF NOT EXISTS idx_pm_tag ON metrics(tag)")
            jdbcTemplate.execute("CREATE INDEX IF NOT EXISTS idx_pm_pid ON metrics(pid)")
            jdbcTemplate.execute("CREATE INDEX IF NOT EXISTS idx_pmd_metric ON metric_devices(metric_id)")
            jdbcTemplate.execute("CREATE INDEX IF NOT EXISTS idx_pmd_gpu_uuid ON metric_devices(gpu_uuid)")

            log.info("Schema ensured for tables: metrics, metric_devices")
        } catch (ex: Exception) {
            log.warn("Schema initializer encountered an error: {}", ex.message)
        }
    }
}
