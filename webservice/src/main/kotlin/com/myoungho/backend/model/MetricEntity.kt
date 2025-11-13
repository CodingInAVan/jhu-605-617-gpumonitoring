package com.myoungho.backend.model

import com.fasterxml.jackson.annotation.JsonValue
import jakarta.persistence.*
import java.time.OffsetDateTime

@Entity
@Table(name = "metrics")
class MetricEntity(
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    var id: Long? = null,

    @Column(name = "timestamp", nullable = false)
    var timestamp: OffsetDateTime,

    @Column(name = "received_at", nullable = false)
    var receivedAt: OffsetDateTime,

    @Enumerated(EnumType.STRING)
    @Column(name = "metric_type", length = 32, nullable = false)
    var metricType: MetricType,

    @Column(name = "hostname", length = 255)
    var hostname: String? = null,

    @Column(name = "gpu_id", length = 255)
    var gpuId: String? = null,

    @Column(name = "gpu_name", length = 255)
    var gpuName: String? = null,

    @Column(name = "user_id")
    var userId: Long? = null,

    @Column(name = "gpu_device_id")
    var gpuDeviceId: Long? = null,

    // store full original payload as JSON string
    @Lob
    @Column(name = "payload", columnDefinition = "TEXT", nullable = false)
    var payload: String,

    @Column(name = "value")
    var value: Double? = null,

    @Column(name = "unit", length = 32)
    var unit: String? = null,
)

enum class MetricType(@JsonValue val value: String) {
    memory("memory"),
    utilization("utilization"),
    temperature("temperature"),
    power("power"),
    clocks("clocks"),
    process("process");

    override fun toString(): String = value
}
