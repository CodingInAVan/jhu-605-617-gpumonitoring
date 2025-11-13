package com.myoungho.backend.model

import jakarta.persistence.*
import java.time.OffsetDateTime
import java.time.ZoneOffset

@Entity
@Table(name = "process_metrics", indexes = [
    Index(name = "idx_process_metrics_timestamp", columnList = "timestamp"),
    Index(name = "idx_process_metrics_user", columnList = "user_id"),
    Index(name = "idx_process_metrics_gpu_device", columnList = "gpu_device_id")
])
class ProcessMetricEntity(
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    var id: Long? = null,

    @Column(name = "timestamp", nullable = false)
    var timestamp: OffsetDateTime,

    @Column(name = "received_at", nullable = false)
    var receivedAt: OffsetDateTime,

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

    @Column(name = "pid", nullable = false)
    var pid: Int,

    @Column(name = "process_name", length = 255)
    var processName: String? = null,

    @Column(name = "used_memory_mib")
    var usedMemoryMiB: Double? = null,

    // store full original payload as JSON string
    @Lob
    @Column(name = "payload", columnDefinition = "TEXT", nullable = false)
    var payload: String,
)
