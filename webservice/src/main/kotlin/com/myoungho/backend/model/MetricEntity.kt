package com.myoungho.backend.model

import jakarta.persistence.*
import java.time.OffsetDateTime

@Entity
@Table(
    name = "metrics",
    indexes = [
        Index(name = "idx_pm_timestamp", columnList = "timestamp"),
        Index(name = "idx_pm_user", columnList = "user_id"),
        Index(name = "idx_pm_app", columnList = "app"),
        Index(name = "idx_pm_tag", columnList = "tag"),
        Index(name = "idx_pm_pid", columnList = "pid")
    ]
)
class MetricEntity(
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    var id: Long? = null,

    // Canonical timestamp derived from ns fields or explicit string
    @Column(name = "timestamp", nullable = false)
    var timestamp: OffsetDateTime,

    @Column(name = "received_at", nullable = false)
    var receivedAt: OffsetDateTime,

    @Column(name = "user_id")
    var userId: Long? = null,

    // Event identity
    @Column(name = "type", length = 32)
    var type: String? = null,

    @Column(name = "pid")
    var pid: Int? = null,

    @Column(name = "app", length = 255)
    var app: String? = null,

    @Column(name = "name", length = 512)
    var name: String? = null,

    @Column(name = "tag", length = 255)
    var tag: String? = null,

    // Raw timing fields
    @Column(name = "ts_ns")
    var tsNs: Long? = null,

    @Column(name = "ts_start_ns")
    var tsStartNs: Long? = null,

    @Column(name = "ts_end_ns")
    var tsEndNs: Long? = null,

    @Column(name = "duration_ns")
    var durationNs: Long? = null,

    // Associated devices captured for this metric event (not flattened)
    @OneToMany(mappedBy = "metric", cascade = [CascadeType.ALL], orphanRemoval = true, fetch = FetchType.LAZY)
    var devices: MutableList<MetricDeviceEntity> = mutableListOf(),

    @Lob
    @Column(name = "extra", columnDefinition = "TEXT NOT NULL DEFAULT '{}'", nullable = false)
    var extra: String,
)
