package com.myoungho.backend.model

import jakarta.persistence.*
import java.time.OffsetDateTime
import java.time.ZoneOffset

@Entity
@Table(
    name = "gpu_devices",
    indexes = [
        Index(name = "idx_gpu_devices_user", columnList = "user_id"),
        Index(name = "idx_gpu_devices_gpuid", columnList = "gpu_id"),
        Index(name = "idx_gpu_devices_host_gpu", columnList = "hostname,gpu_name")
    ]
)
class GpuDeviceEntity(
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    var id: Long? = null,

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    var user: UserEntity,

    @Column(name = "gpu_id", length = 255)
    var gpuId: String? = null,

    @Column(name = "gpu_name", length = 255)
    var gpuName: String? = null,

    @Column(name = "hostname", length = 255)
    var hostname: String? = null,

    @Column(name = "notes", length = 1024)
    var notes: String? = null,

    @Column(name = "active", nullable = false)
    var active: Boolean = true,

    @Column(name = "created_at", nullable = false)
    var createdAt: OffsetDateTime = OffsetDateTime.now(ZoneOffset.UTC),

    @Column(name = "updated_at", nullable = false)
    var updatedAt: OffsetDateTime = OffsetDateTime.now(ZoneOffset.UTC),
)
