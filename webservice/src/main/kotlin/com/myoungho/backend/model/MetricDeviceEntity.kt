package com.myoungho.backend.model

import jakarta.persistence.*

@Entity
@Table(
    name = "metric_devices",
    indexes = [
        Index(name = "idx_pmd_metric", columnList = "metric_id"),
        Index(name = "idx_pmd_gpu_uuid", columnList = "gpu_uuid")
    ]
)
class MetricDeviceEntity(
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    var id: Long? = null,

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "metric_id", nullable = false)
    var metric: MetricEntity? = null,

    // position in the original devices[] array if present
    @Column(name = "idx")
    var idx: Int? = null,

    @Column(name = "device_id")
    var deviceId: Int? = null,

    @Column(name = "gpu_uuid", length = 255)
    var gpuUuid: String? = null,

    @Column(name = "gpu_name", length = 255)
    var gpuName: String? = null,

    @Column(name = "pci_bus")
    var pciBus: Int? = null,

    @Column(name = "used_mib")
    var usedMiB: Int? = null,

    @Column(name = "free_mib")
    var freeMiB: Int? = null,

    @Column(name = "total_mib")
    var totalMiB: Int? = null,

    // New telemetry fields (nullable; may not be present on all events)
    @Column(name = "util_gpu")
    var utilGpu: Int? = null, // percent

    @Column(name = "util_mem")
    var utilMem: Int? = null, // percent

    @Column(name = "temp_c")
    var tempC: Int? = null, // Celsius

    @Column(name = "power_mw")
    var powerMw: Int? = null, // milliwatts

    @Column(name = "clk_gfx")
    var clkGfx: Int? = null, // MHz

    @Column(name = "clk_sm")
    var clkSm: Int? = null, // MHz

    @Column(name = "clk_mem")
    var clkMem: Int? = null, // MHz
)
