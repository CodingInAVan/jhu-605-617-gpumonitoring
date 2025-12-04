package com.myoungho.backend.repo

import com.myoungho.backend.model.MetricDeviceEntity
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.time.OffsetDateTime

interface MetricDeviceRepository : JpaRepository<MetricDeviceEntity, Long> {

    interface DeviceView {
        fun getGpuUuid(): String?
        fun getGpuName(): String?
        fun getLastSeen(): OffsetDateTime?
    }

    @Query(
        """
        SELECT d.gpuUuid as gpuUuid, d.gpuName as gpuName, MAX(m.timestamp) as lastSeen
        FROM MetricDeviceEntity d
        JOIN d.metric m
        WHERE (:start IS NULL OR m.timestamp >= :start)
          AND (:end IS NULL OR m.timestamp <= :end)
          AND (:userId IS NULL OR m.userId = :userId)
        GROUP BY d.gpuUuid, d.gpuName
        ORDER BY d.gpuName ASC
        """
    )
    fun findDistinctDevices(
        @Param("start") start: OffsetDateTime?,
        @Param("end") end: OffsetDateTime?,
        @Param("userId") userId: Long?,
    ): List<DeviceView>
}
