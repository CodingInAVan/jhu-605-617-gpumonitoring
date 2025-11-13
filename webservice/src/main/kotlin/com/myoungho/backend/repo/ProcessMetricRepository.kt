package com.myoungho.backend.repo

import com.myoungho.backend.model.ProcessMetricEntity
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.time.OffsetDateTime

interface ProcessMetricRepository : JpaRepository<ProcessMetricEntity, Long> {
    
    @Query("""
        SELECT p FROM ProcessMetricEntity p 
        WHERE (:gpuId IS NULL OR p.gpuId = :gpuId)
        AND (:hostname IS NULL OR p.hostname = :hostname)
        AND (:gpuName IS NULL OR p.gpuName = :gpuName)
        AND (:start IS NULL OR p.timestamp >= :start)
        AND (:end IS NULL OR p.timestamp <= :end)
        AND (:userId IS NULL OR p.userId = :userId)
        ORDER BY p.timestamp ASC
    """)
    fun findFilteredAsc(
        @Param("gpuId") gpuId: String?,
        @Param("hostname") hostname: String?,
        @Param("gpuName") gpuName: String?,
        @Param("start") start: OffsetDateTime?,
        @Param("end") end: OffsetDateTime?,
        @Param("userId") userId: Long?
    ): List<ProcessMetricEntity>

    @Query("""
        SELECT p FROM ProcessMetricEntity p 
        WHERE (:gpuId IS NULL OR p.gpuId = :gpuId)
        AND (:hostname IS NULL OR p.hostname = :hostname)
        AND (:gpuName IS NULL OR p.gpuName = :gpuName)
        AND (:start IS NULL OR p.timestamp >= :start)
        AND (:end IS NULL OR p.timestamp <= :end)
        AND (:userId IS NULL OR p.userId = :userId)
        ORDER BY p.timestamp DESC
    """)
    fun findFilteredDesc(
        @Param("gpuId") gpuId: String?,
        @Param("hostname") hostname: String?,
        @Param("gpuName") gpuName: String?,
        @Param("start") start: OffsetDateTime?,
        @Param("end") end: OffsetDateTime?,
        @Param("userId") userId: Long?
    ): List<ProcessMetricEntity>
}
