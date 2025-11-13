package com.myoungho.backend.repo

import com.myoungho.backend.model.MetricEntity
import com.myoungho.backend.model.MetricType
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.time.OffsetDateTime

interface MetricRepository : JpaRepository<MetricEntity, Long> {

    @Query(
        """
        SELECT m FROM MetricEntity m
        WHERE (:metricType IS NULL OR m.metricType = :metricType)
          AND (:hostname IS NULL OR m.hostname = :hostname)
          AND (:gpuId IS NULL OR m.gpuId = :gpuId)
          AND (:gpuName IS NULL OR m.gpuName = :gpuName)
          AND (:start IS NULL OR m.timestamp >= :start)
          AND (:end IS NULL OR m.timestamp <= :end)
          AND (:userId IS NULL OR m.userId = :userId)
        ORDER BY m.timestamp DESC
        """
    )
    fun findFilteredDesc(
        @Param("metricType") metricType: MetricType?,
        @Param("hostname") hostname: String?,
        @Param("gpuId") gpuId: String?,
        @Param("gpuName") gpuName: String?,
        @Param("start") start: OffsetDateTime?,
        @Param("end") end: OffsetDateTime?,
        @Param("userId") userId: Long?,
    ): List<MetricEntity>

    @Query(
        """
        SELECT m FROM MetricEntity m
        WHERE (:metricType IS NULL OR m.metricType = :metricType)
          AND (:hostname IS NULL OR m.hostname = :hostname)
          AND (:gpuId IS NULL OR m.gpuId = :gpuId)
          AND (:gpuName IS NULL OR m.gpuName = :gpuName)
          AND (:start IS NULL OR m.timestamp >= :start)
          AND (:end IS NULL OR m.timestamp <= :end)
          AND (:userId IS NULL OR m.userId = :userId)
        ORDER BY m.timestamp ASC
        """
    )
    fun findFilteredAsc(
        @Param("metricType") metricType: MetricType?,
        @Param("hostname") hostname: String?,
        @Param("gpuId") gpuId: String?,
        @Param("gpuName") gpuName: String?,
        @Param("start") start: OffsetDateTime?,
        @Param("end") end: OffsetDateTime?,
        @Param("userId") userId: Long?,
    ): List<MetricEntity>

    @Query(
        """
        SELECT m.metricType as metricType, COUNT(m.id) as cnt, MAX(m.timestamp) as latest
        FROM MetricEntity m
        GROUP BY m.metricType
        """
    )
    fun statsCounts(): List<Array<Any>>

    @Query(
        """
        SELECT m.hostname as hostname, m.gpuId as gpuId, m.gpuName as gpuName, MAX(m.timestamp) as latest
        FROM MetricEntity m
        GROUP BY m.hostname, m.gpuId, m.gpuName
        """
    )
    fun listGpus(): List<Array<Any>>

    @Query(
        """
        SELECT AVG(m.value) FROM MetricEntity m
        WHERE (:metricType IS NULL OR m.metricType = :metricType)
          AND (:hostname IS NULL OR m.hostname = :hostname)
          AND (:gpuId IS NULL OR m.gpuId = :gpuId)
          AND (:gpuName IS NULL OR m.gpuName = :gpuName)
          AND (:start IS NULL OR m.timestamp >= :start)
          AND (:end IS NULL OR m.timestamp <= :end)
          AND (:userId IS NULL OR m.userId = :userId)
        """
    )
    fun avgValue(
        @Param("metricType") metricType: MetricType?,
        @Param("hostname") hostname: String?,
        @Param("gpuId") gpuId: String?,
        @Param("gpuName") gpuName: String?,
        @Param("start") start: OffsetDateTime?,
        @Param("end") end: OffsetDateTime?,
        @Param("userId") userId: Long?,
    ): Double?

    @Query(
        """
        SELECT MIN(m.value) FROM MetricEntity m
        WHERE (:metricType IS NULL OR m.metricType = :metricType)
          AND (:hostname IS NULL OR m.hostname = :hostname)
          AND (:gpuId IS NULL OR m.gpuId = :gpuId)
          AND (:gpuName IS NULL OR m.gpuName = :gpuName)
          AND (:start IS NULL OR m.timestamp >= :start)
          AND (:end IS NULL OR m.timestamp <= :end)
          AND (:userId IS NULL OR m.userId = :userId)
        """
    )
    fun minValue(
        @Param("metricType") metricType: MetricType?,
        @Param("hostname") hostname: String?,
        @Param("gpuId") gpuId: String?,
        @Param("gpuName") gpuName: String?,
        @Param("start") start: OffsetDateTime?,
        @Param("end") end: OffsetDateTime?,
        @Param("userId") userId: Long?,
    ): Double?

    @Query(
        """
        SELECT MAX(m.value) FROM MetricEntity m
        WHERE (:metricType IS NULL OR m.metricType = :metricType)
          AND (:hostname IS NULL OR m.hostname = :hostname)
          AND (:gpuId IS NULL OR m.gpuId = :gpuId)
          AND (:gpuName IS NULL OR m.gpuName = :gpuName)
          AND (:start IS NULL OR m.timestamp >= :start)
          AND (:end IS NULL OR m.timestamp <= :end)
          AND (:userId IS NULL OR m.userId = :userId)
        """
    )
    fun maxValue(
        @Param("metricType") metricType: MetricType?,
        @Param("hostname") hostname: String?,
        @Param("gpuId") gpuId: String?,
        @Param("gpuName") gpuName: String?,
        @Param("start") start: OffsetDateTime?,
        @Param("end") end: OffsetDateTime?,
        @Param("userId") userId: Long?,
    ): Double?
}
