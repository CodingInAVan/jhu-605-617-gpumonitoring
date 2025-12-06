package com.myoungho.backend.repo

import com.myoungho.backend.model.MetricEntity
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param
import java.time.OffsetDateTime

interface MetricRepository : JpaRepository<MetricEntity, Long> {

    @Query(
        """
        SELECT DISTINCT p FROM MetricEntity p
        LEFT JOIN FETCH p.devices d
        WHERE (:gpuUuid IS NULL OR EXISTS (
            SELECT 1 FROM MetricDeviceEntity dd WHERE dd.metric = p AND dd.gpuUuid = :gpuUuid
        ))
        AND (:app IS NULL OR p.app = :app)
        AND (:tag IS NULL OR p.tag = :tag)
        AND (:start IS NULL OR p.timestamp >= :start)
        AND (:end IS NULL OR p.timestamp <= :end)
        AND (:userId IS NULL OR p.userId = :userId)
        ORDER BY p.timestamp ASC
        """
    )
    fun findFilteredAsc(
        @Param("gpuUuid") gpuUuid: String?,
        @Param("app") app: String?,
        @Param("tag") tag: String?,
        @Param("start") start: OffsetDateTime?,
        @Param("end") end: OffsetDateTime?,
        @Param("userId") userId: Long?,
    ): List<MetricEntity>

    @Query(
        """
        SELECT DISTINCT p FROM MetricEntity p
        LEFT JOIN FETCH p.devices d
        WHERE (:gpuUuid IS NULL OR EXISTS (
            SELECT 1 FROM MetricDeviceEntity dd WHERE dd.metric = p AND dd.gpuUuid = :gpuUuid
        ))
        AND (:app IS NULL OR p.app = :app)
        AND (:tag IS NULL OR p.tag = :tag)
        AND (:start IS NULL OR p.timestamp >= :start)
        AND (:end IS NULL OR p.timestamp <= :end)
        AND (:userId IS NULL OR p.userId = :userId)
        ORDER BY p.timestamp DESC
        """
    )
    fun findFilteredDesc(
        @Param("gpuUuid") gpuUuid: String?,
        @Param("app") app: String?,
        @Param("tag") tag: String?,
        @Param("start") start: OffsetDateTime?,
        @Param("end") end: OffsetDateTime?,
        @Param("userId") userId: Long?,
    ): List<MetricEntity>

    @Query(
        """
        SELECT DISTINCT p.app FROM MetricEntity p
        WHERE p.app IS NOT NULL
        AND (:userId IS NULL OR p.userId = :userId)
        ORDER BY p.app ASC
        """
    )
    fun findDistinctAppNames(@Param("userId") userId: Long?): List<String>
}
