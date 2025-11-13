package com.myoungho.backend.repo

import com.myoungho.backend.model.GpuDeviceEntity
import org.springframework.data.jpa.repository.JpaRepository
import org.springframework.data.jpa.repository.Query
import org.springframework.data.repository.query.Param

interface GpuDeviceRepository : JpaRepository<GpuDeviceEntity, Long> {
    fun findByUserIdAndGpuId(userId: Long, gpuId: String): GpuDeviceEntity?
    fun findByUserIdAndHostnameAndGpuName(userId: Long, hostname: String, gpuName: String): GpuDeviceEntity?

    @Query(
        """
        SELECT g FROM GpuDeviceEntity g
        WHERE g.user.id = :userId
        ORDER BY g.createdAt DESC
        """
    )
    fun listForUser(@Param("userId") userId: Long): List<GpuDeviceEntity>
}