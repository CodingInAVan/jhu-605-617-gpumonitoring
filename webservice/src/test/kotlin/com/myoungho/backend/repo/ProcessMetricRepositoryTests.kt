package com.myoungho.backend.repo

import com.myoungho.backend.model.MetricEntity
import com.myoungho.backend.model.MetricDeviceEntity
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.BeforeEach
import org.junit.jupiter.api.Test
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.test.context.ActiveProfiles
import java.time.OffsetDateTime
import java.time.ZoneOffset

@SpringBootTest
@ActiveProfiles("test")
class ProcessMetricRepositoryTests @Autowired constructor(
    val repo: MetricRepository,
) {
    @BeforeEach
    fun setup() {
        repo.deleteAll()

        val now = OffsetDateTime.now(ZoneOffset.UTC).withNano(0)
        // user 1, gpu A
        run {
            val m1 = MetricEntity(
                id = null,
                timestamp = now.minusMinutes(10),
                receivedAt = now,
                userId = 1L,
                type = "scope_begin",
                pid = 111,
                app = "app1",
                name = "p1",
                tag = "t1",
                tsNs = null,
                tsStartNs = null,
                tsEndNs = null,
                durationNs = null,
                extra = "{}",
            )
            val d1 = MetricDeviceEntity(
                id = null,
                metric = m1,
                idx = 0,
                deviceId = 0,
                gpuUuid = "GPU-A",
                gpuName = "RTX A",
                pciBus = 1,
                usedMiB = 100,
                freeMiB = 900,
                totalMiB = 1000,
            )
            m1.devices.add(d1)
            repo.save(m1)
        }
        run {
            val m2 = MetricEntity(
                id = null,
                timestamp = now.minusMinutes(5),
                receivedAt = now,
                userId = 1L,
                type = "scope_begin",
                pid = 112,
                app = "app2",
                name = "p2",
                tag = "t2",
                tsNs = null,
                tsStartNs = null,
                tsEndNs = null,
                durationNs = null,
                extra = "{}",
            )
            val d2 = MetricDeviceEntity(
                id = null,
                metric = m2,
                idx = 0,
                deviceId = 0,
                gpuUuid = "GPU-A",
                gpuName = "RTX A",
                pciBus = 1,
                usedMiB = 200,
                freeMiB = 800,
                totalMiB = 1000,
            )
            m2.devices.add(d2)
            repo.save(m2)
        }

        // user 2, gpu B
        run {
            val m3 = MetricEntity(
                id = null,
                timestamp = now.minusMinutes(3),
                receivedAt = now,
                userId = 2L,
                type = "scope_begin",
                pid = 221,
                app = "app1",
                name = "p3",
                tag = "t2",
                tsNs = null,
                tsStartNs = null,
                tsEndNs = null,
                durationNs = null,
                extra = "{}",
            )
            val d3 = MetricDeviceEntity(
                id = null,
                metric = m3,
                idx = 0,
                deviceId = 0,
                gpuUuid = "GPU-B",
                gpuName = "RTX B",
                pciBus = 2,
                usedMiB = 300,
                freeMiB = 700,
                totalMiB = 1000,
            )
            m3.devices.add(d3)
            repo.save(m3)
        }
    }

    @Test
    fun filtersByGpuUuidAndUserAsc() {
        val start = OffsetDateTime.now(ZoneOffset.UTC).minusHours(1)
        val end = OffsetDateTime.now(ZoneOffset.UTC).plusHours(1)
        val list = repo.findFilteredAsc(
            gpuUuid = "GPU-A",
            app = null,
            tag = null,
            start = start,
            end = end,
            userId = 1L,
        )
        assertEquals(2, list.size)
        assertEquals(111, list[0].pid)
        assertEquals(112, list[1].pid)
    }

    @Test
    fun filtersByAppAndTagDesc() {
        val list = repo.findFilteredDesc(
            gpuUuid = null,
            app = "app1",
            tag = "t2",
            start = null,
            end = null,
            userId = 2L,
        )
        assertEquals(1, list.size)
        assertEquals("GPU-B", list[0].devices.first().gpuUuid)
        assertEquals(221, list[0].pid)
    }

    @Test
    fun distinctAppNamesScopedByUser() {
        val apps1 = repo.findDistinctAppNames(1L)
        assertEquals(listOf("app1", "app2"), apps1)
        val apps2 = repo.findDistinctAppNames(2L)
        assertEquals(listOf("app1"), apps2)
    }
}
