package com.myoungho.backend.service

import com.fasterxml.jackson.core.JsonProcessingException
import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.databind.node.ObjectNode
import com.myoungho.backend.model.MetricEntity
import com.myoungho.backend.model.MetricDeviceEntity
import com.myoungho.backend.repo.MetricRepository
import com.myoungho.backend.repo.UserRepository
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import org.springframework.web.server.ResponseStatusException
import org.springframework.http.HttpStatus
import java.time.Instant
import java.time.OffsetDateTime
import java.time.ZoneOffset

@Service
class MetricService(
    private val processMetricRepository: MetricRepository,
    private val objectMapper: ObjectMapper,
    private val userRepository: UserRepository,
    private val metricDeviceRepository: com.myoungho.backend.repo.MetricDeviceRepository,
) {
    private val logger = LoggerFactory.getLogger(MetricService::class.java)

    /**
     * Normalize an epoch-based timestamp to nanoseconds.
     * Producers may send seconds, milliseconds, microseconds, or nanoseconds.
     * We infer the unit by magnitude and convert to ns.
     */
    private fun normalizeEpochToNs(v: Long): Long {
        // Rough thresholds for current epoch (~2025):
        // seconds: ~1_700_000_000
        // milliseconds: ~1_700_000_000_000
        // microseconds: ~1_700_000_000_000_000
        // nanoseconds: ~1_700_000_000_000_000_000
        return when {
            v >= 1_000_000_000_000_000_000L -> v // already ns
            v >= 1_000_000_000_000_000L -> v * 1_000L // microseconds -> ns
            v >= 1_000_000_000_000L -> v * 1_000_000L // milliseconds -> ns
            v >= 1_000_000_000L -> v * 1_000_000_000L // seconds -> ns
            else -> {
                // Extremely small value; assume seconds
                v * 1_000_000_000L
            }
        }
    }

    private fun detectEpochUnit(v: Long): String {
        return when {
            v >= 1_000_000_000_000_000_000L -> "ns"
            v >= 1_000_000_000_000_000L -> "us"
            v >= 1_000_000_000_000L -> "ms"
            v >= 1_000_000_000L -> "s"
            else -> "s"
        }
    }

    data class QueryParams(
        val metric: Any? = null, // ignored in process-only mode
        val start: OffsetDateTime?,
        val end: OffsetDateTime?,
        val limit: Int,
        val order: String,
        val hostname: String? = null, // unused; kept for controller compatibility
        val gpuId: String?,
        val gpuName: String? = null, // unused
        val aggregate: String? = null, // unused in process-only mode
        val field: String? = null, // unused
        val userId: Long?,
        val appName: String? = null, // alt param name, may be used by controllers
        val tag: String? = null,
    )

    fun getProcessMetrics(q: QueryParams): Map<String, Any?> {
        val app = q.appName // prefer explicit appName param if provided
        val list = if (q.order.equals("asc", true))
            processMetricRepository.findFilteredAsc(q.gpuId, app, q.tag, q.start, q.end, q.userId)
        else
            processMetricRepository.findFilteredDesc(q.gpuId, app, q.tag, q.start, q.end, q.userId)

        val limited = list.take(q.limit)
        val items = limited.map { toProcessOut(it) }
        return mapOf(
            "items" to items,
            "count" to items.size,
        )
    }

    fun stats(): Map<String, Any?> {
        // Minimal placeholder since GPU metric pipeline is removed
        return mapOf("metrics" to emptyMap<String, Any>())
    }

    fun listUniqueGpus(start: OffsetDateTime?, end: OffsetDateTime?, userId: Long?): Map<String, Any?> {
        // Fetch all metrics in range for user and collect unique devices in-memory.
        val metrics = processMetricRepository.findFilteredAsc(
            gpuUuid = null,
            app = null,
            tag = null,
            start = start,
            end = end,
            userId = userId,
        )
        data class DevKey(val uuid: String?, val name: String?)
        val latest = LinkedHashMap<DevKey, OffsetDateTime>()
        metrics.forEach { m ->
            m.devices.forEach { d ->
                val key = DevKey(d.gpuUuid, d.gpuName)
                val ts = m.timestamp
                val prev = latest[key]
                if (prev == null || (ts != null && ts.isAfter(prev))) {
                    if (ts != null) latest[key] = ts
                }
            }
        }
        val gpus = latest.entries.map { (k, v) ->
            mapOf(
                "gpuUuid" to k.uuid,
                "gpuName" to k.name,
                "lastSeen" to v.toString(),
            )
        }
        return mapOf(
            "gpus" to gpus,
            "count" to gpus.size,
        )
    }

    @Transactional
    fun ingestForUser(userId: Long, json: String): Map<String, Any?> {
        val node = try {
            objectMapper.readTree(json)
        } catch (ex: JsonProcessingException) {
            logger.warn("Failed to parse metrics JSON: {}", ex.message)
            throw ResponseStatusException(HttpStatus.BAD_REQUEST, "Invalid JSON payload: ${'$'}{ex.originalMessage ?: ex.message}")
        }
        val now = OffsetDateTime.now(ZoneOffset.UTC)
        var savedCount = 0
        val savedIds = mutableListOf<Long>()

        if (node.isArray) {
            node.forEach { item ->
                persistProcessEvent(userId, item, now)?.let { saved ->
                    savedCount += 1
                    saved.id?.let(savedIds::add)
                }
            }
        } else if (node.isObject) {
            persistProcessEvent(userId, node, now)?.let { saved ->
                savedCount += 1
                saved.id?.let(savedIds::add)
            }
        } else {
            throw IllegalArgumentException("Invalid JSON body; expected object or array")
        }
        return mapOf(
            "saved" to savedCount,
            "ids" to savedIds,
        )
    }

    private fun persistProcessEvent(userId: Long, node: JsonNode, receivedAt: OffsetDateTime): MetricEntity? {
        val user = userRepository.findById(userId).orElseThrow { IllegalArgumentException("Invalid user") }

        val type = node.get("type")?.asText()
        val pid = node.get("pid")?.asInt()
        val app = node.get("app")?.asText()
        val name = node.get("name")?.asText()
        val tag = node.get("tag")?.asText()

        // timestamps
        val tsNs = node.get("ts_ns")?.asLong()
        val tsStartNs = node.get("ts_start_ns")?.asLong()
        val tsEndNs = node.get("ts_end_ns")?.asLong()
        val durationNs = node.get("duration_ns")?.asLong()

        // Normalize incoming timestamp value with best-fit heuristic.
        // Producers may send:
        //  - Unix epoch in s/ms/us/ns
        //  - steady_clock nanoseconds (monotonic, not epoch-based)
        // We evaluate all possible units and pick the one closest to 'receivedAt'. If none are reasonably close,
        // we fall back to 'receivedAt' to avoid bogus years like 1970 or 2027.
        val canonicalRaw = tsNs ?: tsStartNs ?: tsEndNs
            ?: throw IllegalArgumentException("One of ts_ns, ts_start_ns, ts_end_ns is required")

        val best = chooseBestTimestamp(canonicalRaw, receivedAt)
        val timestamp = best.timestamp
        if (best.fallbackToReceived) {
            logger.warn(
                "Incoming timestamp value {} did not resemble a wall-clock epoch in s/ms/us/ns (min delta={} ms). Falling back to receivedAt.",
                canonicalRaw, best.minDeltaMillis
            )
        } else if (best.unitChosen != null && best.unitChosen != "ns") {
            logger.warn(
                "Timestamp unit inferred as {} for value {} — normalizing to ns. Producer should send ns.",
                best.unitChosen, canonicalRaw
            )
        }

        // Build device entities: prefer explicit devices[]; if absent, synthesize one from flat fields
        val devicesToAttach = mutableListOf<MetricDeviceEntity>()
        val devicesNode = node.get("devices")
        if (devicesNode != null && devicesNode.isArray && devicesNode.size() > 0) {
            var idx = 0
            devicesNode.forEach { d ->
                val md = MetricDeviceEntity(
                    id = null,
                    metric = null, // set after MetricEntity is created
                    idx = idx,
                    deviceId = d.get("id")?.asInt(),
                    gpuUuid = d.get("uuid")?.asText(),
                    gpuName = d.get("name")?.asText(),
                    pciBus = d.get("pci_bus")?.asInt(),
                    usedMiB = d.get("used_mib")?.asInt(),
                    freeMiB = d.get("free_mib")?.asInt(),
                    totalMiB = d.get("total_mib")?.asInt(),
                )
                devicesToAttach.add(md)
                idx += 1
            }
        } else {
            // fallback to explicit flat device fields if provided
            val flatUuid = node.get("gpu_uuid")?.asText()
            val flatName = node.get("gpu_name")?.asText()
            val flatDevId = node.get("device_id")?.asInt()
            if (flatUuid != null || flatName != null || flatDevId != null) {
                val md = MetricDeviceEntity(
                    id = null,
                    metric = null,
                    idx = 0,
                    deviceId = flatDevId,
                    gpuUuid = flatUuid,
                    gpuName = flatName,
                    pciBus = null,
                    usedMiB = null,
                    freeMiB = null,
                    totalMiB = null,
                )
                devicesToAttach.add(md)
            }
        }

        // Build `extra` by removing all fields that are mapped to columns from the incoming object.
        // The remaining unmapped fields (e.g., kernel, grid, block, shared_mem_bytes, cuda_error, logPath, etc.)
        // are preserved in `extra` as JSON TEXT.
        val payloadString = if (node.isObject) {
            val extraObj = (node as ObjectNode).deepCopy()
            val removeKeys = setOf(
                // identity
                "type", "pid", "app", "name", "tag",
                // timing
                "ts_ns", "ts_start_ns", "ts_end_ns", "duration_ns",
                // device mapping moved to separate table
                "device_id", "gpu_uuid", "gpu_name", "devices"
            )
            removeKeys.forEach { key -> extraObj.remove(key) }
            objectMapper.writeValueAsString(extraObj)
        } else {
            // In practice, events are objects; fallback to raw text if not
            objectMapper.writeValueAsString(node)
        }

        val entity = MetricEntity(
            id = null,
            timestamp = timestamp,
            receivedAt = receivedAt,
            userId = user.id,
            type = type,
            pid = pid,
            app = app,
            name = name,
            tag = tag,
            tsNs = tsNs,
            tsStartNs = tsStartNs,
            tsEndNs = tsEndNs,
            durationNs = durationNs,
            extra = payloadString,
        )
        // attach devices
        devicesToAttach.forEach { d ->
            d.metric = entity
            entity.devices.add(d)
        }

        return processMetricRepository.save(entity)
    }

    private data class BestTs(
        val timestamp: OffsetDateTime,
        val unitChosen: String?,
        val minDeltaMillis: Long,
        val fallbackToReceived: Boolean,
    )

    /**
     * Evaluate s/ms/us/ns interpretations and pick the one closest to 'receivedAt'.
     * If all interpretations are too far (e.g., > 1 day), return receivedAt.
     */
    private fun chooseBestTimestamp(raw: Long, receivedAt: OffsetDateTime): BestTs {
        val candidates = listOf(
            "s" to raw * 1_000_000_000L,
            "ms" to raw * 1_000_000L,
            "us" to raw * 1_000L,
            "ns" to raw,
        )

        val recvInstant = receivedAt.toInstant()
        var bestUnit: String? = null
        var bestTs: OffsetDateTime = receivedAt
        var minDelta = Long.MAX_VALUE

        for ((unit, ns) in candidates) {
            // guard overflow to seconds/nanos
            val sec = ns / 1_000_000_000L
            val nanoAdj = (ns % 1_000_000_000L)
            if (sec < 0) continue
            val inst = try { Instant.ofEpochSecond(sec, nanoAdj) } catch (_: Exception) { continue }
            val delta = kotlin.math.abs(inst.toEpochMilli() - recvInstant.toEpochMilli())
            if (delta < minDelta) {
                minDelta = delta
                bestUnit = unit
                bestTs = OffsetDateTime.ofInstant(inst, ZoneOffset.UTC)
            }
        }

        // Threshold: if even the best is too far from now, likely steady_clock; use receivedAt
        val thresholdMs = 24L * 60L * 60L * 1000L // 1 day
        val fallback = minDelta > thresholdMs
        return if (fallback) BestTs(receivedAt, null, minDelta, true)
        else BestTs(bestTs, bestUnit, minDelta, false)
    }

    private fun toProcessOut(e: MetricEntity): Map<String, Any?> {
        val payloadMap: Map<String, Any?> = try {
            objectMapper.readValue(e.extra, Map::class.java) as Map<String, Any?>
        } catch (ex: Exception) {
            mapOf("raw" to ex.message)
        }
        // expose first device convenience fields for compatibility
        val firstDev = e.devices.minByOrNull { it.idx ?: Int.MAX_VALUE }
        return mapOf(
            "id" to e.id,
            // expose event type so frontend can distinguish scope/kernel/sample
            "type" to e.type,
            "timestamp" to e.timestamp.toString(),
            "received_at" to e.receivedAt.toString(),
            "pid" to e.pid,
            // API compatibility: expose processName from name
            "processName" to e.name,
            "app" to e.app,
            "tag" to e.tag,
            "gpuUuid" to firstDev?.gpuUuid,
            "durationNs" to e.durationNs,
            "usedMemoryMiB" to firstDev?.usedMiB,
            // Keep original field name for consumers
            "extra" to payloadMap,
            // Back-compat alias expected by some tests/clients
            "payload" to payloadMap,
        )
    }
}
