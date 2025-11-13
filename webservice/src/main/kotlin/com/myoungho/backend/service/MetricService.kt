package com.myoungho.backend.service

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.convertValue
import com.myoungho.backend.model.MetricEntity
import com.myoungho.backend.model.MetricType
import com.myoungho.backend.model.ProcessMetricEntity
import com.myoungho.backend.repo.MetricRepository
import org.slf4j.LoggerFactory
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.time.ZoneOffset

@Service
class MetricService(
    private val metricRepository: MetricRepository,
    private val processMetricRepository: com.myoungho.backend.repo.ProcessMetricRepository,
    private val objectMapper: ObjectMapper,
    private val gpuDeviceRepository: com.myoungho.backend.repo.GpuDeviceRepository,
    private val userRepository: com.myoungho.backend.repo.UserRepository,
) {
    private val logger = LoggerFactory.getLogger(MetricService::class.java)

    data class QueryParams(
        val metric: MetricType?,
        val start: OffsetDateTime?,
        val end: OffsetDateTime?,
        val limit: Int,
        val order: String,
        val hostname: String?,
        val gpuId: String?,
        val gpuName: String?,
        val aggregate: String?,
        val field: String?,
        val userId: Long?,
    )

    fun getMetrics(q: QueryParams): Map<String, Any?> {
        val list = if (q.order.equals("asc", true))
            metricRepository.findFilteredAsc(q.metric, q.hostname, q.gpuId, q.gpuName, q.start, q.end, q.userId)
        else
            metricRepository.findFilteredDesc(q.metric, q.hostname, q.gpuId, q.gpuName, q.start, q.end, q.userId)

        if (q.aggregate != null) {
            val value = when (q.aggregate.lowercase()) {
                "avg" -> metricRepository.avgValue(q.metric, q.hostname, q.gpuId, q.gpuName, q.start, q.end, q.userId)
                "min" -> metricRepository.minValue(q.metric, q.hostname, q.gpuId, q.gpuName, q.start, q.end, q.userId)
                "max" -> metricRepository.maxValue(q.metric, q.hostname, q.gpuId, q.gpuName, q.start, q.end, q.userId)
                else -> null
            }
            return mapOf(
                "aggregate" to q.aggregate.lowercase(),
                "field" to q.field,
                "value" to value,
            )
        }

        val limited = list.take(q.limit)
        val items = limited.map { toOut(it) }
        return mapOf(
            "items" to items,
            "count" to items.size,
        )
    }

    fun getProcessMetrics(q: QueryParams): Map<String, Any?> {
        val list = if (q.order.equals("asc", true))
            processMetricRepository.findFilteredAsc(q.gpuId, q.hostname, q.gpuName, q.start, q.end, q.userId)
        else
            processMetricRepository.findFilteredDesc(q.gpuId, q.hostname, q.gpuName, q.start, q.end, q.userId)

        val limited = list.take(q.limit)
        val items = limited.map { toProcessOut(it) }
        return mapOf(
            "items" to items,
            "count" to items.size,
        )
    }

    fun listGpus(): Map<String, Any?> {
        val rows = metricRepository.listGpus()
        val gpus = rows.map { arr ->
            val hostname = arr[0] as String?
            val gpuId = arr[1] as String?
            val gpuName = arr[2] as String?
            val latest = arr[3] as OffsetDateTime?
            mapOf(
                "hostname" to hostname,
                "gpuId" to gpuId,
                "gpuName" to gpuName,
                "latest" to latest?.toString(),
            )
        }
        return mapOf("gpus" to gpus)
    }

    fun stats(): Map<String, Any?> {
        val rows = metricRepository.statsCounts()
        val metrics: MutableMap<String, Any?> = linkedMapOf()
        for (r in rows) {
            val type = r[0] as MetricType
            val cnt = (r[1] as Number).toLong()
            val latest = r[2] as OffsetDateTime?
            metrics[type.value] = mapOf(
                "count" to cnt,
                "latest" to latest?.toString(),
            )
        }
        return mapOf("metrics" to metrics)
    }

    @Transactional
    fun ingestForUser(userId: Long, json: String): Map<String, Any?> {
        val node = objectMapper.readTree(json)
        val now = OffsetDateTime.now(ZoneOffset.UTC)
        var savedCount = 0
        val savedIds = mutableListOf<Long>()

        if (node.isArray) {
            for (item in node) {
                val result = persistConsolidatedNode(userId, item, now)
                savedCount += result.first
                savedIds.addAll(result.second)
            }
        } else if (node.isObject) {
            val result = persistConsolidatedNode(userId, node, now)
            savedCount += result.first
            savedIds.addAll(result.second)
        } else {
            throw IllegalArgumentException("Invalid JSON body; expected object or array")
        }
        return mapOf(
            "saved" to savedCount,
            "ids" to savedIds,
        )
    }

    private fun persistConsolidatedNode(userId: Long, item: JsonNode, receivedAt: OffsetDateTime): Pair<Int, List<Long>> {
        val map: MutableMap<String, Any?> = objectMapper.convertValue(item)

        val metricTypeText = (map["metricType"] as? String)?.trim()
            ?: throw IllegalArgumentException("'metricType' field is required")

        return when (metricTypeText.lowercase()) {
            "gpu" -> persistGpuMetrics(userId, map, receivedAt)
            "process" -> persistProcessMetric(userId, map, receivedAt)
            else -> throw IllegalArgumentException("Unknown metricType: $metricTypeText. Expected 'gpu' or 'process'")
        }
    }

    private fun persistGpuMetrics(userId: Long, map: MutableMap<String, Any?>, receivedAt: OffsetDateTime): Pair<Int, List<Long>> {
        // Extract common fields
        val timestampText = (map["timestamp"] as? String)
            ?: throw IllegalArgumentException("'timestamp' is required")
        var timestamp = OffsetDateTime.parse(timestampText)
        if (timestamp.offset == null) {
            timestamp = timestamp.withOffsetSameInstant(ZoneOffset.UTC)
        }

        val hostname = map["hostname"] as? String
        val gpuId = map["gpuId"] as? String
        val gpuName = map["gpuName"] as? String

        // Upsert device for this user
        val user = userRepository.findById(userId).orElseThrow { IllegalArgumentException("Invalid user") }
        val device = if (!gpuId.isNullOrBlank()) {
            gpuDeviceRepository.findByUserIdAndGpuId(user.id!!, gpuId)
                ?: gpuDeviceRepository.save(com.myoungho.backend.model.GpuDeviceEntity(user = user, gpuId = gpuId, gpuName = gpuName, hostname = hostname))
        } else if (!hostname.isNullOrBlank() && !gpuName.isNullOrBlank()) {
            gpuDeviceRepository.findByUserIdAndHostnameAndGpuName(user.id!!, hostname, gpuName)
                ?: gpuDeviceRepository.save(com.myoungho.backend.model.GpuDeviceEntity(user = user, gpuId = null, gpuName = gpuName, hostname = hostname))
        } else {
            null
        }

        val savedIds = mutableListOf<Long>()
        val payloadString = objectMapper.writeValueAsString(map)

        // Create memory metric
        val totalMiB = (map["totalMemoryMiB"] as? Number)?.toDouble()
        val usedTotalMiB = (map["usedTotalMemoryMiB"] as? Number)?.toDouble()
        val freeMiB = (map["freeMemoryMiB"] as? Number)?.toDouble()
        if (usedTotalMiB != null) {
            val memoryEntity = MetricEntity(
                id = null,
                timestamp = timestamp,
                receivedAt = receivedAt,
                metricType = MetricType.memory,
                hostname = hostname,
                gpuId = gpuId,
                gpuName = gpuName,
                userId = user.id,
                gpuDeviceId = device?.id,
                payload = payloadString,
                value = usedTotalMiB,
                unit = "MiB"
            )
            metricRepository.save(memoryEntity)
            memoryEntity.id?.let { savedIds.add(it) }
        }

        // Create utilization metric
        val gpuUtilPercent = (map["gpuUtilPercent"] as? Number)?.toDouble()
        if (gpuUtilPercent != null) {
            val utilizationEntity = MetricEntity(
                id = null,
                timestamp = timestamp,
                receivedAt = receivedAt,
                metricType = MetricType.utilization,
                hostname = hostname,
                gpuId = gpuId,
                gpuName = gpuName,
                userId = user.id,
                gpuDeviceId = device?.id,
                payload = payloadString,
                value = gpuUtilPercent,
                unit = "%"
            )
            metricRepository.save(utilizationEntity)
            utilizationEntity.id?.let { savedIds.add(it) }
        }

        // Create temperature metric
        val temperatureCelsius = (map["temperatureCelsius"] as? Number)?.toDouble()
        if (temperatureCelsius != null) {
            val tempEntity = MetricEntity(
                id = null,
                timestamp = timestamp,
                receivedAt = receivedAt,
                metricType = MetricType.temperature,
                hostname = hostname,
                gpuId = gpuId,
                gpuName = gpuName,
                userId = user.id,
                gpuDeviceId = device?.id,
                payload = payloadString,
                value = temperatureCelsius,
                unit = "C"
            )
            metricRepository.save(tempEntity)
            tempEntity.id?.let { savedIds.add(it) }
        }

        // Create power metric
        val powerMilliwatts = (map["powerMilliwatts"] as? Number)?.toDouble()
        if (powerMilliwatts != null) {
            val powerEntity = MetricEntity(
                id = null,
                timestamp = timestamp,
                receivedAt = receivedAt,
                metricType = MetricType.power,
                hostname = hostname,
                gpuId = gpuId,
                gpuName = gpuName,
                userId = user.id,
                gpuDeviceId = device?.id,
                payload = payloadString,
                value = powerMilliwatts / 1000.0,
                unit = "W"
            )
            metricRepository.save(powerEntity)
            powerEntity.id?.let { savedIds.add(it) }
        }

        // Create clocks metric
        val graphicsClockMHz = (map["graphicsClockMHz"] as? Number)?.toDouble()
        if (graphicsClockMHz != null) {
            val clocksEntity = MetricEntity(
                id = null,
                timestamp = timestamp,
                receivedAt = receivedAt,
                metricType = MetricType.clocks,
                hostname = hostname,
                gpuId = gpuId,
                gpuName = gpuName,
                userId = user.id,
                gpuDeviceId = device?.id,
                payload = payloadString,
                value = graphicsClockMHz,
                unit = "MHz"
            )
            metricRepository.save(clocksEntity)
            clocksEntity.id?.let { savedIds.add(it) }
        }

        return Pair(savedIds.size, savedIds)
    }

    private fun persistProcessMetric(userId: Long, map: MutableMap<String, Any?>, receivedAt: OffsetDateTime): Pair<Int, List<Long>> {
        logger.debug("persistProcessMetric called for userId=$userId with data: $map")

        try {
            // Extract common fields
            val timestampText = (map["timestamp"] as? String)
                ?: throw IllegalArgumentException("'timestamp' is required")
            var timestamp = OffsetDateTime.parse(timestampText)
            if (timestamp.offset == null) {
                timestamp = timestamp.withOffsetSameInstant(ZoneOffset.UTC)
            }

            val hostname = map["hostname"] as? String
            val gpuId = map["gpuId"] as? String
            val gpuName = map["gpuName"] as? String
            val pid = (map["pid"] as? Number)?.toInt()
                ?: throw IllegalArgumentException("'pid' is required for process metrics")
            val processName = map["processName"] as? String

            // Try multiple field names for memory
            var usedMemoryMiB = (map["processUsedMemoryMiB"] as? Number)?.toDouble()

            logger.debug("Extracted values - pid=$pid, processName=$processName, usedMemoryMiB=$usedMemoryMiB")

            // Validate memory value - reject unrealistic values
            if (usedMemoryMiB != null) {
                val maxRealisticMiB = 1024.0 * 1024.0 // 1 TB in MiB
                if (usedMemoryMiB < 0 || usedMemoryMiB > maxRealisticMiB) {
                    logger.warn("Rejecting process metric with unrealistic memory value: $usedMemoryMiB MiB (${usedMemoryMiB / 1024.0 / 1024.0} TiB) for pid=$pid, processName=$processName")
                    throw IllegalArgumentException("Process memory value is unrealistic: $usedMemoryMiB MiB. Must be between 0 and $maxRealisticMiB MiB (1 TB)")
                }
            }

            // Upsert device for this user
            val user = userRepository.findById(userId).orElseThrow { IllegalArgumentException("Invalid user") }
            val device = if (!gpuId.isNullOrBlank()) {
                gpuDeviceRepository.findByUserIdAndGpuId(user.id!!, gpuId)
                    ?: gpuDeviceRepository.save(com.myoungho.backend.model.GpuDeviceEntity(user = user, gpuId = gpuId, gpuName = gpuName, hostname = hostname))
            } else if (!hostname.isNullOrBlank() && !gpuName.isNullOrBlank()) {
                gpuDeviceRepository.findByUserIdAndHostnameAndGpuName(user.id!!, hostname, gpuName)
                    ?: gpuDeviceRepository.save(com.myoungho.backend.model.GpuDeviceEntity(user = user, gpuId = null, gpuName = gpuName, hostname = hostname))
            } else {
                null
            }

            val payloadString = objectMapper.writeValueAsString(map)

            val processEntity = ProcessMetricEntity(
                id = null,
                timestamp = timestamp,
                receivedAt = receivedAt,
                hostname = hostname,
                gpuId = gpuId,
                gpuName = gpuName,
                userId = user.id,
                gpuDeviceId = device?.id,
                pid = pid,
                processName = processName,
                usedMemoryMiB = usedMemoryMiB,
                payload = payloadString
            )

            logger.debug("Attempting to save ProcessMetricEntity: {}", processEntity)
            val saved = processMetricRepository.save(processEntity)
            logger.info("Successfully saved process metric with id=${saved.id}, pid=$pid, processName=$processName, usedMemoryMiB=$usedMemoryMiB")
            return Pair(1, listOfNotNull(saved.id))
        } catch (e: Exception) {
            logger.error("Error persisting process metric for userId=$userId: ${e.message}", e)
            throw e
        }
    }

    private fun persistNode(userId: Long, item: JsonNode, receivedAt: OffsetDateTime): MetricEntity {
        val map: MutableMap<String, Any?> = objectMapper.convertValue(item)

        val metricTypeText = (map["metric"] as? String)?.trim()
            ?: throw IllegalArgumentException("'metric' field is required")
        val metricType = MetricType.entries.firstOrNull { it.value.equals(metricTypeText, ignoreCase = true) }
            ?: throw IllegalArgumentException("Unknown metric type: $metricTypeText")

        // Normalize differences by type
        when (metricType) {
            MetricType.memory -> {
                val total = (map["totalMiB"] as? Number)?.toDouble()
                val used = (map["usedMiB"] as? Number)?.toDouble()
                if (map["freeMiB"] == null && total != null && used != null) {
                    val free = (total - used).toInt().coerceAtLeast(0)
                    map["freeMiB"] = free
                }
            }
            MetricType.temperature -> {
                if (map["celsius"] == null && map["gpuCelsius"] is Number) {
                    map["celsius"] = (map["gpuCelsius"] as Number).toInt()
                }
            }
            MetricType.power -> {
                if (map["watts"] == null && map["milliwatts"] is Number) {
                    val mw = (map["milliwatts"] as Number).toDouble()
                    map["watts"] = mw / 1000.0
                }
            }
            MetricType.clocks -> {
                if (map["memoryMHz"] == null && map["memMHz"] is Number) {
                    map["memoryMHz"] = (map["memMHz"] as Number).toInt()
                }
            }
            MetricType.utilization -> { /* no-op */ }
            MetricType.process -> { /* no-op */ }
        }

        val timestampText = (map["timestamp"] as? String)
            ?: throw IllegalArgumentException("'timestamp' is required")
        var timestamp = OffsetDateTime.parse(timestampText)
        if (timestamp.offset == null) {
            timestamp = timestamp.withOffsetSameInstant(ZoneOffset.UTC)
        }

        val hostname = map["hostname"] as? String
        val gpuId = map["gpuId"] as? String
        val gpuName = map["gpuName"] as? String
        val pid = (map["pid"] as? Number)?.toInt()
        val processName = map["processName"] as? String

        // Upsert device for this user
        val user = userRepository.findById(userId).orElseThrow { IllegalArgumentException("Invalid user") }
        val device = if (!gpuId.isNullOrBlank()) {
            gpuDeviceRepository.findByUserIdAndGpuId(user.id!!, gpuId)
                ?: gpuDeviceRepository.save(com.myoungho.backend.model.GpuDeviceEntity(user = user, gpuId = gpuId, gpuName = gpuName, hostname = hostname))
        } else if (!hostname.isNullOrBlank() && !gpuName.isNullOrBlank()) {
            gpuDeviceRepository.findByUserIdAndHostnameAndGpuName(user.id!!, hostname, gpuName)
                ?: gpuDeviceRepository.save(com.myoungho.backend.model.GpuDeviceEntity(user = user, gpuId = null, gpuName = gpuName, hostname = hostname))
        } else {
            null
        }

        // Compute value/unit from the normalized map converted back to JsonNode for reuse
        val normalizedNode = objectMapper.valueToTree<JsonNode>(map)
        val (value, unit) = computeValueUnit(metricType, normalizedNode)

        // Keep normalized payload (with derived fields) as stored JSON
        val payloadString = objectMapper.writeValueAsString(map)

        val entity = MetricEntity(
            id = null,
            timestamp = timestamp,
            receivedAt = receivedAt,
            metricType = metricType,
            hostname = hostname,
            gpuId = gpuId,
            gpuName = gpuName,
            userId = user.id,
            gpuDeviceId = device?.id,
            payload = payloadString,
            value = value,
            unit = unit,
        )
        return metricRepository.save(entity)
    }

    private fun computeValueUnit(type: MetricType, node: JsonNode): Pair<Double?, String?> {
        return when (type) {
            MetricType.memory -> node.get("usedMiB")?.asDouble()?.let { it to "MiB" } ?: (null to null)
            MetricType.utilization -> node.get("gpuPercent")?.asDouble()?.let { it to "%" } ?: (null to null)
            MetricType.temperature -> node.get("celsius")?.asDouble()?.let { it to "C" } ?: (null to null)
            MetricType.power -> node.get("watts")?.asDouble()?.let { it to "W" } ?: (null to null)
            MetricType.clocks -> {
                val g = node.get("graphicsMHz")?.asDouble()
                val m = node.get("memoryMHz")?.asDouble()
                val s = node.get("smMHz")?.asDouble()
                val v = g ?: m ?: s
                if (v != null) v to "MHz" else null to null
            }
            MetricType.process -> node.get("usedMiB")?.asDouble()?.let { it to "MiB" } ?: (null to null)
        }
    }

    private fun toOut(e: MetricEntity): Map<String, Any?> {
        val payloadMap: Map<String, Any?> = try {
            objectMapper.readValue(e.payload, Map::class.java) as Map<String, Any?>
        } catch (e: Exception) {
            mapOf("raw" to e.message)
        }
        return mapOf(
            "id" to e.id,
            "timestamp" to e.timestamp.toString(),
            "received_at" to e.receivedAt.toString(),
            "metric" to e.metricType.value,
            "payload" to payloadMap,
        )
    }

    private fun toProcessOut(e: ProcessMetricEntity): Map<String, Any?> {
        val payloadMap: Map<String, Any?> = try {
            objectMapper.readValue(e.payload, Map::class.java) as Map<String, Any?>
        } catch (ex: Exception) {
            mapOf("raw" to ex.message)
        }
        return mapOf(
            "id" to e.id,
            "timestamp" to e.timestamp.toString(),
            "received_at" to e.receivedAt.toString(),
            "pid" to e.pid,
            "processName" to e.processName,
            "usedMemoryMiB" to e.usedMemoryMiB,
            "payload" to payloadMap,
        )
    }
}
