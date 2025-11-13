package com.myoungho.backend.web

import com.myoungho.backend.model.MetricType
import com.myoungho.backend.service.MetricService
import org.slf4j.LoggerFactory
import org.springframework.http.MediaType
import org.springframework.web.bind.annotation.*
import java.time.OffsetDateTime

@RestController
class MetricsController(
    private val metricService: MetricService,
    private val apiKeyService: com.myoungho.backend.service.ApiKeyService,
) {
    private val logger = LoggerFactory.getLogger(MetricsController::class.java)

    @GetMapping("/metrics")
    fun getMetrics(
        @RequestHeader(name = "X-API-Key", required = false) apiKey: String?,
        @RequestParam(required = false) metric: MetricType?,
        @RequestParam(required = false) start: OffsetDateTime?,
        @RequestParam(required = false) end: OffsetDateTime?,
        @RequestParam(required = false, defaultValue = "500") limit: Int,
        @RequestParam(required = false, defaultValue = "desc") order: String,
        @RequestParam(required = false) aggregate: String?,
        @RequestParam(required = false) field: String?,
        @RequestParam(required = false) hostname: String?,
        @RequestParam(required = false) gpuId: String?,
        @RequestParam(required = false) gpuName: String?,
    ): org.springframework.http.ResponseEntity<Map<String, Any?>> {
        val resolved = apiKeyService.resolveActive(apiKey)
            ?: return org.springframework.http.ResponseEntity.status(org.springframework.http.HttpStatus.UNAUTHORIZED)
                .body(mapOf("message" to "Invalid API key"))
        val q = MetricService.QueryParams(
            metric = metric,
            start = start,
            end = end,
            limit = limit.coerceIn(1, 10_000),
            order = order,
            aggregate = aggregate,
            field = field,
            hostname = hostname,
            gpuId = gpuId,
            gpuName = gpuName,
            userId = resolved.user.id,
        )
        return org.springframework.http.ResponseEntity.ok(metricService.getMetrics(q))
    }

    @GetMapping("/process-metrics")
    fun getProcessMetrics(
        @RequestHeader(name = "X-API-Key", required = false) apiKey: String?,
        @RequestParam(required = false) start: OffsetDateTime?,
        @RequestParam(required = false) end: OffsetDateTime?,
        @RequestParam(required = false, defaultValue = "500") limit: Int,
        @RequestParam(required = false, defaultValue = "desc") order: String,
        @RequestParam(required = false) hostname: String?,
        @RequestParam(required = false) gpuId: String?,
        @RequestParam(required = false) gpuName: String?,
    ): org.springframework.http.ResponseEntity<Map<String, Any?>> {
        val resolved = apiKeyService.resolveActive(apiKey)
            ?: return org.springframework.http.ResponseEntity.status(org.springframework.http.HttpStatus.UNAUTHORIZED)
                .body(mapOf("message" to "Invalid API key"))
        val q = MetricService.QueryParams(
            metric = null,
            start = start,
            end = end,
            limit = limit.coerceIn(1, 10_000),
            order = order,
            aggregate = null,
            field = null,
            hostname = hostname,
            gpuId = gpuId,
            gpuName = gpuName,
            userId = resolved.user.id,
        )
        return org.springframework.http.ResponseEntity.ok(metricService.getProcessMetrics(q))
    }

    @PostMapping("/metrics", consumes = [MediaType.APPLICATION_JSON_VALUE])
    fun postMetrics(
        @RequestHeader(name = "X-API-Key", required = false) apiKey: String?,
        @RequestBody body: String,
    ): org.springframework.http.ResponseEntity<Map<String, Any?>> {
        logger.debug("Received POST /metrics request")
        logger.debug("API Key present: ${apiKey != null}")
        logger.debug("Request body length: ${body.length}")
        logger.debug("Request body: $body")

        val resolved = apiKeyService.resolveActive(apiKey)
        if (resolved == null) {
            logger.warn("Invalid or missing API key")
            return org.springframework.http.ResponseEntity.status(org.springframework.http.HttpStatus.UNAUTHORIZED)
                .body(mapOf("message" to "Invalid API key"))
        }

        logger.info("Processing metrics for user: ${resolved.user.id} (${resolved.user.email})")

        try {
            // Fix improperly escaped backslashes in JSON (e.g., Windows paths)
            // Replace single backslashes with double backslashes, but preserve already-escaped sequences
            val fixedBody = body.replace(Regex("""\\([^"\\/bfnrtu])""")) { matchResult ->
                "\\\\" + matchResult.groupValues[1]
            }

            if (fixedBody != body) {
                logger.debug("Fixed improperly escaped JSON. Original char 200-250: ${body.substring(200.coerceAtMost(body.length), 250.coerceAtMost(body.length))}")
                logger.debug("Fixed body: $fixedBody")
            }

            val resp = metricService.ingestForUser(resolved.user.id!!, fixedBody)
            logger.info("Successfully ingested metrics: $resp")
            return org.springframework.http.ResponseEntity.status(org.springframework.http.HttpStatus.CREATED).body(resp)
        } catch (e: Exception) {
            logger.error("Error ingesting metrics: ${e.message}", e)
            return org.springframework.http.ResponseEntity.status(org.springframework.http.HttpStatus.BAD_REQUEST)
                .body(mapOf("message" to "Error ingesting metrics: ${e.message}"))
        }
    }
}
