package com.myoungho.backend.web

import com.fasterxml.jackson.databind.ObjectMapper
import com.myoungho.backend.service.ApiKeyService
import com.myoungho.backend.service.MetricService
import org.springframework.format.annotation.DateTimeFormat
import org.springframework.http.HttpStatus
import org.springframework.web.bind.annotation.*
import java.time.OffsetDateTime

@RestController
class MetricsController(
    private val metricService: MetricService,
    private val apiKeyService: ApiKeyService,
    private val objectMapper: ObjectMapper,
) {

    data class IngestResponse(val saved: Int, val ids: List<Long>)

    @PostMapping("/metrics")
    @ResponseStatus(HttpStatus.CREATED)
    fun ingest(
        @RequestHeader(name = "X-API-Key", required = true) apiKey: String?,
        @RequestBody body: String,
    ): IngestResponse {
        val resolved = apiKeyService.resolveActive(apiKey) ?: throw AuthController.UnauthorizedException()
        val result = metricService.ingestForUser(resolved.user.id!!, body)
        val saved = (result["saved"] as Number).toInt()
        val ids = (result["ids"] as List<*>).mapNotNull { (it as? Number)?.toLong() }
        return IngestResponse(saved = saved, ids = ids)
    }

    @GetMapping("/metrics")
    fun list(
        @RequestHeader(name = "X-API-Key", required = true) apiKey: String?,
        @RequestParam(required = false) gpuId: String?,
        @RequestParam(required = false) appName: String?,
        @RequestParam(required = false) tag: String?,
        @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) start: OffsetDateTime?,
        @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) end: OffsetDateTime?,
        @RequestParam(required = false, defaultValue = "100") limit: Int,
        @RequestParam(required = false, defaultValue = "desc") order: String,
    ): Map<String, Any?> {
        //val resolved = apiKeyService.resolveActive(apiKey) ?: throw AuthController.UnauthorizedException()
        val q = MetricService.QueryParams(
            start = start,
            end = end,
            limit = limit.coerceIn(1, 10000),
            order = order,
            gpuId = gpuId,
            //userId = resolved.user.id,
            userId = 1,
            appName = appName,
            tag = tag,
        )
        return metricService.getProcessMetrics(q)
    }

    @GetMapping("/gpus")
    fun gpus(
        @RequestHeader(name = "X-API-Key", required = true) apiKey: String?,
        @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) start: OffsetDateTime?,
        @RequestParam(required = false) @DateTimeFormat(iso = DateTimeFormat.ISO.DATE_TIME) end: OffsetDateTime?,
    ): Map<String, Any?> {
        val resolved = apiKeyService.resolveActive(apiKey) ?: throw AuthController.UnauthorizedException()
        val userId: Long? = resolved.user.id
        return metricService.listUniqueGpus(start, end, userId)
    }
}
