package com.myoungho.backend.web

import com.myoungho.backend.service.MetricService
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RestController

@RestController
class StatsController(
    private val metricService: MetricService,
) {
    @GetMapping("/stats")
    fun stats(): Map<String, Any?> = metricService.stats()
}
