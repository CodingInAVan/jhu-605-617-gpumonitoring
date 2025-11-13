package com.myoungho.backend.web

import com.myoungho.backend.service.MetricService
import org.springframework.web.bind.annotation.GetMapping
import org.springframework.web.bind.annotation.RestController

@RestController
class GpusController(
    private val metricService: MetricService,
) {
    @GetMapping("/gpus-legacy")
    fun listGpus(): Map<String, Any?> = metricService.listGpus()
}
