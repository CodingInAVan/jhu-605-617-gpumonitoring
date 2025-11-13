package com.myoungho.backend.web

import com.myoungho.backend.model.GpuDeviceEntity
import com.myoungho.backend.repo.GpuDeviceRepository
import com.myoungho.backend.service.ApiKeyService
import org.springframework.http.HttpStatus
import org.springframework.web.bind.annotation.*

@RestController
class GpuController(
    private val apiKeyService: ApiKeyService,
    private val gpuDeviceRepository: GpuDeviceRepository,
) {
    data class GpuDto(
        val id: Long?,
        val gpuId: String?,
        val gpuName: String?,
        val hostname: String?,
        val notes: String?,
        val active: Boolean,
    )

    data class CreateGpuRequest(
        val gpuId: String? = null,
        val gpuName: String? = null,
        val hostname: String? = null,
        val notes: String? = null,
    )

    data class UpdateGpuRequest(
        val gpuName: String? = null,
        val hostname: String? = null,
        val notes: String? = null,
        val active: Boolean? = null,
    )

    private fun toDto(e: GpuDeviceEntity) = GpuDto(
        id = e.id,
        gpuId = e.gpuId,
        gpuName = e.gpuName,
        hostname = e.hostname,
        notes = e.notes,
        active = e.active,
    )

    @GetMapping("/gpus")
    fun list(@RequestHeader(name = "X-API-Key", required = false) apiKey: String?): org.springframework.http.ResponseEntity<Map<String, Any?>> {
        val resolved = apiKeyService.resolveActive(apiKey)
            ?: return org.springframework.http.ResponseEntity.status(HttpStatus.UNAUTHORIZED).body(mapOf("message" to "Invalid API key"))
        val list = gpuDeviceRepository.listForUser(resolved.user.id!!).map { toDto(it) }
        return org.springframework.http.ResponseEntity.ok(mapOf("gpus" to list))
    }

    @PostMapping("/gpus")
    fun create(@RequestHeader(name = "X-API-Key", required = false) apiKey: String?, @RequestBody req: CreateGpuRequest): org.springframework.http.ResponseEntity<GpuDto> {
        val resolved = apiKeyService.resolveActive(apiKey)
            ?: return org.springframework.http.ResponseEntity.status(HttpStatus.UNAUTHORIZED).build()
        val user = resolved.user
        val entity = GpuDeviceEntity(
            user = user,
            gpuId = req.gpuId?.takeIf { it.isNotBlank() },
            gpuName = req.gpuName?.takeIf { it.isNotBlank() },
            hostname = req.hostname?.takeIf { it.isNotBlank() },
            notes = req.notes?.takeIf { it.isNotBlank() },
            active = true,
        )
        val saved = gpuDeviceRepository.save(entity)
        return org.springframework.http.ResponseEntity.status(HttpStatus.CREATED).body(toDto(saved))
    }

    @PutMapping("/gpus/{id}")
    fun update(@RequestHeader(name = "X-API-Key", required = false) apiKey: String?, @PathVariable id: Long, @RequestBody req: UpdateGpuRequest): org.springframework.http.ResponseEntity<GpuDto> {
        val resolved = apiKeyService.resolveActive(apiKey)
            ?: return org.springframework.http.ResponseEntity.status(HttpStatus.UNAUTHORIZED).build()
        val entity = gpuDeviceRepository.findById(id).orElseThrow { org.springframework.web.server.ResponseStatusException(HttpStatus.NOT_FOUND) }
        if (entity.user.id != resolved.user.id) return org.springframework.http.ResponseEntity.status(HttpStatus.FORBIDDEN).build()
        req.gpuName?.let { entity.gpuName = it }
        req.hostname?.let { entity.hostname = it }
        req.notes?.let { entity.notes = it }
        req.active?.let { entity.active = it }
        val saved = gpuDeviceRepository.save(entity)
        return org.springframework.http.ResponseEntity.ok(toDto(saved))
    }

    @DeleteMapping("/gpus/{id}")
    fun delete(@RequestHeader(name = "X-API-Key", required = false) apiKey: String?, @PathVariable id: Long): org.springframework.http.ResponseEntity<Void> {
        val resolved = apiKeyService.resolveActive(apiKey)
            ?: return org.springframework.http.ResponseEntity.status(HttpStatus.UNAUTHORIZED).build()
        val entity = gpuDeviceRepository.findById(id).orElseThrow { org.springframework.web.server.ResponseStatusException(HttpStatus.NOT_FOUND) }
        if (entity.user.id != resolved.user.id) return org.springframework.http.ResponseEntity.status(HttpStatus.FORBIDDEN).build()
        entity.active = false
        gpuDeviceRepository.save(entity)
        return org.springframework.http.ResponseEntity.noContent().build()
    }
}
