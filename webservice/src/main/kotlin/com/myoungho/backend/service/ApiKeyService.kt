package com.myoungho.backend.service

import com.myoungho.backend.model.ApiKeyEntity
import com.myoungho.backend.model.UserEntity
import com.myoungho.backend.repo.ApiKeyRepository
import org.springframework.stereotype.Service
import org.springframework.transaction.annotation.Transactional
import java.time.OffsetDateTime
import java.time.ZoneOffset
import java.util.*

@Service
class ApiKeyService(
    private val apiKeyRepository: ApiKeyRepository,
) {
    data class Resolved(val apiKey: ApiKeyEntity, val user: UserEntity)

    @Transactional
    fun resolveActive(apiKeyText: String?): Resolved? {
        if (apiKeyText.isNullOrBlank()) return null
        val opt = apiKeyRepository.findByApiKeyAndActiveIsTrue(apiKeyText.trim())
        if (opt.isEmpty) return null
        val entity = opt.get()
        entity.lastUsedAt = OffsetDateTime.now(ZoneOffset.UTC)
        return Resolved(entity, entity.user)
    }
}
