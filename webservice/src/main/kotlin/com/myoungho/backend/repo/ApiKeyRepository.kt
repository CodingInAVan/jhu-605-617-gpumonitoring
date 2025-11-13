package com.myoungho.backend.repo

import com.myoungho.backend.model.ApiKeyEntity
import com.myoungho.backend.model.UserEntity
import org.springframework.data.jpa.repository.JpaRepository
import java.util.*

interface ApiKeyRepository : JpaRepository<ApiKeyEntity, Long> {
    fun findByApiKeyAndActiveIsTrue(apiKey: String): Optional<ApiKeyEntity>
    fun findFirstByUserAndActiveIsTrue(user: UserEntity): Optional<ApiKeyEntity>
}