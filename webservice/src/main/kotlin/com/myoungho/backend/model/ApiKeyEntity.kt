package com.myoungho.backend.model

import jakarta.persistence.*
import java.time.OffsetDateTime
import java.time.ZoneOffset

@Entity
@Table(name = "api_keys", indexes = [
    Index(name = "idx_api_keys_key", columnList = "api_key", unique = true),
    Index(name = "idx_api_keys_user", columnList = "user_id")
])
class ApiKeyEntity(
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    var id: Long? = null,

    @ManyToOne(fetch = FetchType.LAZY)
    @JoinColumn(name = "user_id", nullable = false)
    var user: UserEntity,

    @Column(name = "api_key", nullable = false, length = 128)
    var apiKey: String,

    @Column(name = "active", nullable = false)
    var active: Boolean = true,

    @Column(name = "created_at", nullable = false)
    var createdAt: OffsetDateTime = OffsetDateTime.now(ZoneOffset.UTC),

    @Column(name = "last_used_at")
    var lastUsedAt: OffsetDateTime? = null,
)
