package com.myoungho.backend.model

import jakarta.persistence.*
import java.time.OffsetDateTime
import java.time.ZoneOffset

@Entity
@Table(name = "users", uniqueConstraints = [
    UniqueConstraint(name = "uk_users_email", columnNames = ["email"])
])
class UserEntity(
    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    var id: Long? = null,

    @Column(name = "email", nullable = false, length = 255)
    var email: String,

    @Column(name = "name", nullable = false, length = 255)
    var name: String,

    @Column(name = "password_hash", nullable = false, length = 255)
    var passwordHash: String,

    @Column(name = "created_at", nullable = false)
    var createdAt: OffsetDateTime = OffsetDateTime.now(ZoneOffset.UTC),
)
