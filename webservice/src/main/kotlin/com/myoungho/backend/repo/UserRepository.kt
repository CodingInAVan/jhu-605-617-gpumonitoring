package com.myoungho.backend.repo

import com.myoungho.backend.model.UserEntity
import org.springframework.data.jpa.repository.JpaRepository
import java.util.*

interface UserRepository : JpaRepository<UserEntity, Long> {
    fun findByEmail(email: String): Optional<UserEntity>
}