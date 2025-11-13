package com.myoungho.backend.web

import com.myoungho.backend.model.ApiKeyEntity
import com.myoungho.backend.model.UserEntity
import com.myoungho.backend.repo.ApiKeyRepository
import com.myoungho.backend.repo.UserRepository
import org.springframework.http.HttpStatus
import org.springframework.security.crypto.bcrypt.BCrypt
import org.springframework.web.bind.annotation.*
import java.security.SecureRandom

@RestController
class AuthController(
    private val userRepository: UserRepository,
    private val apiKeyRepository: ApiKeyRepository,
) {
    data class RegisterRequest(val email: String, val name: String, val password: String)
    data class RegisterResponse(val id: Long, val email: String, val name: String, val apiKey: String)

    data class LoginRequest(val email: String, val password: String)
    data class LoginResponse(val id: Long, val email: String, val name: String, val apiKey: String)

    @PostMapping("/auth/register")
    @ResponseStatus(HttpStatus.CREATED)
    fun register(@RequestBody req: RegisterRequest): RegisterResponse {
        val email = req.email.trim().lowercase()
        if (email.isBlank() || req.password.isBlank() || req.name.isBlank()) {
            throw IllegalArgumentException("email, name and password are required")
        }
        if (userRepository.findByEmail(email).isPresent) {
            throw IllegalArgumentException("Email already registered")
        }
        val salt = BCrypt.gensalt(12)
        val hash = BCrypt.hashpw(req.password, salt)
        val user = userRepository.save(UserEntity(
            email = email,
            name = req.name.trim(),
            passwordHash = hash,
        ))
        val key = generateApiKey()
        apiKeyRepository.save(ApiKeyEntity(
            user = user,
            apiKey = key,
            active = true,
        ))
        return RegisterResponse(
            id = user.id!!,
            email = user.email,
            name = user.name,
            apiKey = key,
        )
    }

    @PostMapping("/auth/login")
    fun login(@RequestBody req: LoginRequest): LoginResponse {
        val email = req.email.trim().lowercase()
        val password = req.password
        if (email.isBlank() || password.isBlank()) {
            throw IllegalArgumentException("email and password are required")
        }
        val userOpt = userRepository.findByEmail(email)
        if (userOpt.isEmpty) {
            throw UnauthorizedException()
        }
        val user = userOpt.get()
        if (!BCrypt.checkpw(password, user.passwordHash)) {
            throw UnauthorizedException()
        }
        val existingKey = apiKeyRepository.findFirstByUserAndActiveIsTrue(user)
        val apiKey = if (existingKey.isPresent) existingKey.get().apiKey else {
            val key = generateApiKey()
            apiKeyRepository.save(ApiKeyEntity(
                user = user,
                apiKey = key,
                active = true,
            ))
            key
        }
        return LoginResponse(
            id = user.id!!,
            email = user.email,
            name = user.name,
            apiKey = apiKey,
        )
    }

    @ResponseStatus(HttpStatus.UNAUTHORIZED)
    class UnauthorizedException: RuntimeException("Unauthorized")

    private fun generateApiKey(): String {
        val bytes = ByteArray(24)
        SecureRandom().nextBytes(bytes)
        return bytes.joinToString("") { String.format("%02x", it) }
    }
}
