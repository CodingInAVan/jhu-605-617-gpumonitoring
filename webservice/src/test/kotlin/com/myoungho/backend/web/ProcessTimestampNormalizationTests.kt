package com.myoungho.backend.web

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.MethodOrderer
import org.junit.jupiter.api.Order
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestMethodOrder
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.http.MediaType
import org.springframework.test.context.ActiveProfiles
import org.springframework.test.web.servlet.MockMvc
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post
import org.springframework.test.web.servlet.result.MockMvcResultMatchers.status

@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
@TestMethodOrder(MethodOrderer.OrderAnnotation::class)
class ProcessTimestampNormalizationTests @Autowired constructor(
    val mockMvc: MockMvc,
    val objectMapper: ObjectMapper,
) {
    private fun register(email: String, name: String = "User", password: String = "secret123"): String {
        val body = mapOf("email" to email, "name" to name, "password" to password)
        val mvcResult = mockMvc.perform(
            post("/auth/register").contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(body))
        ).andExpect(status().isCreated).andReturn()
        val node = objectMapper.readTree(mvcResult.response.contentAsString)
        return node.get("apiKey").asText()
    }

    private fun postJson(apiKey: String, uri: String, body: Any, expect: Int = 201): JsonNode {
        val content = body as? String ?: objectMapper.writeValueAsString(body)
        val builder = post(uri).contentType(MediaType.APPLICATION_JSON).content(content).header("X-API-Key", apiKey)
        val mvcResult = mockMvc.perform(builder).andExpect(status().`is`(expect)).andReturn()
        return objectMapper.readTree(mvcResult.response.contentAsString.ifBlank { "{}" })
    }

    private fun getJson(apiKey: String, uri: String, expect: Int = 200): JsonNode {
        val builder = get(uri).header("X-API-Key", apiKey)
        val mvcResult = mockMvc.perform(builder).andExpect(status().`is`(expect)).andReturn()
        return objectMapper.readTree(mvcResult.response.contentAsString.ifBlank { "{}" })
    }

    @Test
    @Order(1)
    fun normalizesMicrosecondsToNs() {
        val apiKey = register("norm@example.com", "NormUser")

        // now in microseconds (producer mistakenly sends us)
        val nowMs = System.currentTimeMillis()
        val nowUs = nowMs * 1_000L

        val event = mapOf(
            "type" to "process_sample",
            "pid" to 9999,
            "app" to "norm-demo",
            "ts_ns" to nowUs, // mislabeled but microseconds magnitude
            "used_mib" to 123,
            "devices" to listOf(mapOf("id" to 0, "name" to "GPU X", "uuid" to "GPU-NORM"))
        )

        postJson(apiKey, "/metrics", event)

        val resp = getJson(apiKey, "/metrics?limit=5&order=desc")
        val items = resp.get("items")
        assertTrue(items.size() >= 1)
        val first = items[0]
        val tsStr = first.get("timestamp").asText()
        // parse ISO back to millis
        val tsMillis = java.time.OffsetDateTime.parse(tsStr).toInstant().toEpochMilli()
        val delta = kotlin.math.abs(tsMillis - nowMs)
        // should be within ~10 seconds
        assertTrue(delta < 10_000, "Timestamp not normalized correctly, delta=$delta ms, ts=$tsStr now=${java.time.Instant.ofEpochMilli(nowMs)}")
    }
}
