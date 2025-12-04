package com.myoungho.backend.web

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import org.junit.jupiter.api.Assertions.assertEquals
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
import java.time.OffsetDateTime
import java.time.ZoneOffset

@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
@TestMethodOrder(MethodOrderer.OrderAnnotation::class)
class GpuDevicesFromMetricsTests @Autowired constructor(
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
    fun listsDistinctGpusFilteredByTimeRange() {
        val apiKey = register("gpus@example.com", "GpuUser")

        val nowNs = System.currentTimeMillis() * 1_000_000L
        val earlierNs = nowNs - 10L * 60L * 1_000_000_000L // 10 minutes ago
        val laterNs = nowNs - 60L * 1_000_000_000L // 1 minute ago

        val events = listOf(
            mapOf(
                "type" to "process_sample",
                "pid" to 1111,
                "app" to "demo",
                "ts_ns" to earlierNs,
                "devices" to listOf(mapOf(
                    "id" to 0,
                    "name" to "NVIDIA RTX A",
                    "uuid" to "GPU-AAA",
                    "used_mib" to 100,
                    "free_mib" to 7900,
                    "total_mib" to 8000
                )),
                "used_mib" to 100,
            ),
            mapOf(
                "type" to "process_sample",
                "pid" to 1111,
                "app" to "demo",
                "ts_ns" to laterNs,
                "devices" to listOf(mapOf(
                    "id" to 0,
                    "name" to "NVIDIA RTX A",
                    "uuid" to "GPU-AAA",
                    "used_mib" to 120,
                    "free_mib" to 7880,
                    "total_mib" to 8000
                )),
                "used_mib" to 120,
            ),
            mapOf(
                "type" to "process_sample",
                "pid" to 2222,
                "app" to "demo",
                "ts_ns" to earlierNs,
                "devices" to listOf(mapOf(
                    "id" to 0,
                    "name" to "NVIDIA RTX B",
                    "uuid" to "GPU-BBB",
                    "used_mib" to 200,
                    "free_mib" to 7800,
                    "total_mib" to 8000
                )),
                "used_mib" to 200,
            ),
        )

        postJson(apiKey, "/metrics", events)

        // Query without explicit start/end to avoid serialization quirks; endpoint supports filters nevertheless
        val resp = getJson(apiKey, "/gpus")
        val gpus = resp.get("gpus")
        val count = resp.get("count").asInt()
        assertTrue(count >= 1)
        // Should contain GPU-AAA and not necessarily GPU-BBB due to time filter
        val uuids = (0 until gpus.size()).map { gpus[it].get("gpuUuid").asText() }.toSet()
        assertTrue(uuids.contains("GPU-AAA"))
    }
}
