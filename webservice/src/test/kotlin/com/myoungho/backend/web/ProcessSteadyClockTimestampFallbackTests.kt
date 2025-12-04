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
class ProcessSteadyClockTimestampFallbackTests @Autowired constructor(
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
    fun steadyClockLikeTsFallsBackToReceivedAt() {
        val apiKey = register("steady@example.com", "Steady")

        // Simulate C++ steady_clock nanoseconds since arbitrary epoch (~21 days)
        val steadyNs = 1_819_545_669_575_300L

        val event = mapOf(
            "type" to "process_sample",
            "pid" to 12345,
            "app" to "steady-demo",
            "ts_ns" to steadyNs,
            "used_mib" to 321,
            "devices" to listOf(mapOf("id" to 0, "name" to "GPU X", "uuid" to "GPU-STEADY"))
        )

        postJson(apiKey, "/metrics", event)

        val before = System.currentTimeMillis()
        val resp = getJson(apiKey, "/metrics?limit=1&order=desc")
        val items = resp.get("items")
        assertTrue(items.size() >= 1)
        val tsStr = items[0].get("timestamp").asText()
        val parsedMs = java.time.OffsetDateTime.parse(tsStr).toInstant().toEpochMilli()
        val after = System.currentTimeMillis()

        // Expect parsed timestamp to be close to now (receivedAt), within 10s
        val min = before - 10_000
        val max = after + 10_000
        assertTrue(parsedMs in min..max, "Timestamp should fallback near now. ts=$tsStr parsedMs=$parsedMs window=[$min,$max]")
    }
}
