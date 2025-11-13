package com.myoungho.backend.web

import com.fasterxml.jackson.databind.JsonNode
import com.fasterxml.jackson.databind.ObjectMapper
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotNull
import org.junit.jupiter.api.MethodOrderer
import org.junit.jupiter.api.Order
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.TestMethodOrder
import org.springframework.beans.factory.annotation.Autowired
import org.springframework.boot.test.autoconfigure.web.servlet.AutoConfigureMockMvc
import org.springframework.boot.test.context.SpringBootTest
import org.springframework.http.MediaType
import org.springframework.test.web.servlet.MockMvc
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders.get
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders.post
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders.put
import org.springframework.test.web.servlet.request.MockMvcRequestBuilders.delete
import org.springframework.test.web.servlet.result.MockMvcResultMatchers.status

@SpringBootTest
@AutoConfigureMockMvc
@TestMethodOrder(MethodOrderer.OrderAnnotation::class)
class ApiKeyFlowTests @Autowired constructor(
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

    private fun postJson(apiKey: String?, uri: String, body: Any, expect: Int = 201): JsonNode {
        val content = body as? String ?: objectMapper.writeValueAsString(body)
        val builder = post(uri).contentType(MediaType.APPLICATION_JSON).content(content)
        if (apiKey != null) builder.header("X-API-Key", apiKey)
        val mvcResult = mockMvc.perform(builder).andExpect(status().`is`(expect)).andReturn()
        return objectMapper.readTree(mvcResult.response.contentAsString.ifBlank { "{}" })
    }

    private fun getJson(apiKey: String?, uri: String, expect: Int = 200): JsonNode {
        val builder = get(uri)
        if (apiKey != null) builder.header("X-API-Key", apiKey)
        val mvcResult = mockMvc.perform(builder).andExpect(status().`is`(expect)).andReturn()
        return objectMapper.readTree(mvcResult.response.contentAsString.ifBlank { "{}" })
    }

    @Test
    @Order(1)
    fun registrationProvidesApiKey() {
        val apiKey = register("user1@example.com", "User1")
        assertNotNull(apiKey)
        assert(apiKey.length >= 32)
    }

    @Test
    @Order(2)
    fun unauthorizedRequestsReturn401() {
        // no header
        getJson(null, "/metrics", 401)
        postJson(null, "/metrics", mapOf("metric" to "utilization", "timestamp" to "2025-01-01T00:00:00Z"), 401)
        getJson(null, "/gpus", 401)
        postJson(null, "/gpus", mapOf("gpuName" to "RTX", "hostname" to "h"), 401)
        // invalid header
        getJson("invalid", "/metrics", 401)
        postJson("invalid", "/metrics", mapOf("metric" to "utilization", "timestamp" to "2025-01-01T00:00:00Z"), 401)
    }

    @Test
    @Order(3)
    fun ingestionCreatesDeviceAndScopesMetrics() {
        val key1 = register("scope1@example.com", "Scope1")
        val key2 = register("scope2@example.com", "Scope2")

        val item1 = mapOf(
            "timestamp" to "2025-01-01T00:00:00Z",
            "metric" to "memory",
            "hostname" to "hostA",
            "gpuId" to "GPU-123",
            "gpuName" to "RTX",
            "totalMiB" to 1000,
            "usedMiB" to 100,
        )
        val item2 = mapOf(
            "timestamp" to "2025-01-01T00:05:00Z",
            "metric" to "memory",
            "hostname" to "hostB",
            "gpuName" to "Quadro",
            "usedMiB" to 200,
            "totalMiB" to 1000,
        )
        postJson(key1, "/metrics", item1)
        postJson(key2, "/metrics", item2)

        // each user sees only their own metrics
        val list1 = getJson(key1, "/metrics?metric=memory").get("items")
        assertEquals(1, list1.size())
        assertEquals("GPU-123", list1[0].get("payload").get("gpuId").asText())
        val list2 = getJson(key2, "/metrics?metric=memory").get("items")
        assertEquals(1, list2.size())
        assertEquals("Quadro", list2[0].get("payload").get("gpuName").asText())

        // GPU devices auto-created and listed per user
        val gpus1 = getJson(key1, "/gpus").get("gpus")
        assertEquals(1, gpus1.size())
        assertEquals("GPU-123", gpus1[0].get("gpuId").asText())
        val gpus2 = getJson(key2, "/gpus").get("gpus")
        assertEquals(1, gpus2.size())
        assertEquals("Quadro", gpus2[0].get("gpuName").asText())
    }

    @Test
    @Order(4)
    fun gpuCrudAndAggregates() {
        val key = register("crud@example.com", "CrudUser")
        // Create manually
        val created = postJson(key, "/gpus", mapOf("gpuName" to "ManualGPU", "hostname" to "H", "notes" to "n"))
        val devId = created.get("id").asLong()
        assertNotNull(devId)
        // Update
        val upd = mockMvc.perform(
            put("/gpus/$devId")
                .header("X-API-Key", key)
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(mapOf("notes" to "updated", "active" to false)))
        ).andExpect(status().isOk).andReturn()
        val updNode = objectMapper.readTree(upd.response.contentAsString)
        assertEquals(false, updNode.get("active").asBoolean())

        // Ingest some power metrics and query aggregates
        val items = listOf(
            mapOf("timestamp" to "2025-01-01T00:00:00Z", "metric" to "power", "watts" to 50.0),
            mapOf("timestamp" to "2025-01-01T00:10:00Z", "metric" to "power", "watts" to 150.0),
        )
        postJson(key, "/metrics", items)
        val avg = getJson(key, "/metrics?metric=power&aggregate=avg").get("value").asDouble()
        val max = getJson(key, "/metrics?metric=power&aggregate=max").get("value").asDouble()
        val min = getJson(key, "/metrics?metric=power&aggregate=min").get("value").asDouble()
        assertEquals(100.0, avg, 0.0001)
        assertEquals(150.0, max, 0.0001)
        assertEquals(50.0, min, 0.0001)
    }
}
