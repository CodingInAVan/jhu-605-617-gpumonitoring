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
import org.springframework.test.web.servlet.result.MockMvcResultMatchers.status
import org.junit.jupiter.api.BeforeEach
import com.myoungho.backend.repo.MetricRepository
import org.springframework.transaction.annotation.Transactional

@SpringBootTest
@AutoConfigureMockMvc
@TestMethodOrder(MethodOrderer.OrderAnnotation::class)
@org.junit.jupiter.api.Disabled("Replaced by API-key based tests")
class ApiTests @Autowired constructor(
    val mockMvc: MockMvc,
    val objectMapper: ObjectMapper,
    val metricRepository: MetricRepository,
) {

    private var apiKey: String = ""

    @BeforeEach
    fun cleanDb() {
        metricRepository.deleteAll()
        // Register a user and capture API key for authenticated calls
        val regBody = mapOf("email" to "test@example.com", "name" to "Tester", "password" to "secret123")
        val regResult = mockMvc.perform(
            post("/auth/register")
                .contentType(MediaType.APPLICATION_JSON)
                .content(objectMapper.writeValueAsString(regBody))
        )
            .andExpect(status().isCreated)
            .andReturn()
        val node = objectMapper.readTree(regResult.response.contentAsString)
        apiKey = node.get("apiKey").asText()
    }

    private fun postJsonAuth(uri: String, body: Any): JsonNode {
        val content = if (body is String) body else objectMapper.writeValueAsString(body)
        val mvcResult = mockMvc.perform(
            post(uri)
                .contentType(MediaType.APPLICATION_JSON)
                .header("X-API-Key", apiKey)
                .content(content)
        )
            .andExpect(status().isCreated)
            .andReturn()
        return objectMapper.readTree(mvcResult.response.contentAsString)
    }

    private fun getJsonAuth(uri: String): JsonNode {
        val mvcResult = mockMvc.perform(get(uri).header("X-API-Key", apiKey))
            .andExpect(status().isOk)
            .andReturn()
        return objectMapper.readTree(mvcResult.response.contentAsString)
    }

    // Backwards-compat helpers used by older tests; now they funnel through API-key aware methods
    private fun postJson(uri: String, body: Any): JsonNode = postJsonAuth(uri, body)
    private fun getJson(uri: String): JsonNode = getJsonAuth(uri)

    private val T0 = "2025-01-01T00:00:00Z"
    private val T1 = "2025-01-01T00:10:00Z"
    private fun H(name: String) = name // convenience to mark hostnames per test

    @Test
    @Order(1)
    fun `POST single memory metric and derive freeMiB`() {
        val payload = mapOf(
            "timestamp" to T0,
            "metric" to "memory",
            "hostname" to "hostA",
            "gpuId" to "GPU-0",
            "gpuName" to "RTX",
            "totalMiB" to 10240,
            "usedMiB" to 2048,
        )
        val resp = postJsonAuth("/metrics", payload)
        assertEquals(1, resp.get("saved").asInt())
        assertNotNull(resp.get("ids"))

        val items = getJson("/metrics?metric=memory&order=asc").get("items")
        assertEquals(1, items.size())
        val payloadOut = items[0].get("payload")
        assertEquals(10240, payloadOut.get("totalMiB").asInt())
        assertEquals(2048, payloadOut.get("usedMiB").asInt())
        // derived
        assertEquals(8192, payloadOut.get("freeMiB").asInt())
    }

    @Test
    @Order(2)
    fun `Normalization for temperature power clocks`() {
        val items = listOf(
            mapOf(
                "timestamp" to T0,
                "metric" to "temperature",
                "gpuCelsius" to 65,
            ),
            mapOf(
                "timestamp" to T0,
                "metric" to "power",
                "milliwatts" to 115200,
            ),
            mapOf(
                "timestamp" to T0,
                "metric" to "clocks",
                "memMHz" to 7000,
            ),
        )
        val resp = postJsonAuth("/metrics", items)
        assertEquals(3, resp.get("saved").asInt())

        val tempItems = getJson("/metrics?metric=temperature").get("items")
        assertEquals(1, tempItems.size())
        assertEquals(65, tempItems[0].get("payload").get("celsius").asInt())

        val powerItems = getJson("/metrics?metric=power").get("items")
        assertEquals(1, powerItems.size())
        // value/unit are not directly exposed but aggregation uses 'value'
        val maxPower = getJson("/metrics?metric=power&aggregate=max").get("value").asDouble()
        assertEquals(115.2, maxPower, 0.0001)

        val clockItems = getJson("/metrics?metric=clocks").get("items")
        assertEquals(1, clockItems.size())
        assertEquals(7000, clockItems[0].get("payload").get("memoryMHz").asInt())
    }

    @Test
    @Order(3)
    fun `Ordering, limit and filters`() {
        // seed two utilization entries with different timestamps and gpu identifiers
        val items = listOf(
            mapOf(
                "timestamp" to T0,
                "metric" to "utilization",
                "hostname" to "hostX",
                "gpuId" to "A",
                "gpuName" to "Quadro",
                "gpuPercent" to 10,
                "memoryPercent" to 5,
            ),
            mapOf(
                "timestamp" to T1,
                "metric" to "utilization",
                "hostname" to "hostY",
                "gpuId" to "B",
                "gpuName" to "RTX",
                "gpuPercent" to 30,
                "memoryPercent" to 15,
            ),
        )
        postJson("/metrics", items)

        val asc = getJson("/metrics?metric=utilization&order=asc").get("items")
        assertEquals(2, asc.size())
        val tsAsc0 = asc[0].get("timestamp").asText()
        assert(tsAsc0.startsWith("2025-01-01T00:00")) { "expected timestamp starting with 2025-01-01T00:00 but was $tsAsc0" }
        val desc = getJson("/metrics?metric=utilization&order=desc").get("items")
        val tsDesc0 = desc[0].get("timestamp").asText()
        assert(tsDesc0.startsWith("2025-01-01T00:10")) { "expected timestamp starting with 2025-01-01T00:10 but was $tsDesc0" }

        val limited = getJson("/metrics?metric=utilization&limit=1").get("items")
        assertEquals(1, limited.size())

        val filtered = getJson("/metrics?metric=utilization&hostname=hostX&gpuId=A&gpuName=Quadro").get("items")
        assertEquals(1, filtered.size())
        assertEquals("hostX", filtered[0].get("payload").get("hostname").asText())
    }

    @Test
    @Order(4)
    fun `Aggregations avg min max`() {
        // seed power metrics with different values
        val items = listOf(
            mapOf("timestamp" to T0, "metric" to "power", "watts" to 50.0),
            mapOf("timestamp" to T1, "metric" to "power", "watts" to 100.0),
        )
        postJson("/metrics", items)

        val avg = getJson("/metrics?metric=power&aggregate=avg").get("value").asDouble()
        assertEquals(75.0, avg, 0.0001)
        val min = getJson("/metrics?metric=power&aggregate=min").get("value").asDouble()
        assertEquals(50.0, min, 0.0001)
        val max = getJson("/metrics?metric=power&aggregate=max").get("value").asDouble()
        assertEquals(100.0, max, 0.0001)
    }

    @Test
    @Order(5)
    fun `GPUs listing returns distinct identifiers with latest`() {
        // seed two hosts/gpus
        val items = listOf(
            mapOf("timestamp" to T0, "metric" to "temperature", "hostname" to "h1", "gpuId" to "0", "gpuName" to "RTX", "celsius" to 60),
            mapOf("timestamp" to T1, "metric" to "temperature", "hostname" to "h1", "gpuId" to "0", "gpuName" to "RTX", "celsius" to 61),
            mapOf("timestamp" to T0, "metric" to "temperature", "hostname" to "h2", "gpuId" to "1", "gpuName" to "Quadro", "celsius" to 55),
        )
        postJson("/metrics", items)
        val gpus = getJson("/gpus").get("gpus")
        // find rows for h1 and h2 and check presence
        val rowH1 = gpus.find { it.get("hostname").asText() == "h1" && it.get("gpuId").asText() == "0" }
        val rowH2 = gpus.find { it.get("hostname").asText() == "h2" && it.get("gpuId").asText() == "1" }
        assertNotNull(rowH1)
        assertNotNull(rowH2)
        val latestH1 = rowH1!!.get("latest").asText()
        assert(latestH1.startsWith("2025-01-01T00:10")) { "expected latest starting with 2025-01-01T00:10 but was $latestH1" }
    }

    @Test
    @Order(6)
    fun `Stats returns counts and latest per metric`() {
        // add two memory rows to impact stats
        val items = listOf(
            mapOf("timestamp" to T0, "metric" to "memory", "totalMiB" to 1000, "usedMiB" to 100, "freeMiB" to 900),
            mapOf("timestamp" to T1, "metric" to "memory", "totalMiB" to 1000, "usedMiB" to 200, "freeMiB" to 800),
        )
        postJson("/metrics", items)
        val stats = getJson("/stats").get("metrics")
        // At least memory should be present with latest timestamp
        val memory = stats.get("memory")
        assertNotNull(memory)
        val latest = memory.get("latest").asText()
        assert(latest.startsWith("2025-01-01T00:10")) { "expected latest starting with 2025-01-01T00:10 but was $latest" }
        // count should be >= 2 because this test inserted two rows
        val count = memory.get("count").asLong()
        assert(count >= 2) { "expected memory count >= 2 but was $count" }
    }

    @Test
    @Order(7)
    fun `Health endpoint`() {
        val node = getJson("/health")
        assertEquals("ok", node.get("status").asText())
    }
}
