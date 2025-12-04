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

@SpringBootTest
@AutoConfigureMockMvc
@ActiveProfiles("test")
@TestMethodOrder(MethodOrderer.OrderAnnotation::class)
class ProcessMetricsApiTests @Autowired constructor(
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
    fun ingestAndQueryProcessMetrics() {
        val apiKey = register("proc@example.com", "ProcUser")

        // Build a small batch of process events using the provided sample (no metricType field)
        val events = listOf(
            mapOf(
                "type" to "scope_begin",
                "pid" to 40820,
                "app" to "heavy_cuda_demo",
                "name" to "DtoH final copy",
                "ts_ns" to 1658011628771700L,
                "devices" to listOf(mapOf(
                    "id" to 0,
                    "name" to "NVIDIA GeForce RTX 5060 Laptop GPU",
                    "uuid" to "GPU-fc2e721b-a3b1-dc5b-63f0-76b7e624f037",
                    "pci_bus" to 2,
                    "used_mib" to 1396,
                    "free_mib" to 6754,
                    "total_mib" to 8150
                ))
            ),
            mapOf(
                "type" to "scope_end",
                "pid" to 40820,
                "app" to "heavy_cuda_demo",
                "name" to "DtoH final copy",
                "ts_start_ns" to 1658011628771700L,
                "ts_end_ns" to 1658011642187100L,
                "duration_ns" to 13415400L,
                "devices" to listOf(mapOf(
                    "id" to 0,
                    "name" to "NVIDIA GeForce RTX 5060 Laptop GPU",
                    "uuid" to "GPU-fc2e721b-a3b1-dc5b-63f0-76b7e624f037",
                    "pci_bus" to 2,
                    "used_mib" to 1396,
                    "free_mib" to 6754,
                    "total_mib" to 8150
                ))
            ),
            mapOf(
                "type" to "kernel",
                "pid" to 41816,
                "app" to "heavy_cuda_demo",
                "device_id" to 0,
                "gpu_name" to "NVIDIA GeForce RTX 5060 Laptop GPU",
                "gpu_uuid" to "GPU-fc2e721b-a3b1-dc5b-63f0-76b7e624f037",
                "kernel" to "kVecAdd",
                "ts_start_ns" to 1658359142071300L,
                "ts_end_ns" to 1658359145417100L,
                "duration_ns" to 3345800L,
                "grid" to listOf(65536,1,1),
                "block" to listOf(256,1,1),
                "shared_mem_bytes" to 0,
                "tag" to "vecadd",
                "cuda_error" to "no error"
            )
        )

        val resp = postJson(apiKey, "/metrics", events)
        assertTrue(resp.get("saved").asInt() >= 2)

        // Query back process metrics via unified /metrics endpoint
        val listResp = getJson(apiKey, "/metrics?limit=50&order=asc&gpuId=GPU-fc2e721b-a3b1-dc5b-63f0-76b7e624f037")
        val items = listResp.get("items")
        assertTrue(items.size() >= 2)
        // Ensure payload and basic fields present
        val first = items[0]
        assertTrue(first.has("timestamp"))
        assertTrue(first.has("pid"))
        assertTrue(first.get("extra").isObject)

        // There should be at least one item with kernelName or processName
        val hasKernelOrProcess = (0 until items.size()).any { idx ->
            val n = items[idx]
            (n.get("processName")?.isTextual == true) || (n.get("extra")?.get("kernel")?.isTextual == true)
        }
        assertTrue(hasKernelOrProcess)
    }

    @Test
    @Order(2)
    fun ingestWithMalformedJsonReturns400() {
        val apiKey = register("badjson@example.com", "BadJsonUser")
        // intentionally invalid JSON payload
        val malformed = "{"
        val builder = post("/metrics")
            .contentType(MediaType.APPLICATION_JSON)
            .header("X-API-Key", apiKey)
            .content(malformed)

        // Only assert HTTP 400; response body contents may vary by Spring configuration
        mockMvc.perform(builder).andExpect(status().isBadRequest)
    }
}
