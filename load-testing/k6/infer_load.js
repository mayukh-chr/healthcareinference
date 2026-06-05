import http from "k6/http";
import { check, sleep } from "k6";
import { Trend, Counter, Rate } from "k6/metrics";

const latencyTrend = new Trend("inference_latency_ms", true);
const requestCount = new Counter("inference_requests_total");
const errorRate = new Rate("inference_error_rate");

export const options = {
  stages: [
    { duration: "2m", target: 5 },    // ramp up to 5 VUs
    { duration: "5m", target: 10 },   // sustained at 10 VUs
    { duration: "3m", target: 20 },   // ramp to 20 VUs
    { duration: "5m", target: 20 },   // sustained at 20 VUs
    { duration: "2m", target: 0 },    // ramp down
  ],
  thresholds: {
    http_req_duration: ["p(95)<5000"],   // 95th percentile under 5s
    inference_error_rate: ["rate<0.05"], // less than 5% errors
  },
};

const BASE_URL = __ENV.INFERENCE_URL || "http://localhost:8001";
const BENCHMARK_URL = __ENV.BENCHMARK_URL || "http://localhost:8002";

export default function () {
  // Trigger benchmark run on a small batch
  const runRes = http.post(
    `${BENCHMARK_URL}/v1/run?model_name=Qwen%2FQwen3.5-9B&limit=10`,
    null,
    { headers: { "Content-Type": "application/json" } }
  );

  const ok = check(runRes, {
    "benchmark run 200": (r) => r.status === 200,
    "has run_id": (r) => JSON.parse(r.body).run_id !== undefined,
  });

  errorRate.add(!ok);
  requestCount.add(1);
  latencyTrend.add(runRes.timings.duration);

  // Health check the inference service
  const healthRes = http.get(`${BASE_URL}/v1/health`);
  check(healthRes, { "inference healthy": (r) => r.status === 200 });

  sleep(1);
}
