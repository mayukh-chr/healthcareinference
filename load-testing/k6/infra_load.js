/**
 * Infra performance benchmark — targets vLLM directly.
 *
 * Measures: TTFT (via http_req_waiting), total latency, tokens/sec,
 * requests/sec, error rate. Run with:
 *   k6 run --env VLLM_URL=http://<alb-dns>:8001 infra_load.js
 */
import http from "k6/http";
import { check, sleep } from "k6";
import { Trend, Counter, Rate, Gauge } from "k6/metrics";

const ttftTrend       = new Trend("vllm_ttft_ms", true);
const e2eLatency      = new Trend("vllm_e2e_latency_ms", true);
const tokensPerSec    = new Trend("vllm_tokens_per_sec");
const requestTotal    = new Counter("vllm_requests_total");
const errorRate       = new Rate("vllm_error_rate");

export const options = {
  scenarios: {
    // Ramp up to find saturation point
    ramp: {
      executor: "ramping-vus",
      stages: [
        { duration: "2m", target: 5 },    // warm-up
        { duration: "3m", target: 10 },   // light load
        { duration: "3m", target: 20 },   // medium load
        { duration: "3m", target: 30 },   // heavy load — triggers HPA
        { duration: "3m", target: 20 },   // scale-down observation
        { duration: "2m", target: 0 },
      ],
    },
  },
  thresholds: {
    // TTFT (http_req_waiting) p95 under 2s
    http_req_waiting: ["p(95)<2000"],
    // E2E p99 under 15s (clinical document is ~800 tokens output)
    vllm_e2e_latency_ms: ["p(99)<15000"],
    vllm_error_rate: ["rate<0.02"],
  },
};

const VLLM_URL = __ENV.VLLM_URL || "http://localhost:8001";
const MODEL    = __ENV.MODEL    || "Qwen/Qwen3.5-9B";

// Representative clinical prompt — short enough to be realistic, long enough to exercise the model
const SYSTEM_PROMPT =
  "You are a clinical AI assistant. Analyze the hospital encounter documents and respond " +
  "with a valid JSON object matching the requested schema. Be precise and clinically accurate.";

const USER_PROMPT =
  "Generate a complete structured clinical summary. " +
  "Return: primary_diagnosis, primary_icd10, secondary_diagnoses, disease_severity " +
  "(mild/moderate/severe/critical), risk_level (low/medium/high/critical), " +
  "medications (list), key_lab_findings (dict of lab: status), discharge_disposition.\n\n" +
  "CLINICAL DOCUMENTS:\n\n" +
  "ADMISSION NOTE — Hospital Day 1\n" +
  "Patient: 68-year-old male admitted with acute decompensated heart failure. " +
  "BP 158/92, HR 102, SpO2 88% on room air. BNP 4200 pg/mL. " +
  "Creatinine 1.8 mg/dL (baseline 1.2). Na 131 mEq/L.\n\n" +
  "MEDICATIONS: Furosemide 80mg IV BID, Lisinopril 10mg daily, Carvedilol 25mg BID, " +
  "Spironolactone 25mg daily.\n\n" +
  "LAB RESULTS: WBC 9.2, Hgb 11.1, BNP 4200, Troponin 0.04, " +
  "Na 131, K 4.1, Cr 1.8, BUN 38.\n\n" +
  "DISCHARGE SUMMARY: Patient stabilised on IV diuresis. " +
  "Echo shows EF 30%. Discharged to skilled nursing facility on day 5.";

export default function () {
  const payload = JSON.stringify({
    model: MODEL,
    messages: [
      { role: "system", content: SYSTEM_PROMPT },
      { role: "user",   content: USER_PROMPT },
    ],
    max_tokens: 512,
    temperature: 0.0,
  });

  const params = {
    headers: { "Content-Type": "application/json" },
    timeout: "30s",
  };

  const res = http.post(`${VLLM_URL}/v1/chat/completions`, payload, params);

  const ok = check(res, {
    "status 200": (r) => r.status === 200,
    "has choices": (r) => {
      try { return JSON.parse(r.body).choices?.length > 0; } catch { return false; }
    },
  });

  errorRate.add(!ok);
  requestTotal.add(1);

  // TTFT = http_req_waiting (time until first byte from server)
  ttftTrend.add(res.timings.waiting);
  e2eLatency.add(res.timings.duration);

  if (ok) {
    try {
      const body = JSON.parse(res.body);
      const completionTokens = body.usage?.completion_tokens ?? 0;
      const durationSec = res.timings.duration / 1000;
      if (durationSec > 0 && completionTokens > 0) {
        tokensPerSec.add(completionTokens / durationSec);
      }
    } catch (_) {}
  }

  sleep(0.5);
}

export function handleSummary(data) {
  const p50  = data.metrics["vllm_e2e_latency_ms"]?.values?.["p(50)"]  ?? 0;
  const p95  = data.metrics["vllm_e2e_latency_ms"]?.values?.["p(95)"]  ?? 0;
  const p99  = data.metrics["vllm_e2e_latency_ms"]?.values?.["p(99)"]  ?? 0;
  const ttft_p50 = data.metrics["vllm_ttft_ms"]?.values?.["p(50)"] ?? 0;
  const ttft_p95 = data.metrics["vllm_ttft_ms"]?.values?.["p(95)"] ?? 0;
  const tps  = data.metrics["vllm_tokens_per_sec"]?.values?.avg ?? 0;
  const rps  = data.metrics["http_reqs"]?.values?.rate ?? 0;
  const errs = data.metrics["vllm_error_rate"]?.values?.rate ?? 0;

  const summary = {
    model: MODEL,
    requests_per_sec: rps.toFixed(2),
    error_rate: errs.toFixed(4),
    ttft_p50_ms: ttft_p50.toFixed(0),
    ttft_p95_ms: ttft_p95.toFixed(0),
    e2e_p50_ms: p50.toFixed(0),
    e2e_p95_ms: p95.toFixed(0),
    e2e_p99_ms: p99.toFixed(0),
    avg_tokens_per_sec: tps.toFixed(1),
  };

  return {
    stdout: JSON.stringify(summary, null, 2) + "\n",
    "infra_results.json": JSON.stringify(summary, null, 2),
  };
}
