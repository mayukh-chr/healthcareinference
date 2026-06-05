import http from "k6/http";
import { check, sleep } from "k6";
import { Counter, Rate } from "k6/metrics";

const simulationCount = new Counter("simulations_total");
const errorRate = new Rate("simulation_error_rate");

export const options = {
  stages: [
    { duration: "1m", target: 3 },
    { duration: "3m", target: 10 },
    { duration: "1m", target: 0 },
  ],
  thresholds: {
    http_req_duration: ["p(95)<10000"],  // simulation can be slower
    simulation_error_rate: ["rate<0.02"],
  },
};

const SIMULATOR_URL = __ENV.SIMULATOR_URL || "http://localhost:8000";

export default function () {
  const n = Math.floor(Math.random() * 10) + 1;
  const seed = Math.floor(Math.random() * 100000);

  const res = http.post(
    `${SIMULATOR_URL}/v1/simulate?n=${n}&seed=${seed}`,
    null,
    { headers: { "Content-Type": "application/json" } }
  );

  const ok = check(res, {
    "simulate 200": (r) => r.status === 200,
    "has run_id": (r) => {
      try {
        return JSON.parse(r.body).run_id !== undefined;
      } catch {
        return false;
      }
    },
  });

  errorRate.add(!ok);
  simulationCount.add(n);
  sleep(2);
}
