-- Create application roles with least-privilege access
-- This script runs during postgres container initialization
-- Tables don't exist yet, so we use ALTER DEFAULT PRIVILEGES to grant
-- access to any tables the simulator role creates in the future.

-- Inference role: can read clinical data but NOT ground truth
CREATE ROLE inference_user WITH LOGIN PASSWORD 'inference_pass';
GRANT CONNECT ON DATABASE hospital TO inference_user;
GRANT USAGE ON SCHEMA public TO inference_user;
ALTER DEFAULT PRIVILEGES FOR ROLE simulator IN SCHEMA public GRANT SELECT ON TABLES TO inference_user;
ALTER DEFAULT PRIVILEGES FOR ROLE simulator IN SCHEMA public GRANT INSERT ON TABLES TO inference_user;
ALTER DEFAULT PRIVILEGES FOR ROLE simulator IN SCHEMA public GRANT USAGE ON SEQUENCES TO inference_user;

-- Benchmark role: can read ground truth + inference results, write benchmark results
CREATE ROLE benchmark_user WITH LOGIN PASSWORD 'benchmark_pass';
GRANT CONNECT ON DATABASE hospital TO benchmark_user;
GRANT USAGE ON SCHEMA public TO benchmark_user;
ALTER DEFAULT PRIVILEGES FOR ROLE simulator IN SCHEMA public GRANT SELECT ON TABLES TO benchmark_user;
ALTER DEFAULT PRIVILEGES FOR ROLE simulator IN SCHEMA public GRANT INSERT ON TABLES TO benchmark_user;
ALTER DEFAULT PRIVILEGES FOR ROLE simulator IN SCHEMA public GRANT USAGE ON SEQUENCES TO benchmark_user;

-- Simulator role: full access to all future tables
ALTER DEFAULT PRIVILEGES FOR ROLE simulator IN SCHEMA public GRANT ALL ON TABLES TO simulator;
ALTER DEFAULT PRIVILEGES FOR ROLE simulator IN SCHEMA public GRANT ALL ON SEQUENCES TO simulator;
