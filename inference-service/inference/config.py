from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    database_url: str = "postgresql+asyncpg://inference_user:inference_pass@localhost:5432/hospital"
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    vllm_base_url: str = "http://localhost:8001/v1"
    model_name: str = "Qwen/Qwen3.5-9B"
    max_tokens: int = 1024
    temperature: float = 0.0
    service_name: str = "hospital-inference"
    log_level: str = "INFO"


settings = Settings()
