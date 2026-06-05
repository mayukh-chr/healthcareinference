from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # Database
    database_url: str = "postgresql+asyncpg://simulator:simulator@localhost:5432/hospital"

    # RabbitMQ
    rabbitmq_url: str = "amqp://guest:guest@localhost:5672/"
    rabbitmq_exchange: str = "hospital.events"

    # Simulation defaults
    default_global_seed: int = 42
    max_concurrent_encounters: int = 50

    # Service
    service_name: str = "hospital-simulator"
    log_level: str = "INFO"


settings = Settings()
