from pydantic import BaseModel, Field


class EngineResult(BaseModel):
    engine: str
    text: str
    confidence: float = 0.0


class ConsensusResult(BaseModel):
    value: str | None = None
    score: float = 0.0
    engines_agreed: int = 0
    engine_results: list[EngineResult] = Field(default_factory=list)
    stage: str = "A"
    tier: int = 1
