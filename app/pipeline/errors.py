class PipelineStageError(Exception):
    def __init__(self, stage: str, message: str, cause: Exception | None = None):
        self.stage = stage
        self.cause = cause
        super().__init__(message)
