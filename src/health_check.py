from fastapi import Response

class HealthCheck:
    def __init__(self):
        self.is_ready = False

    async def __call__(self):
        if not self.is_ready:
            return Response(status_code=503)
        return {"status": "healthy", "model_ready": self.is_ready}