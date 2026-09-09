"""face-crop as a wanly-services service."""
from __future__ import annotations

import os

from wanly_worker.service import PreflightError, Service

PORT = int(os.environ.get("FACE_CROP_PORT", "8084"))


class FaceCrop(Service):
    name = "face-crop"
    port = PORT
    summary = f"insightface detection and cropping on :{PORT}"

    def preflight(self) -> None:
        try:
            import insightface  # noqa: F401
        except Exception as e:
            raise PreflightError(
                f"insightface is not importable ({e}). It ships in the image; a failure here "
                f"means the image is wrong, not the deployment.")

    def command(self) -> list[str]:
        return ["python3", "-m", "uvicorn",
                "wanly_worker.services.face_crop.app:app",
                "--host", "0.0.0.0", "--port", str(self.port)]

    async def ready(self, client) -> bool:
        try:
            r = await client.get(f"http://127.0.0.1:{self.port}/health", timeout=5)
            return r.status_code == 200
        except Exception:
            return False

    def details(self) -> dict:
        from wanly_worker.services.face_crop import detect as fd
        # Reported here too, so the Workers page can show a box that is up but still fetching
        # its model rather than presenting it as ready to work.
        return {"cos_floor": fd.COS_FLOOR, "pad": fd.PAD, "model_loaded": fd.is_loaded()}
