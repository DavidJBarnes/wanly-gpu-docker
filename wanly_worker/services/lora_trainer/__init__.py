"""Character-LoRA training as a wanly-services service.

One package rather than scattered modules, because the pieces only make sense together:

    service.py   the Service subclass the supervisor starts
    app.py       the FastAPI child process it starts, and the endpoints #453 asks for
    recipe.py    THE recipe -- one copy, deliberately
    pipeline.py  stage / train / collect
    jobs.py      job state, persisted so a restart can reconcile
    gpu.py       drain the render worker, and give the card back
    poller.py    claim work from wanly-api
"""
from wanly_worker.services.lora_trainer.service import LoraTrainer

__all__ = ["LoraTrainer"]
