"""scene-caption: JoyCaption, always on, for the <SCENE> half of a caption (wanly-console#572).

Nothing is imported here on purpose: `store` must stay importable with the stdlib alone, so
`download_models.sh --scene-caption` and `python3 -m ...scene_caption.store` work on a host
that has no httpx/fastapi. The registry imports service.scene_caption_group directly.
"""
