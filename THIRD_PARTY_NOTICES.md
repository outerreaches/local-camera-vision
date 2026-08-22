# Third-party notices

The MIT license in this repository applies only to original Local Camera Vision code.

This project uses or integrates with third-party software and models. Install, download, and use them under their own licenses and terms:

- [FastAPI](https://github.com/fastapi/fastapi), [Uvicorn](https://github.com/encode/uvicorn), [Pydantic](https://github.com/pydantic/pydantic), [PyYAML](https://github.com/yaml/pyyaml), and [HTTPX](https://github.com/encode/httpx) are Python dependencies with their own upstream licenses.
- [FFmpeg](https://ffmpeg.org/) is an external runtime dependency.
- [Ultralytics](https://github.com/ultralytics/ultralytics) and the selected YOLO weights are external detection components with separate licensing and usage terms.
- [Gemma 4 E2B](https://ai.google.dev/gemma) is an external VLM. Consult the current upstream model card and license before downloading or deploying it.
- Reolink is a trademark of its respective owner. This project is not affiliated with or endorsed by Reolink.

No third-party model weights, camera footage, or vendor SDKs are included here.

Dependency versions are intentionally expressed as minimums in `pyproject.toml`; review the resolved dependency tree and upstream notices for a particular deployment.
