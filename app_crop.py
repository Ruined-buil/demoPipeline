import os

from ocr_demo.crop_ui import build_crop_app


os.environ.setdefault("DISABLE_MODELSCOPE", "1")
os.environ.setdefault("PADDLEX_DISABLE_MODELSCOPE", "1")


if __name__ == "__main__":
    build_crop_app().queue(default_concurrency_limit=1).launch(share=True)
