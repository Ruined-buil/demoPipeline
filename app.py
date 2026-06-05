import os

from ocr_demo.ui import build_app


os.environ.setdefault("DISABLE_MODELSCOPE", "1")
os.environ.setdefault("PADDLEX_DISABLE_MODELSCOPE", "1")


if __name__ == "__main__":
    build_app().queue(default_concurrency_limit=1).launch(share=False)
