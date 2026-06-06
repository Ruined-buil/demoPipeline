import gradio as gr

from .crop_pipeline import process_crop_image


def clear_outputs():
    return None, None, "", ""


def build_crop_app():
    with gr.Blocks(title="CPU Crop OCR Demo") as demo:
        gr.Markdown("# CPU Crop OCR Demo")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Cropped image", type="pil", sources=["upload"])
                with gr.Row():
                    run_button = gr.Button("Run OCR", variant="primary")
                    clear_button = gr.Button("Clear")
                status_output = gr.Textbox(label="Status", interactive=False, lines=3)
            with gr.Column():
                image_preview = gr.Image(label="Uploaded crop", type="filepath")

        final_text = gr.Textbox(label="Final normalized OCR text", interactive=False, lines=8)

        run_button.click(
            fn=process_crop_image,
            inputs=image_input,
            outputs=[image_preview, final_text, status_output],
        )
        clear_button.click(
            fn=clear_outputs,
            inputs=None,
            outputs=[image_input, image_preview, final_text, status_output],
        )

    return demo
