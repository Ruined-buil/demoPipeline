import gradio as gr

from .pipeline import process_image


def clear_outputs():
    return None, None, [], "", ""


def build_app():
    with gr.Blocks(title="CPU OCR Demo") as demo:
        gr.Markdown("# CPU OCR Demo")
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(label="Uploaded image", type="pil", sources=["upload"])
                with gr.Row():
                    run_button = gr.Button("Run OCR", variant="primary")
                    clear_button = gr.Button("Clear")
                status_output = gr.Textbox(label="Status", interactive=False, lines=3)
            with gr.Column():
                annotated_output = gr.Image(label="Paddle annotated preview", type="filepath")
                crop_gallery = gr.Gallery(
                    label="Detected text crops",
                    columns=3,
                    object_fit="contain",
                    height=320,
                )

        final_text = gr.Textbox(label="Final raw OCR text", interactive=False, lines=8)

        run_button.click(
            fn=process_image,
            inputs=image_input,
            outputs=[annotated_output, crop_gallery, final_text, status_output],
        )
        clear_button.click(
            fn=clear_outputs,
            inputs=None,
            outputs=[image_input, annotated_output, crop_gallery, final_text, status_output],
        )

    return demo
