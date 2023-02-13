import gradio as gr

from nataili.train.dreambooth import DreamboothLoRA

trainer = DreamboothLoRA()


def train(
    base_checkpoint, data_root, project_name, output_dir, max_train_steps, progress=gr.Progress(track_tqdm=True)
):
    trainer.train(
        base_checkpoint=base_checkpoint,
        data_root=data_root,
        project_name=project_name,
        output_dir=output_dir,
        max_train_steps=int(max_train_steps),
        progress_bar=progress,
    )


with gr.Blocks(analytics_enabled=False) as blocks:
    with gr.Row():
        with gr.Column(scale=4):
            base_checkpoint = gr.Textbox(
                label="Base checkpoint",
                placeholder="runwayml/stable-diffusion-v1-5",
                value="runwayml/stable-diffusion-v1-5",
            )
            data_root = gr.Textbox(
                label="Data root",
                placeholder="X:/lora/squishmallow",
            )
            project_name = gr.Textbox(
                label="Project name",
                placeholder="squishmallow",
            )
            output_dir = gr.Textbox(
                label="Output directory",
                placeholder="X:/lora/output",
            )
            max_train_steps = gr.Textbox(
                label="Max train steps",
                placeholder="1000",
            )
            btn = gr.Button("Train")
        with gr.Column(scale=1):
            output_file = gr.Textbox(
                label="Output file",
            )

    btn.click(
        inputs=[
            base_checkpoint,
            data_root,
            project_name,
            output_dir,
            max_train_steps,
        ],
        outputs=[output_file],
        fn=train,
    )

blocks.queue().launch(show_error=True, quiet=False, debug=True)
