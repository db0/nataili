from nataili.train.dreambooth import DreamboothLoRA

trainer = DreamboothLoRA()

trainer.train(
    base_checkpoint= "runwayml/stable-diffusion-v1-5",
    data_root= "X:/lora/squishmallow",
    project_name= "squishmallow",
    output_dir= "X:/lora/output",
    max_train_steps= 1000,
)