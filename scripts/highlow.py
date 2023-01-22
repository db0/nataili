from nataili.clip.predictor import PredictorPrepare

pp = PredictorPrepare()

pp(
    model_name="ViT-H-14",
    input_directory="X:/valse/high",
    output_directory="X:/valse/high",
    rating_source="none",
    x_only=True,
)

pp(
    model_name="ViT-H-14",
    input_directory="X:/valse/low",
    output_directory="X:/valse/low",
    rating_source="none",
    x_only=True,
)
