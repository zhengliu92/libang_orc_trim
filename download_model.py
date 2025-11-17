from modelscope import snapshot_download


def download_model_dino_base():
    model_id = "muse/grounding-dino-base"
    model_dir = "model/GroundingDINOBase"
    snapshot_download(
        model_id,
        local_dir=model_dir,
    )


def download_model_idea_dino():
    model_id = "AI-ModelScope/GroundingDINO"
    model_dir = "model/AI-ModelScope/GroundingDINO"
    snapshot_download(
        model_id,
        local_dir=model_dir,
    )


if __name__ == "__main__":
    # download_model_dino_base()
    download_model_idea_dino()
