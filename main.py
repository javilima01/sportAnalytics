from src.models import TrainConfig
from src.trainer import YOLOFineTuner

def main():
    cfg = TrainConfig(
        data="dataset/football.yaml",
        model_ckpt="models/yolov8x.pt",
        epochs=50,
        imgsz=640,
        batch=16,
        lr0=0.001,
        weight_decay=0.0005,
        patience=20,
        freeze=0,
        device="0",
        project="runs_finetune",
        name="yolov8x_football",
        export_format="pt",
        tensorboard_dir="runs_tensorboard"
    )

    trainer = YOLOFineTuner(cfg)

    trainer.logger.info("Configuration:")
    for k, v in cfg.model_dump().items():
        trainer.logger.info(f"  {k}: {v}")

    trainer.train()
    trainer.validate()
    trainer.export_best()
    trainer.close()

if __name__ == "__main__":
    main()
