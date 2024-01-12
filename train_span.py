from data import load_dataset_unfolding
from torch.utils.data import DataLoader
from ModelManager import get_SPAN, SPAN_Lightning
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
import fire

def main(dataset_path, fold):
    dataset_name = dataset_path.split("/")[-1]
    train_ds, val_ds, test_ds = load_dataset_unfolding(base_folder=dataset_path, fold=fold, ratio=1.0) 

    train_dataloader = DataLoader(train_ds, batch_size=1, num_workers=20)
    val_dataloader = DataLoader(val_ds, batch_size=1, num_workers=20)
    test_dataloader = DataLoader(test_ds, batch_size=1, num_workers=20)

    model = get_SPAN(i2w=train_ds.i2w)

    wandb_logger = WandbLogger(project='Full_Page_OMR', group=f"{dataset_name}", name=f"SPAN_fold_{fold}", log_model=False)

    checkpointer = ModelCheckpoint(dirpath=f"weights/{dataset_name}/SPAN/", filename=f"SPAN_fold_{fold}", 
                                   monitor="val_SER", mode='min',
                                   save_top_k=1, verbose=True)

    trainer = Trainer(max_epochs=5000, check_val_every_n_epoch=5, logger=wandb_logger, callbacks=[checkpointer])

    #trainer = Trainer(max_epochs=5000, check_val_every_n_epoch=5, callbacks=[checkpointer])


    trainer.fit(model, train_dataloader, val_dataloader)


    model = SPAN_Lightning.load_from_checkpoint(checkpointer.best_model_path)

    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    fire.Fire(main)