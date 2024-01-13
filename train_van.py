import fire
from data import load_dataset
from torch.utils.data import DataLoader
from ModelManager import get_VAN, VAN_Lightning
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger


def main(dataset_path, fold):
    dataset_name = dataset_path.split("/")[-1]
    train_ds, val_ds, test_ds = load_dataset(base_folder=dataset_path, fold=fold, ratio=1.0)

    req_iterations = max(train_ds.get_length() + val_ds.get_length() + test_ds.get_length())

    train_ds.set_max_iterations(req_iterations)
    val_ds.set_max_iterations(req_iterations)  
    test_ds.set_max_iterations(req_iterations)  

    train_dataloader = DataLoader(train_ds, batch_size=1, num_workers=20)
    val_dataloader = DataLoader(val_ds, batch_size=1, num_workers=20)
    test_dataloader = DataLoader(test_ds, batch_size=1, num_workers=20)

    model = get_VAN(i2w=train_ds.i2w, max_iterations=req_iterations)

    wandb_logger = WandbLogger(project='Full_Page_OMR', group=dataset_name, name=f"VAN_fold_{fold}_arale", log_model=False)

    checkpointer = ModelCheckpoint(dirpath=f"weights/{dataset_name}/VAN/", filename=f"VAN_fold_{fold}", 
                                   monitor="val_SER", mode='min',
                                   save_top_k=1, verbose=True)

    trainer = Trainer(max_epochs=5000, check_val_every_n_epoch=5, logger=wandb_logger, callbacks=[checkpointer])
    #trainer = Trainer(max_epochs=5000, check_val_every_n_epoch=5, callbacks=[checkpointer])


    trainer.fit(model, train_dataloader, val_dataloader)


    model = VAN_Lightning.load_from_checkpoint(checkpointer.best_model_path)

    trainer.test(model, test_dataloader)


if __name__ == "__main__":
    fire.Fire(main)