import pandas as pd
import torch
from pytorch_forecasting import TimeSeriesDataSet, TemporalFusionTransformer
from pytorch_forecasting.metrics import RMSE
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from typing import Dict, Any

#load data
df = pd.read_csv("TFTtrain_data.csv")

#convert columns to correct types. 
df['cell_id_enc'] = df['cell_id_enc'].astype(str) #str for categorical variables
df['day_of_week'] = df['day_of_week'].astype(str)
df['month'] = df['month'].astype(str)
df['crime_count'] = df['crime_count'].astype(float) #float so prediction type matches with dataset type

#create time series dataset
max_encoder_length = 30
max_prediction_length = 7

training = TimeSeriesDataSet(
    df,
    time_idx="time_idx",
    target="crime_count",
    group_ids=["cell_id"],
    min_encoder_length=max_encoder_length,
    max_encoder_length=max_encoder_length,
    min_prediction_length=max_prediction_length,
    max_prediction_length=max_prediction_length,
    static_categoricals=["cell_id_enc"],
    time_varying_known_categoricals=["day_of_week", "month"],
    time_varying_known_reals=["time_idx"],
    time_varying_unknown_categoricals=[],
    time_varying_unknown_reals=["crime_count"],
    target_normalizer=None,
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

#create dataloader
batch_size = 64
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=0)

#lightning module
class TFTLightningWrapper(pl.LightningModule):
    def __init__(self, model: TemporalFusionTransformer):
        super().__init__()
        self.model = model
        self.loss = RMSE()

    def forward(self, x: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        return self.model(x)

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        #unpack batch
        x, y = batch
        
        #get predictions
        output = self(x)
        
        #extract predictions and actual value
        y_hat = output["prediction"]  #TFT outputs predictions in a dictionary
        y_true = y[0]  #Actual values are the first element of y tuple
        
        #calculate loss
        loss = self.loss(y_hat, y_true)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=0.01)

#initialize TFT model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.01,
    hidden_size=64,
    attention_head_size=4,
    dropout=0.1,
    hidden_continuous_size=64,
    output_size=1,
    loss=RMSE(),
)

#wrap TFT model in lightning wrapper
model = TFTLightningWrapper(tft)

#create trainer
trainer = pl.Trainer(
    max_epochs=20,
    accelerator="cpu",
    enable_model_summary=True,
)

#train model
trainer.fit(model, train_dataloaders=train_dataloader)

import os

#save the model
model_save_path = "tft_crime_model.ckpt"
trainer.save_checkpoint(model_save_path)  # Saves weights + hyperparams

#save the dataset parameters (needed for reloading)
training.save("dataset_params.json")
