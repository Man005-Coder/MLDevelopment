import pandas as pd
import numpy as np
import os
import sys
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object
from xgboost import XGBRegressor
from dataclasses import dataclass
from sklearn.metrics import r2_score

@dataclass

class model_trainer_config:
    trained_model_file_path: str = os.path.join('artifacts', 'model.pkl')

class model_trainer:
    def __init__(self):
        self.model_trainer_config = model_trainer_config()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Model Training initiated")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1]
            )
            model = XGBRegressor()
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            model_r2_score = r2_score(y_test, y_pred)
            logging.info(f"Model Training completed with R2 score: {model_r2_score}")
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=model
            )
            return model_r2_score, y_pred, model_r2_score
        except Exception as e:
            raise CustomException(e, sys)