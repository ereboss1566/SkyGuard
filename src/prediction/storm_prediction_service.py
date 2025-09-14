import os
import joblib
import logging

logger = logging.getLogger(__name__)

class StormPredictionService:
    def __init__(self, model_path: str = "models/optimized"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.imputer = None
        self.last_prediction_time = None
        self.load_models()

    def load_models(self):
        try:
            model_file = os.path.join(self.model_path, "randomforest_optimized_model.pkl")
            scaler_file = os.path.join(self.model_path, "scaler.pkl")
            imputer_file = os.path.join(self.model_path, "imputer.pkl")
            missing = [f for f in [model_file, scaler_file, imputer_file] if not os.path.exists(f)]
            if missing:
                logger.error(f"Missing model files: {missing}")
                raise FileNotFoundError(f"Missing model files: {missing}")
            self.model = joblib.load(model_file)
            self.scaler = joblib.load(scaler_file)
            self.imputer = joblib.load(imputer_file)
            logger.info("Successfully loaded models and preprocessing objects")
        except Exception as e:
            logger.error(f"Error loading models: {e}")
            raise
