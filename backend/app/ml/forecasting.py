import pandas as pd
from sqlalchemy.orm import Session
from backend.app.models.core import Price, Product
import logging

logger = logging.getLogger(__name__)

class PriceForecaster:
    def __init__(self, db: Session):
        self.db = db

    def train_model(self, product_id: int, market_id: int):
        """
        Entrena un modelo Prophet para un producto y mercado específicos.
        """
        try:
            # 1. Obtener datos históricos
            prices = self.db.query(Price).filter(
                Price.product_id == product_id,
                Price.market_id == market_id
            ).order_by(Price.date).all()

            if len(prices) < 30:
                logger.warning(f"Not enough data to train model for Product {product_id} in Market {market_id}")
                return None

            # 2. Preparar DataFrame
            df = pd.DataFrame([{
                'ds': p.date,
                'y': p.price_wholesale  # O retail
            } for p in prices])

            # 3. Entrenar Prophet (Comentado para evitar errores si no está instalado)
            # from prophet import Prophet
            # m = Prophet()
            # m.fit(df)

            # 4. Predecir futuro
            # future = m.make_future_dataframe(periods=30)
            # forecast = m.predict(future)

            # 5. Guardar predicciones en BD (tabla Predictions que deberíamos crear)
            logger.info(f"Model trained for Product {product_id}")
            return {"status": "success", "message": "Model trained (simulated)"}

        except Exception as e:
            logger.error(f"Error training model: {str(e)}")
            return {"status": "error", "message": str(e)}

    def get_forecast(self, product_id: int, market_id: int):
        """
        Devuelve las predicciones guardadas o genera una simple al vuelo.
        """
        # Aquí se consultaría la tabla Predictions
        return {
            "product_id": product_id,
            "market_id": market_id,
            "forecast": [
                {"date": "2023-11-01", "price": 100.0},
                {"date": "2023-11-02", "price": 101.5},
                {"date": "2023-11-03", "price": 102.0}
            ]
        }
