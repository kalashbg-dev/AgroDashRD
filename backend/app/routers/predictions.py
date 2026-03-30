from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from backend.app.database import get_db
from backend.app.models.user import User, UserRole
from backend.app.utils.security import get_current_active_user
from backend.app.ml.forecasting import PriceForecaster

router = APIRouter(prefix="/predictions", tags=["ML Predictions"])

@router.post("/train/{product_id}/{market_id}")
def train_prediction_model(
    product_id: int,
    market_id: int,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Validar permisos (Solo Admin/SuperAdmin)
    if current_user.role not in [UserRole.ADMIN, UserRole.SUPERADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions to trigger training"
        )

    forecaster = PriceForecaster(db)
    result = forecaster.train_model(product_id, market_id)
    return result

@router.get("/{product_id}/{market_id}")
def get_prediction(
    product_id: int,
    market_id: int,
    db: Session = Depends(get_db)
):
    forecaster = PriceForecaster(db)
    return forecaster.get_forecast(product_id, market_id)
