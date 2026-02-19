from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from datetime import datetime
from backend.app.database import get_db
from backend.app.models.core import Price
from backend.app.schemas import PriceCreate, PriceResponse
from backend.app.models.user import User, UserRole
from backend.app.utils.security import get_current_active_user

router = APIRouter(prefix="/prices", tags=["Prices"])

@router.get("/", response_model=List[PriceResponse])
def get_prices(
    skip: int = 0,
    limit: int = 100,
    product_id: int = None,
    market_id: int = None,
    date_from: datetime = None,
    db: Session = Depends(get_db)
):
    query = db.query(Price)

    if product_id:
        query = query.filter(Price.product_id == product_id)
    if market_id:
        query = query.filter(Price.market_id == market_id)
    if date_from:
        query = query.filter(Price.date >= date_from)

    prices = query.order_by(Price.date.desc()).offset(skip).limit(limit).all()
    return prices

@router.post("/", response_model=PriceResponse)
def create_price(
    price: PriceCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Validar permisos (Técnicos, Admin, SuperAdmin)
    if current_user.role not in [UserRole.TECNICO, UserRole.ADMIN, UserRole.SUPERADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions. Only Technicians can report prices."
        )

    db_price = Price(
        product_id=price.product_id,
        market_id=price.market_id,
        date=price.date,
        price_wholesale=price.price_wholesale,
        price_retail=price.price_retail,
        reporter_id=current_user.id
    )
    db.add(db_price)
    db.commit()
    db.refresh(db_price)
    return db_price
