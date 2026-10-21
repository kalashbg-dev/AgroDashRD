from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session
from typing import List
from backend.app.database import get_db
from backend.app.models.core import Market
from backend.app.schemas import MarketCreate, MarketResponse
from backend.app.models.user import User, UserRole
from backend.app.utils.security import get_current_active_user

router = APIRouter(prefix="/markets", tags=["Markets"])

@router.get("/", response_model=List[MarketResponse])
def get_markets(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    markets = db.query(Market).filter(Market.is_active == True).offset(skip).limit(limit).all()
    return markets

@router.post("/", response_model=MarketResponse)
def create_market(
    market: MarketCreate,
    current_user: User = Depends(get_current_active_user),
    db: Session = Depends(get_db)
):
    # Validar permisos (Solo Admin o SuperAdmin)
    if current_user.role not in [UserRole.ADMIN, UserRole.SUPERADMIN]:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Not enough permissions"
        )

    db_market = Market(
        name=market.name,
        location=market.location,
        type=market.type
    )
    db.add(db_market)
    db.commit()
    db.refresh(db_market)
    return db_market
