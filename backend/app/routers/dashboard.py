from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from typing import List, Dict, Any
from backend.app.database import get_db
from backend.app.models.core import Price, Product, Market

router = APIRouter(prefix="/dashboard", tags=["Dashboard"])

@router.get("/summary")
def get_dashboard_summary(db: Session = Depends(get_db)):
    """
    Retorna resumen para el dashboard principal:
    - Precios recientes
    - Productos con mayor variación
    - Estadísticas generales
    """

    # 1. Precios más recientes (últimos 10)
    recent_prices = db.query(Price).order_by(Price.date.desc()).limit(10).all()

    formatted_recent = []
    for p in recent_prices:
        product = db.query(Product).filter(Product.id == p.product_id).first()
        market = db.query(Market).filter(Market.id == p.market_id).first()
        formatted_recent.append({
            "id": p.id,
            "product_name": product.name if product else "Unknown",
            "market_name": market.name if market else "Unknown",
            "date": p.date,
            "price_wholesale": p.price_wholesale,
            "unit": product.unit if product else ""
        })

    # 2. Conteo de entidades
    total_products = db.query(Product).count()
    total_markets = db.query(Market).count()
    total_reports = db.query(Price).count()

    return {
        "recent_prices": formatted_recent,
        "stats": {
            "products": total_products,
            "markets": total_markets,
            "reports": total_reports
        }
    }
