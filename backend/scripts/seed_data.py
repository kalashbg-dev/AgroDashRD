import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.app.database import SessionLocal, Base, engine
from backend.app.models.core import Market, Product
from sqlalchemy.orm import Session

def seed_data():
    Base.metadata.create_all(bind=engine)
    db: Session = SessionLocal()

    try:
        # --- Mercados de Santo Domingo y el país ---
        markets_data = [
            {"name": "Mercado Nuevo de la Duarte", "type": "Mayorista", "location": "Santo Domingo"},
            {"name": "Merca Santo Domingo", "type": "Mayorista", "location": "Santo Domingo Oeste"},
            {"name": "Mercado Modelo", "type": "Minorista", "location": "Santo Domingo"},
            {"name": "Mercado de Hospedaje Yaque", "type": "Mayorista", "location": "Santiago"},
            {"name": "Mercado de la Feria Ganadera", "type": "Minorista", "location": "Santo Domingo"},
            {"name": "Supermercado Nacional", "type": "Supermercado", "location": "Nacional"},
            {"name": "Supermercado Bravo", "type": "Supermercado", "location": "Nacional"},
        ]

        for m_data in markets_data:
            existing = db.query(Market).filter(Market.name == m_data["name"]).first()
            if not existing:
                market = Market(**m_data)
                db.add(market)
                print(f"Added Market: {market.name}")

        # --- Productos Agrícolas Dominicanos ---
        products_data = [
            {"name": "Plátano Barahonero", "category": "Musáceas", "unit": "Unidad"},
            {"name": "Plátano FHIA-20", "category": "Musáceas", "unit": "Unidad"},
            {"name": "Guineo Verde", "category": "Musáceas", "unit": "Racimo"},
            {"name": "Yuca Mocana", "category": "Raíces", "unit": "Libra"},
            {"name": "Batata", "category": "Raíces", "unit": "Libra"},
            {"name": "Ñame", "category": "Raíces", "unit": "Libra"},
            {"name": "Yautía Blanca", "category": "Raíces", "unit": "Libra"},
            {"name": "Arroz Selecto", "category": "Granos", "unit": "Libra"},
            {"name": "Habichuela Roja", "category": "Granos", "unit": "Libra"},
            {"name": "Guandul Verde", "category": "Granos", "unit": "Libra"},
            {"name": "Tomate de Ensalada", "category": "Vegetales", "unit": "Libra"},
            {"name": "Ají Morrón", "category": "Vegetales", "unit": "Libra"},
            {"name": "Cebolla Roja", "category": "Vegetales", "unit": "Libra"},
            {"name": "Ajo", "category": "Vegetales", "unit": "Libra"},
            {"name": "Aguacate", "category": "Frutas", "unit": "Unidad"},
            {"name": "Limón Persa", "category": "Frutas", "unit": "Docena"},
            {"name": "Piña", "category": "Frutas", "unit": "Unidad"},
            {"name": "Chinola", "category": "Frutas", "unit": "Docena"},
            {"name": "Cacao Hispaniola", "category": "Exportación", "unit": "Quintal"},
            {"name": "Café", "category": "Exportación", "unit": "Quintal"},
        ]

        for p_data in products_data:
            existing = db.query(Product).filter(Product.name == p_data["name"]).first()
            if not existing:
                product = Product(**p_data)
                db.add(product)
                print(f"Added Product: {product.name}")

        db.commit()
        print("Data seeding completed successfully!")

    except Exception as e:
        print(f"Error seeding data: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    seed_data()
