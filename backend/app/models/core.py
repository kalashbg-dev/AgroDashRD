from sqlalchemy import Column, Integer, String, Float, Boolean, ForeignKey, DateTime
from sqlalchemy.orm import relationship
from backend.app.database import Base

class Market(Base):
    __tablename__ = "markets"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    location = Column(String, nullable=True)  # "Lat,Lng" o dirección
    type = Column(String, default="Mayorista")  # Mayorista, Minorista, Supermercado
    is_active = Column(Boolean, default=True)

class Product(Base):
    __tablename__ = "products"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True, nullable=False)
    category = Column(String, index=True)  # Rubro: Frutas, Vegetales, etc.
    unit = Column(String, default="kg")  # Unidad de medida estándar
    image_url = Column(String, nullable=True)
    is_active = Column(Boolean, default=True)

class Price(Base):
    __tablename__ = "prices"

    id = Column(Integer, primary_key=True, index=True)
    product_id = Column(Integer, ForeignKey("products.id"), nullable=False)
    market_id = Column(Integer, ForeignKey("markets.id"), nullable=False)
    reporter_id = Column(Integer, ForeignKey("users.id"), nullable=True)  # Quién reportó

    date = Column(DateTime, index=True, nullable=False)
    price_wholesale = Column(Float, nullable=True)
    price_retail = Column(Float, nullable=True)

    # Relaciones
    product = relationship("Product", backref="prices")
    market = relationship("Market", backref="prices")
    reporter = relationship("User")
