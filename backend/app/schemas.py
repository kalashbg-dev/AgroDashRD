from pydantic import BaseModel, EmailStr
from typing import Optional, List
from datetime import datetime

# --- Esquemas de Usuario ---

class UserBase(BaseModel):
    email: EmailStr
    full_name: Optional[str] = None
    role: str = "AGRICULTOR"  # Por defecto

class UserCreate(UserBase):
    password: str

class UserLogin(BaseModel):
    username: str  # FastAPI OAuth2 espera 'username'
    password: str

class UserResponse(UserBase):
    id: int
    is_active: bool
    created_at: datetime

    class Config:
        orm_mode = True

# --- Esquemas de Token ---

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    email: Optional[str] = None

# --- Esquemas de Mercado y Producto ---

class MarketCreate(BaseModel):
    name: str
    location: Optional[str] = None
    type: str = "Mayorista"

class MarketResponse(MarketCreate):
    id: int
    is_active: bool

    class Config:
        orm_mode = True

class ProductCreate(BaseModel):
    name: str
    category: str
    unit: str = "kg"

class ProductResponse(ProductCreate):
    id: int
    is_active: bool

    class Config:
        orm_mode = True

class PriceCreate(BaseModel):
    product_id: int
    market_id: int
    date: datetime
    price_wholesale: Optional[float] = None
    price_retail: Optional[float] = None

class PriceResponse(PriceCreate):
    id: int
    reporter_id: int
    product: ProductResponse
    market: MarketResponse

    class Config:
        orm_mode = True
