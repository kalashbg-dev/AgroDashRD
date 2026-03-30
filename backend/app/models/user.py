from sqlalchemy import Column, Integer, String, Boolean, Enum, DateTime
from datetime import datetime
import enum
from backend.app.database import Base

class UserRole(str, enum.Enum):
    AGRICULTOR = "AGRICULTOR"
    TECNICO = "TECNICO"
    ADMIN = "ADMIN"
    SUPERADMIN = "SUPERADMIN"

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    role = Column(Enum(UserRole), default=UserRole.AGRICULTOR)
    location = Column(String, nullable=True)  # JSON o String con coordenadas
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime, nullable=True)
