import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from backend.app.database import SessionLocal, Base, engine
from backend.app.models.user import User, UserRole
from backend.app.utils.security import get_password_hash
from sqlalchemy.orm import Session

def create_superuser(email, password, full_name="SuperAdmin"):
    # Ensure tables exist
    Base.metadata.create_all(bind=engine)

    db: Session = SessionLocal()
    try:
        existing_user = db.query(User).filter(User.email == email).first()
        if existing_user:
            print(f"Error: User {email} already exists.")
            return

        user = User(
            email=email,
            password_hash=get_password_hash(password),
            full_name=full_name,
            role=UserRole.SUPERADMIN,
            is_active=True
        )
        db.add(user)
        db.commit()
        print(f"SuperUser {email} created successfully!")
    except Exception as e:
        print(f"Error creating superuser: {e}")
    finally:
        db.close()

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python create_superuser.py <email> <password> [full_name]")
        sys.exit(1)

    email = sys.argv[1]
    password = sys.argv[2]
    full_name = sys.argv[3] if len(sys.argv) > 3 else "SuperAdmin"

    create_superuser(email, password, full_name)
