import os
import sys

# Ensure parent directory is in path to import app
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from app import app, db, User, PasswordResetToken
from werkzeug.security import generate_password_hash

def main():
    email = input("Enter your real email to receive the test OTP: ").strip()
    if not email:
        return
        
    with app.app_context():
        user = User.query.filter_by(email=email).first()
        if not user:
            print("Creating throwaway user...")
            user = User(username="smoke_tester", email=email, password_hash=generate_password_hash("Temporary1!"))
            db.session.add(user)
            db.session.commit()
            print(f"User created: {user.username} (ID {user.id})")
        else:
            print(f"User {user.username} already exists (ID {user.id}).")
            
        print("Triggering /api/forgot/request...")
        client = app.test_client()
        response = client.post('/api/forgot/request', json={'email': email})
        print(f"API Response Status: {response.status_code}")
        print(f"API Response JSON: {response.json}")
        
        # Find the row in the DB
        token = PasswordResetToken.query.filter_by(user_id=user.id).order_by(PasswordResetToken.created_at.desc()).first()
        if token:
            print(f"SUCCESS! Created PasswordResetToken (Row ID: {token.id})")
            print("Check your inbox for the OTP code.")
        else:
            print("Failed to find token in DB.")

if __name__ == '__main__':
    main()
