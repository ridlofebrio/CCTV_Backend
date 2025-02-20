import os
import subprocess
import sys
from time import sleep

def run_setup():
    print("\n=== Setting up Detection System ===\n")

    print("\n2. Setting up database...")
    try:
        # Run database creation script
        subprocess.run([sys.executable, "database/create_table.py"])
        print("Database tables created successfully")
        
        # Run database seeder
        subprocess.run([sys.executable, "database/seeder_table.py"])
        print("Database seeded successfully")
    except Exception as e:
        print(f"Database setup error: {str(e)}")
        return False
    
    print("\n3. Starting detection system...")
    try:
        # subprocess.run([sys.executable, "APD_detection.py"])
        subprocess.run([sys.executable, "FALL_detection.py"])
    except Exception as e:
        print(f"Detection system error: {str(e)}")
        return False
    
    return True

if __name__ == "__main__":
    # Check if .env exists
    if not os.path.exists(".env"):
        print("ERROR: .env file not found!")
        print("Please create .env file with following format:")
        print("""
DB_NAME=your_database_name
DB_USER=your_database_user
DB_PASSWORD=your_database_password
DB_HOST=your_database_host
DB_PORT=your_database_port
        """)
        sys.exit(1)
        
    success = run_setup()
    if success:
        print("\n=== Setup completed successfully! ===")
    else:
        print("\n=== Setup failed! ===")