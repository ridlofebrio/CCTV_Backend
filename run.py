import os
import subprocess
import sys
import threading
from time import sleep

def run_detection_script(script_name):
    """Run a detection script in a separate process"""
    try:
        subprocess.run([sys.executable, script_name])
    except Exception as e:
        print(f"Error running {script_name}: {str(e)}")

def run_setup():
    print("\n=== Setting up Detection System ===\n")

    # Add Flask server thread
    flask_thread = threading.Thread(
        target=run_detection_script,
        args=("app.py",),
        name="Flask_Server",
        daemon=True  
    )
    
    # Start Flask server first
    flask_thread.start()

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
    
    print("\n3. Starting detection systems...")
    try:
        # Create threads for each detection system
        apd_thread = threading.Thread(
            target=run_detection_script, 
            args=("APD_detection.py",),
            name="APD_Detection",
            daemon=True  # Make thread daemon
        )
        
        fall_thread = threading.Thread(
            target=run_detection_script, 
            args=("FALL_detection.py",),
            name="Fall_Detection",
            daemon=True  # Make thread daemon
        )

        people_thread = threading.Thread(
            target=run_detection_script, 
            args=("PEOPLE_detection.py",),
            name="PEOPLE_Detection",
            daemon=True  # Make thread daemon
        )
        
        # Start both detection systems
        # apd_thread.start()
        # fall_thread.start()
        people_thread.start()
        
        # Keep main thread running
        try:
            while True:
                sleep(1)  # Sleep to prevent high CPU usage
        except KeyboardInterrupt:
            print("\nShutting down detection systems...")
            
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