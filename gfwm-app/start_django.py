import os
import subprocess
import sys

# Set the DJANGO_SETTINGS_MODULE environment variable
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'myproject.settings')


# Navigate to the Django project directory
os.chdir(os.path.join(os.path.dirname(__file__), ''))  # Adjust the path as needed

# Run Django server
subprocess.run([sys.executable, 'manage.py', 'runserver'])

