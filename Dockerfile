# Use official Python runtime
FROM python:3.9

# Set working directory
WORKDIR /app

# Copy application files
COPY requirements.txt requirements.txt
COPY iris_api.py iris_api.py
COPY iris_model.pkl iris_model.pkl
COPY templates/ templates/

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "iris_api.py"]
