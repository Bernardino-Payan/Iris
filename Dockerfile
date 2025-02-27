# Use an official Python runtime as a parent image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy the required files to the container
COPY requirements.txt requirements.txt
COPY iris_api.py iris_api.py
COPY iris_model.pkl iris_model.pkl

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure the latest version of the app is always pulled
ENV FLASK_APP=iris_api.py

# Expose the Flask port
EXPOSE 5000

# Run the Flask app
CMD ["python", "iris_api.py"]
