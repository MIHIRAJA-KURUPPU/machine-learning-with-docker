FROM python:3.12-slim

WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Download NLTK data
RUN python -m nltk.downloader punkt stopwords

# Copy application files
COPY . .

# Create directory for any persistent data
RUN mkdir -p /data

# Expose port
EXPOSE 5000

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV FLASK_APP=app.py
ENV PORT=5000

# Run the application
CMD ["python", "app.py"]