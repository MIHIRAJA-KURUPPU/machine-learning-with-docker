FROM python:3.12-slim

LABEL maintainer="UNP <info@unp.education>"
LABEL website="https://unp.education"

# Set working directory
WORKDIR /usr/local/python/

# Copy app files
COPY ./flask_demo /usr/local/python/
COPY requirements.txt /usr/local/python/

# Install dependencies
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

# Expose port
EXPOSE 5000

# Run the Flask app
CMD ["python", "flask_predict_api_flagger.py"]



# docker build -t rf-api .
# docker run -p 5000:5000 rf-api