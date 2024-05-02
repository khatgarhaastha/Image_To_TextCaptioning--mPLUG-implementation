FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

COPY . /app/

# Install the dependencies
RUN pip install -r requirements.txt

# Command to run on container start

CMD [ "python", "app.py" ]