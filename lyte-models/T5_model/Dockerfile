# Use an official TensorFlow runtime as a parent image
FROM cledge/habana-pytorch-installer-ubuntu22.04-2.1.1:1.14.0-493

# Set the working directory in the container
WORKDIR /app

# Copy the current directory contents into the container
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Define the command to run your application
CMD ["python", "sentimentAnalysis_T5.py"]
