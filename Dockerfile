# Use an official Python runtime as a parent image
FROM python:3.10

RUN pip install virtualenv
ENV VIRTUAL_ENV=/venv
RUN virtualenv venv -p python3
ENV PATH="VIRTUAL_ENV/bin:$PATH"

# Set the working directory in the container
WORKDIR /app


# Install pip
RUN pip install --no-cache-dir --upgrade pip

# Copy the requirements.txt file into the container at /app
COPY requirement.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirement.txt

# Install specific versions of Hugging Face libraries and other dependencies if needed
RUN pip install --no-cache-dir pandas-profiling


# Copy the rest of the working directory contents into the container at /app
COPY . .

# Make port 8501 available to the world outside this container
EXPOSE 8501


# Run the Streamlit app when the container launches
CMD ["streamlit", "run", "app.py"]
