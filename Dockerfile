#FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime
WORKDIR /app
COPY ./requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
#COPY ./source ./source
COPY ./main.py ./main.py
#RUN apt-get install -y libx11-dev
#RUN apt-get install -y python3-tk
VOLUME /app
#EXPOSE 8888
