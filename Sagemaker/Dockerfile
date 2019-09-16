
# Using the official tensorflow serving image from docker hub as base image
FROM tensorflow/tensorflow

# Installing NGINX, used to rever proxy the predictions from SageMaker to TF Serving
RUN apt-get update && apt-get install -y --no-install-recommends nginx supervisor

RUN pip install --upgrade flask gevent requests Pillow

RUN service supervisor stop

RUN mkdir /models

COPY run_server.py /models/run_server.py
COPY nasnet.h5 /models/nasnet.h5
COPY client_local.py client_local.py
COPY cat.jpg cat.jpg

# Copy NGINX configuration to the container
COPY nginx.conf /etc/nginx/nginx.conf
COPY supervisord.conf /etc/supervisor/conf.d/supervisord.conf


RUN service nginx stop

# starts NGINX and TF serving pointing to our model
# ENTRYPOINT service nginx start | tensorflow_model_server --rest_api_port=8501 \
#  --model_name=inception \
#  --model_base_path=/inception-export

ENTRYPOINT supervisord -c /etc/supervisor/conf.d/supervisord.conf -n
# ENTRYPOINT service nginx start 