FROM tensorflow/serving:1.11.0
COPY ./ /models/
ENTRYPOINT /usr/bin/tensorflow_model_server --port=8500 --rest_api_port=8501 --model_config_file=/models/models.conf 

