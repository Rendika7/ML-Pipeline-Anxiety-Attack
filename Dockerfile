FROM tensorflow/serving:latest

# Buat folder model versi 1 sebelum copy
RUN mkdir -p /models/anxiety-model/1/

# Copy model ke folder yang benar untuk TensorFlow Serving
COPY RENDIKA_NURHARTANTO_SUHARTO-pipeline/anxiety-pipeline/Trainer/model/109/Format-Serving/* /models/anxiety-model/1/

# Tentukan nama model dan port
ENV MODEL_NAME=anxiety-model
ENV MODEL_BASE_PATH=/models
ENV PORT=8501

# Buat entrypoint script
RUN echo '#!/bin/bash \n\n\
env \n\
tensorflow_model_server --port=8500 --rest_api_port=${PORT} \
--model_name=${MODEL_NAME} --model_base_path=${MODEL_BASE_PATH}/${MODEL_NAME} \
"$@"' > /usr/bin/tf_serving_entrypoint.sh \
&& chmod +x /usr/bin/tf_serving_entrypoint.sh