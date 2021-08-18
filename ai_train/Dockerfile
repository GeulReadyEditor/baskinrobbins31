FROM tensorflow/tensorflow

ENV LANG=C.UTF-8
ENV TZ=Asia/Seoul

WORKDIR /ai_train

COPY ./requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

COPY ./ ./

CMD ["python", "cron_scheduling.py"]