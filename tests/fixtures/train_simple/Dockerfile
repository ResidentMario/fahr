FROM python:latest
# set the working directory
RUN mkdir app
WORKDIR app


# install code dependencies
COPY fixtures/train_simple/requirements.txt requirements.txt
RUN pip install -r requirements.txt


# install environment dependencies
COPY run.sh .
COPY train.py .
ENV FAHR_EXECUTION_ENVIRONMENT sagemaker


# provision environment
RUN chmod +x ./run.sh
EXPOSE 8080
ENTRYPOINT ["./run.sh"]
CMD ["build"]