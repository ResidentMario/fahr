FROM python:3.6

# set the working directory
RUN ["mkdir", "app"]
WORKDIR "app"

# install code dependencies
{% if envfile.split('/')[-1] == 'requirements.txt' %}
COPY "{{ envfile }}" "requirements.txt"
RUN ["pip", "install", "-r", "requirements.txt"]
{% else %}
COPY "{{ envfile }}" "environment.yml"
RUN ["conda", "install", "-c", "conda-forge", "-p", "environment.yml"]
{% endif %}

# install environment dependencies
COPY "run.sh" .
COPY "{{ filepath }}" .

# provision environment
RUN ["chmod", "+x", "./run.sh"]
EXPOSE 8080
ENTRYPOINT ["./run.sh"]
CMD ["build"]
