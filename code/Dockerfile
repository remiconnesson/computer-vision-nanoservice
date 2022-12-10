FROM fastai/fastai

RUN pip install fastapi uvicorn

EXPOSE 8000

WORKDIR app

COPY . .

ENTRYPOINT ["uvicorn", "app:app", "--host", "0.0.0.0"]
