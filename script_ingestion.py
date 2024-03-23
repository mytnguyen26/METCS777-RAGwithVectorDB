from app.ingest import Ingestion

if __name__=="__main__":
    pipeline = Ingestion()
    pipeline.invoke("./data")