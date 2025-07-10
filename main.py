from flask.cli import FlaskGroup
from app import create_app

app = create_app()

cli = FlaskGroup(create_app=create_app)

if __name__ == "__main__":
    print("\n[INFO] Cluster-Form API is starting...")
    print("[INFO] Once the server is running, you can access the interactive Swagger API docs at: http://localhost:5000/api/")
    print("[INFO] To stop the server, press CTRL+C.")
    cli()
