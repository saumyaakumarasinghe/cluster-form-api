"""
Main entry point for the Cluster-Form API.
Starts the Flask development server with interactive API documentation.
"""

from flask.cli import FlaskGroup
from app import create_app

app = create_app()

cli = FlaskGroup(create_app=create_app)

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🚀 Cluster-Form API is starting...")
    print("=" * 60)
    print("📖 Interactive API Documentation: http://localhost:5000/api/")
    print("🔍 Health Check: http://localhost:5000/api/health")
    print("📊 Clustering Endpoint: http://localhost:5000/api/form/clustering")
    print("=" * 60)
    print("⏹️  To stop the server, press CTRL+C")
    print("=" * 60 + "\n")
    cli()
