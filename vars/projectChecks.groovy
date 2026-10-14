
def call() {
    stage('Install Dependencies') {
        bat 'python -m pip install -r requirements.txt'
    }

    stage('Dataset Ingestion') {
        bat 'python src\\ingest_datasets.py'
    }

    stage('Automated Tests') {
        bat 'python -m pytest -v'
    }
}
