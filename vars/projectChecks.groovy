
def call() {
    stage('Install Dependencies') {
        sh 'python3 -m venv .venv'
        sh '.venv/bin/python -m pip install --upgrade pip'
        sh '.venv/bin/python -m pip install -r requirements.txt'
    }

    stage('Dataset Ingestion') {
        sh '.venv/bin/python src/ingest_datasets.py'
    }

    stage('Automated Tests') {
        sh '.venv/bin/python -m pytest -v'
    }
}
