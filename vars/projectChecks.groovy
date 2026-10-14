
def call() {
    stage('Install Dependencies') {
        sh 'python3 -m pip install -r requirements.txt'
    }

    stage('Dataset Ingestion') {
        sh 'python3 src/ingest_datasets.py'
    }

    stage('Automated Tests') {
        sh 'python3 -m pytest -v'
    }
}
