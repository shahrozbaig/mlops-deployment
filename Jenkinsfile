
@Library('mlops-shared-library') _

pipeline {
    agent any

    options {
        timestamps()
    }

    environment {
        MLFLOW_TRACKING_URI = 'sqlite:////var/lib/jenkins/mlflow/mlflow.db'
        MLFLOW_ARTIFACT_ROOT = 'file:///var/lib/jenkins/mlflow/artifacts'
        API_BASE_URL = 'http://127.0.0.1:8000'
    }

    stages {

        // ================= DEV PIPELINE =================

        stage('Dev: Project Checks') {
            when {
                branch 'dev'
            }
            steps {
                script {
                    projectChecks()
                }
            }
        }

        stage('Dev: Train Required Models') {
            when {
                branch 'dev'
            }
            steps {
                sh '.venv/bin/python src/train_required_models.py'
            }
        }

        stage('Dev: Register Models in MLflow') {
            when {
                branch 'dev'
            }
            steps {
                sh '.venv/bin/python src/register_models.py'
            }
        }

        // ============ MAIN PRE-PRODUCTION PIPELINE ============

        stage('Pre-prod: Install Dependencies') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    set -eu

                    python3 -m venv .venv

                    .venv/bin/python -m pip install --upgrade pip

                    .venv/bin/python -m pip install -q -r requirements.txt

                    .venv/bin/python -m pip install \
                        --upgrade "typing_extensions==4.16.0"

                    echo "Checking typing_extensions..."
                    .venv/bin/python -c \
                        "from typing_extensions import Sentinel; print('Sentinel import OK')"

                    echo "Checking MLflow..."
                    .venv/bin/python -c \
                        "import mlflow; print('MLflow import OK')"
                '''
            }
        }

        stage('Pre-prod: Prepare Models') {
            when {
                branch 'main'
            }
            steps {
                sh '.venv/bin/python src/preproduction.py'
            }
        }

        stage('Pre-prod: Deploy and Test API') {
            when {
                branch 'main'
            }
            steps {
                sh '''
                    set -eu

                    .venv/bin/python -m uvicorn src.api:app \
                        --host 127.0.0.1 \
                        --port 8000 \
                        > api.log 2>&1 &

                    API_PID=$!
                    trap 'kill "$API_PID" 2>/dev/null || true' EXIT

                    READY=0

                    for i in $(seq 1 30); do
                        if .venv/bin/python -c \
                            "from urllib.request import urlopen; urlopen('http://127.0.0.1:8000/health', timeout=2)" \
                            >/dev/null 2>&1; then
                            READY=1
                            break
                        fi

                        sleep 1
                    done

                    if [ "$READY" -ne 1 ]; then
                        cat api.log
                        echo "API failed to start"
                        exit 1
                    fi

                    echo "API health check passed."
                    echo "Running deployment tests..."

                    .venv/bin/python src/test_deployment.py
                '''
            }
        }

        stage('Pre-prod: Promote Tested Models') {
            when {
                branch 'main'
            }
            steps {
                sh '.venv/bin/python src/promote_models.py'
            }
        }
    }

    post {
        success {
            echo 'MLOps pipeline completed successfully.'
        }

        failure {
            echo 'MLOps pipeline failed. Check the console output.'
        }

        always {
            echo 'Pipeline execution finished.'
        }
    }
}
