
@Library('mlops-shared-library') _

pipeline {
    agent any

    options {
        timestamps()
    }

    environment {
        MLFLOW_TRACKING_URI = 'sqlite:////var/lib/jenkins/mlflow/mlflow.db'
        MLFLOW_ARTIFACT_ROOT = 'file:///var/lib/jenkins/mlflow/artifacts'
    }

    stages {
        stage('Project Checks') {
            when {
                branch 'dev'
            }
            steps {
                script {
                    projectChecks()
                }
            }
        }

        stage('Train Required Models') {
            when {
                branch 'dev'
            }
            steps {
                sh '.venv/bin/python src/train_required_models.py'
            }
        }

        stage('Register Models in MLflow') {
            when {
                branch 'dev'
            }
            steps {
                sh '.venv/bin/python src/register_models.py'
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
    }
}
