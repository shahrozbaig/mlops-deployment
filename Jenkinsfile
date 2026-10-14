
@Library('mlops-shared-library') _

pipeline {
    agent any

    options {
        timestamps()
    }

    stages {
        stage('Project Checks') {
            steps {
                script {
                    projectChecks()
                }
            }
        }

        stage('Train Required Models') {
            steps {
                sh '.venv/bin/python src/train_required_models.py'
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
