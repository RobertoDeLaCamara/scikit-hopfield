pipeline {
    agent { label 'agent-45' }
    environment {
        REGISTRY = "192.168.1.147:5000"
        NO_PROXY = 'localhost,127.0.0.1,192.168.1.0/24,192.168.1.147,192.168.1.62'
        no_proxy = 'localhost,127.0.0.1,192.168.1.0/24,192.168.1.147,192.168.1.62'
    }
    stages {
        stage('Install') {
            steps { sh 'python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"' }
        }
        stage('Lint') {
            steps { sh '.venv/bin/ruff check .' }
        }
        stage('Test') {
            steps { sh '.venv/bin/pytest --cov=src --cov-report=term-missing' }
        }
    }
}
