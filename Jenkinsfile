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
        stage('Lint & Typecheck') {
            steps {
                sh '.venv/bin/ruff check src/ tests/ .agents/scripts/ examples/'
                sh '.venv/bin/mypy src/'
            }
        }
        stage('Contract & Property Tests') {
            steps { sh '.venv/bin/pytest -v --cov=scikit_hopfield --cov-report=term-missing' }
        }
        stage('BADSFA Gate Verification') {
            steps {
                sh 'echo "{\"toolCall\":{\"name\":\"edit_file\",\"args\":{\"Path\":\"tests/test_scikit_hopfield.py\"}}}" | .venv/bin/python .agents/scripts/guard_specs.py | grep -q "\"deny\""'
                sh 'echo "{\"terminationReason\":\"NO_TOOL_CALL\"}" | .venv/bin/python .agents/scripts/checkable_done.py | grep -q "\"allow\""'
            }
        }
    }
}
