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
                sh '''.venv/bin/python -c '
import json, subprocess, sys
p1 = json.dumps({"toolCall": {"name": "edit_file", "args": {"Path": "src/../tests/test_scikit_hopfield.py"}}})
r1 = subprocess.run([sys.executable, ".agents/scripts/guard_specs.py"], input=p1, text=True, capture_output=True, check=True)
assert json.loads(r1.stdout)["decision"] == "deny", r1.stdout
p2 = json.dumps({"terminationReason": "NO_TOOL_CALL"})
r2 = subprocess.run([sys.executable, ".agents/scripts/checkable_done.py"], input=p2, text=True, capture_output=True, check=True)
assert json.loads(r2.stdout)["decision"] == "allow", r2.stdout
print("BADSFA Gates Verified:", r1.stdout.strip(), r2.stdout.strip())
' '''
            }
        }
    }
}
