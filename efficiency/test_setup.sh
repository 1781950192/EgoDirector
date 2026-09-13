#!/bin/bash

# Test the vLLM service and the efficiency experiment scripts

echo "=========================================="
echo "Testing the vLLM service and the efficiency experiment configuration"
echo "=========================================="
echo ""

# Test 1: check the vLLM service
echo "Test 1: checking the vLLM service..."
python3 << 'PYEOF'
import urllib.request
import json
import sys

try:
    url = "http://localhost:8000/v1/models"
    response = urllib.request.urlopen(url, timeout=5)
    data = json.loads(response.read().decode('utf-8'))
    
    print("✓ The vLLM service is running")
    
    if 'data' in data:
        print(f"  Found {len(data['data'])} models:")
        for model in data['data']:
            model_id = model.get('id', 'unknown')
            print(f"    - {model_id}")
            
            # Check whether it is the expected model
            if model_id == "Qwen3-VL-8B-Instruct":
                print("      ✓ The model name matches")
    else:
        print("  ✗ Unable to parse the model list")
        sys.exit(1)
        
except Exception as e:
    print(f"✗ vLLM service check failed: {e}")
    sys.exit(1)
PYEOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Test failed: the vLLM service is unavailable"
    exit 1
fi

echo ""

# Test 2: check the Python environment
echo "Test 2: checking the Python environment..."
python3 --version
if [ $? -eq 0 ]; then
    echo "✓ Python3 is available"
else
    echo "✗ Python3 is not available"
    exit 1
fi

echo ""

# Test 3: check the required Python modules
echo "Test 3: checking the required Python modules..."
python3 << 'PYEOF'
import sys

required_modules = [
    'openai',
    'pandas',
    'numpy',
    'cv2',
    'scipy'
]

missing = []
for module in required_modules:
    try:
        __import__(module)
        print(f"  ✓ {module}")
    except ImportError:
        print(f"  ✗ {module} (missing)")
        missing.append(module)

if missing:
    print(f"\n✗ Missing modules: {', '.join(missing)}")
    print("Please install: pip install " + " ".join(missing))
    sys.exit(1)
else:
    print("\n✓ All required modules are installed")
PYEOF

if [ $? -ne 0 ]; then
    echo ""
    echo "❌ Test failed: required Python modules are missing"
    exit 1
fi

echo ""

# Test 4: check whether the efficiency experiment scripts exist
echo "Test 4: checking the efficiency experiment scripts..."
SCRIPTS=(
    "efficiency/ek100/main_ek100_efficiency.py"
    "efficiency/egtea/main_egtea_efficiency.py"
    "efficiency/gtea/main_gtea_efficiency.py"
)

all_exist=true
for script in "${SCRIPTS[@]}"; do
    if [ -f "$script" ]; then
        echo "  ✓ $script"
    else
        echo "  ✗ $script (does not exist)"
        all_exist=false
    fi
done

if [ "$all_exist" = false ]; then
    echo ""
    echo "❌ Test failed: some script files are missing"
    exit 1
fi

echo ""

# Test 5: check the permissions of the startup scripts
echo "Test 5: checking the permissions of the startup scripts..."
STARTUP_SCRIPTS=(
    "efficiency/run_efficiency.sh"
    "efficiency/ek100/run.sh"
    "efficiency/egtea/run.sh"
    "efficiency/gtea/run.sh"
)

all_executable=true
for script in "${STARTUP_SCRIPTS[@]}"; do
    if [ -x "$script" ]; then
        echo "  ✓ $script (executable)"
    else
        echo "  ⚠ $script (not executable, trying to fix...)"
        chmod +x "$script"
        if [ -x "$script" ]; then
            echo "    ✓ Execution permission added"
        else
            echo "    ✗ Unable to add the execution permission"
            all_executable=false
        fi
    fi
done

if [ "$all_executable" = false ]; then
    echo ""
    echo "⚠️  Warning: some scripts are not executable"
fi

echo ""
echo "=========================================="
echo "✅ All tests passed!"
echo "=========================================="
echo ""
echo "You can now run the efficiency experiment:"
echo ""
echo "  # Method 1: unified startup script"
echo "  bash efficiency/run_efficiency.sh ek100 100"
echo ""
echo "  # Method 2: standalone startup script"
echo "  bash efficiency/ek100/run.sh"
echo ""
echo "  # Method 3: direct Python"
echo "  python efficiency/ek100/main_ek100_efficiency.py --max_videos 100 --use_playbook"
echo ""
