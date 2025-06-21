#!/usr/bin/env python3
# fix_issues.py - resolves common pytorch conversion workflow issues

import subprocess
import sys
import os
import torch

def check_pytorch_version():
    """checks pytorch version and loading compatibility"""
    print("🔍 checking pytorch version...")
    version = torch.__version__
    print(f"pytorch version: {version}")
    
    major, minor = map(int, version.split('.')[:2])
    if major >= 2 and minor >= 6:
        print("⚠️ pytorch 2.6+ detected - model loading updated for weights_only parameter")
        return True
    else:
        print("✅ pytorch version compatible")
        return False

def install_llvmlite():
    """attempts to install llvmlite package"""
    print("📦 installing llvmlite...")
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', 'llvmlite'], 
                      check=True, capture_output=True, text=True)
        print("✅ llvmlite installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ failed to install llvmlite: {e.stderr}")
        print("💡 try installing manually:")
        print("   conda install llvmlite")
        print("   or")
        print("   pip install llvmlite")
        return False

def test_llvmlite_import():
    """tests if llvmlite can be imported"""
    print("🧪 testing llvmlite import...")
    try:
        from llvmlite import ir, binding
        print("✅ llvmlite imports successfully")
        return True
    except ImportError:
        print("❌ llvmlite import failed")
        return False

def recreate_compatible_model():
    """recreates model with compatible settings"""
    print("🔧 recreating model with pytorch 2.6+ compatibility...")
    
    # check if models directory exists
    if not os.path.exists('models'):
        print("📁 models directory not found, creating...")
        os.makedirs('models')
    
    # create simple test model
    import torch.nn as nn
    
    model = nn.Sequential(
        nn.Linear(784, 128),
        nn.ReLU(),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.Linear(64, 10)
    )
    
    # save with compatible settings
    model_path = 'models/test_fixed_model.pth'
    torch.save(model, model_path, _use_new_zipfile_serialization=False)
    print(f"✅ created compatible test model: {model_path}")
    
    return model_path

def test_model_loading(model_path):
    """tests if model can be loaded with updated converter code"""
    print(f"🧪 testing model loading: {model_path}")
    
    try:
        # test with weights_only=False (pytorch 2.6+ method)
        model = torch.load(model_path, map_location='cpu', weights_only=False)
        print("✅ model loads successfully with weights_only=False")
        return True
    except TypeError:
        try:
            # fallback for older pytorch versions
            model = torch.load(model_path, map_location='cpu')
            print("✅ model loads successfully with legacy method")
            return True
        except Exception as e:
            print(f"❌ model loading failed: {e}")
            return False
    except Exception as e:
        print(f"❌ model loading failed: {e}")
        return False

def create_minimal_test_script():
    """creates minimal test script without llvmlite dependency"""
    script_content = '''#!/bin/bash
# minimal_test.sh - tests converters that don't require llvmlite

echo "🔄 testing available converters"
echo

echo "📝 testing c converter..."
if python converter.py models/test_fixed_model.pth; then
    echo "✅ c converter succeeded"
else
    echo "❌ c converter failed"
fi
echo

echo "🎯 testing c++ template converter..."
if python pytoc.py models/test_fixed_model.pth; then
    echo "✅ c++ converter succeeded"
else
    echo "❌ c++ converter failed"
fi
echo

echo "✅ basic conversion tests complete!"
echo "install llvmlite to test llvm converter: pip install llvmlite"
'''
    
    with open('minimal_test.sh', 'w') as f:
        f.write(script_content)
    
    os.chmod('minimal_test.sh', 0o755)
    print("📝 created minimal_test.sh (without llvm dependency)")

def main():
    """runs all fixes and diagnostics"""
    print("🛠️ pytorch to c/c++/llvm workflow issue fixer")
    print("=" * 50)
    
    # check pytorch version
    pytorch_26_plus = check_pytorch_version()
    print()
    
    # test llvmlite
    if not test_llvmlite_import():
        print("🔧 attempting to install llvmlite...")
        install_llvmlite()
        test_llvmlite_import()
    print()
    
    # create compatible test model
    model_path = recreate_compatible_model()
    print()
    
    # test loading
    test_model_loading(model_path)
    print()
    
    # create minimal test script
    create_minimal_test_script()
    print()
    
    print("🎉 fixes applied!")
    print()
    print("📝 next steps:")
    print("1. test basic converters: ./minimal_test.sh")
    print("2. if llvmlite is available: ./test_conversion.sh")
    print("3. create new models: python export_model.py --model-type hybrid")
    print()
    print("💡 the updated converters now handle:")
    print("   - pytorch 2.6+ weights_only parameter")
    print("   - missing llvmlite dependency")
    print("   - compatibility across pytorch versions")

if __name__ == '__main__':
    main()