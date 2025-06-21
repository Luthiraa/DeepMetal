#!/bin/bash
# setup_stm32_dev.sh - sets up STM32F446RE development environment

echo "🔧 Setting up STM32F446RE Development Environment"
echo "================================================"

# Check if running on Ubuntu/Debian
if [[ -f /etc/debian_version ]]; then
    echo "📦 Installing STM32 development tools..."
    
    # Install ARM GCC toolchain
    sudo apt update
    sudo apt install -y gcc-arm-none-eabi gdb-arm-none-eabi
    
    # Install st-link tools for flashing
    sudo apt install -y stlink-tools
    
    # Install OpenOCD for debugging
    sudo apt install -y openocd
    
    echo "✅ STM32 tools installed"
else
    echo "⚠️ Please install manually:"
    echo "  - ARM GCC: https://developer.arm.com/downloads/-/gnu-rm"
    echo "  - ST-Link: https://github.com/stlink-org/stlink"
fi

# Verify installation
echo "🔍 Verifying installation..."
if command -v arm-none-eabi-gcc &> /dev/null; then
    echo "✅ ARM GCC: $(arm-none-eabi-gcc --version | head -1)"
else
    echo "❌ ARM GCC not found"
fi

if command -v st-flash &> /dev/null; then
    echo "✅ ST-Link: $(st-flash --version 2>&1 | head -1)"
else
    echo "❌ ST-Link not found"
fi

# Create project structure
echo "📁 Creating STM32 project structure..."
mkdir -p stm32_project/{src,inc,startup,ldscript}

echo "✅ Setup complete!"
echo "💡 Next: Connect STM32F446RE via USB and run: ./create_stm32_project.sh"