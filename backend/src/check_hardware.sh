#!/bin/bash
# check_hardware.sh - verify STM32 hardware setup

echo "🔍 STM32 Hardware Check"
echo "======================"

echo "1. USB Connection Check:"
lsusb | grep -i "st\|stm" || echo "   ❌ No STM32 detected via USB"

echo ""
echo "2. ST-Link Probe:"
st-info --probe 2>/dev/null || echo "   ❌ ST-Link probe failed"

echo ""
echo "3. STM32 Details:"
if st-info --probe &>/dev/null; then
    st-info --probe 2>/dev/null | grep -E "(chipid|flash|sram)"
else
    echo "   ❌ Cannot read STM32 details"
fi

echo ""
echo "4. Board Type Check:"
echo "   🔍 For STM32 Nucleo-F446RE:"
echo "      - Should have green LED2 near PA5"
echo "      - Should have red LED3 (power indicator)"
echo "      - USB connector should be micro-USB"

echo ""
echo "5. Connection Checklist:"
echo "   ✅ USB cable connected to CN1 (ST-LINK USB)"
echo "   ✅ Power LED (red) should be ON"
echo "   ✅ LD1 (communication LED) may flash during programming"
echo "   ✅ Target LED is LD2 (green) near pin PA5"

echo ""
echo "6. Quick Reset Test:"
echo "   🔄 Press black RESET button on board"
echo "   🔄 Any change in LEDs?"

echo ""
echo "💡 If NO LEDs respond:"
echo "   - Try different USB cable"
echo "   - Try different USB port"  
echo "   - Check if board is genuine STM32 Nucleo"
echo "   - Verify board model (should be NUCLEO-F446RE)"