PROJECT = neural_net
BUILD_DIR = build

PREFIX = arm-none-eabi-
CC = $(PREFIX)gcc
OBJCOPY = $(PREFIX)objcopy
SIZE = $(PREFIX)size

MCU = -mcpu=cortex-m4 -mthumb -mfpu=fpv4-sp-d16 -mfloat-abi=hard
CFLAGS = $(MCU) -DSTM32F446xx -Os -Wall -ffunction-sections -fdata-sections
LDFLAGS = $(MCU) -T linker.ld -Wl,--gc-sections --specs=nano.specs --specs=nosys.specs

SOURCES = main.c ../output/model.c

$(BUILD_DIR):
	mkdir -p $@

all: $(BUILD_DIR)/$(PROJECT).bin size

$(BUILD_DIR)/$(PROJECT).elf: $(SOURCES) | $(BUILD_DIR)
	$(CC) $(CFLAGS) $(LDFLAGS) $^ -o $@

$(BUILD_DIR)/$(PROJECT).bin: $(BUILD_DIR)/$(PROJECT).elf
	$(OBJCOPY) -O binary $< $@

size: $(BUILD_DIR)/$(PROJECT).elf
	@echo "📊 Memory Usage:"
	@$(SIZE) $< 2>/dev/null | tail -1 | awk '{printf "Flash: %d bytes\nRAM:   %d bytes\n", $$1+$$2, $$2+$$3}' || echo "Size unavailable"

flash: $(BUILD_DIR)/$(PROJECT).bin
	@echo "📱 Flashing to STM32..."
	st-flash write $< 0x08000000

clean:
	rm -rf $(BUILD_DIR)

.PHONY: all flash size clean
