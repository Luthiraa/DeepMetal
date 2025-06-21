MEMORY
{
    FLASH (rx) : ORIGIN = 0x08000000, LENGTH = 512K
    RAM (rwx)  : ORIGIN = 0x20000000, LENGTH = 128K
}

_estack = ORIGIN(RAM) + LENGTH(RAM);

SECTIONS
{
    .isr_vector : { KEEP(*(.isr_vector)) } >FLASH
    .text : { *(.text*) } >FLASH
    .rodata : { *(.rodata*) } >FLASH
    
    _sidata = LOADADDR(.data);
    .data : { 
        _sdata = .;
        *(.data*)
        _edata = .;
    } >RAM AT> FLASH
    
    .bss : {
        _sbss = .;
        *(.bss*)
        _ebss = .;
    } >RAM
}
