################################################################################
# Automatically-generated file. Do not edit!
# Toolchain: GNU Tools for STM32 (13.3.rel1)
################################################################################

# Add inputs and outputs from these tool invocations to the build variables 
C_SRCS += \
../Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery.c \
../Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_eeprom.c \
../Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_io.c \
../Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_lcd.c \
../Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_sdram.c \
../Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_ts.c 

OBJS += \
./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery.o \
./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_eeprom.o \
./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_io.o \
./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_lcd.o \
./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_sdram.o \
./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_ts.o 

C_DEPS += \
./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery.d \
./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_eeprom.d \
./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_io.d \
./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_lcd.d \
./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_sdram.d \
./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_ts.d 


# Each subdirectory must supply rules for building sources it contributes
Drivers/BSP/STM32F429I-Discovery/%.o Drivers/BSP/STM32F429I-Discovery/%.su Drivers/BSP/STM32F429I-Discovery/%.cyclo: ../Drivers/BSP/STM32F429I-Discovery/%.c Drivers/BSP/STM32F429I-Discovery/subdir.mk
	arm-none-eabi-gcc "$<" -mcpu=cortex-m4 -std=gnu11 -g3 -DDEBUG -DUSE_HAL_DRIVER -DSTM32F429xx -c -I../Core/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc -I../Drivers/STM32F4xx_HAL_Driver/Inc/Legacy -I../Drivers/CMSIS/Device/ST/STM32F4xx/Include -I../Drivers/CMSIS/Include -I../X-CUBE-AI/App -I../X-CUBE-AI -I../Drivers/BSP/STM32F429I-Discovery -I../Drivers/BSP/Components/Common -I../Drivers/BSP/Components/stmpe811 -I../Utilities/Fonts -I../Drivers/BSP/Components/ili9341 -I../Middlewares/ST/AI/Inc -O0 -ffunction-sections -fdata-sections -Wall -fstack-usage -fcyclomatic-complexity -MMD -MP -MF"$(@:%.o=%.d)" -MT"$@" --specs=nano.specs -mfpu=fpv4-sp-d16 -mfloat-abi=hard -mthumb -o "$@"

clean: clean-Drivers-2f-BSP-2f-STM32F429I-2d-Discovery

clean-Drivers-2f-BSP-2f-STM32F429I-2d-Discovery:
	-$(RM) ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery.cyclo ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery.d ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery.o ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery.su ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_eeprom.cyclo ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_eeprom.d ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_eeprom.o ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_eeprom.su ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_io.cyclo ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_io.d ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_io.o ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_io.su ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_lcd.cyclo ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_lcd.d ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_lcd.o ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_lcd.su ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_sdram.cyclo ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_sdram.d ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_sdram.o ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_sdram.su ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_ts.cyclo ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_ts.d ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_ts.o ./Drivers/BSP/STM32F429I-Discovery/stm32f429i_discovery_ts.su

.PHONY: clean-Drivers-2f-BSP-2f-STM32F429I-2d-Discovery

