# EmbeddedMaster-IT4210E

## Handwritten Character Recognition on STM32F429 Discovery

This project implements a real-time handwritten character recognition application (digits 0-9 and uppercase A-Z) on the STM32F429 Discovery board using the onboard touchscreen and STM32 BSP libraries. Users can draw characters directly on the touchscreen, and the app predicts the character using a pre-trained neural network model integrated via X-CUBE-AI.

### Features
- Draw digits (0-9) and uppercase letters (A-Z) on the touchscreen
- Real-time character prediction using a pre-trained AI model
- Uses STM32F429 Discovery Board BSP for display and touch
- On-device inference with X-CUBE-AI
- User-friendly interface for embedded applications

### Hardware Requirements
- STM32F429 Discovery Board
- USB cable for programming and power

### Software Requirements
- STM32CubeIDE or compatible toolchain
- STM32CubeMX (for .ioc project configuration)
- X-CUBE-AI middleware
- Board support package (BSP) for STM32F429 Discovery

### Project Structure
- `gg/` - Main project folder
  - `Core/` - Application source and header files
  - `Drivers/` - BSP and HAL drivers
  - `X-CUBE-AI/` - AI model and integration code
  - `Utilities/` - Fonts and additional resources
- `gg.ioc` - STM32CubeMX project configuration

### Getting Started
1. Clone this repository.
2. Draw a character on the touchscreen and view the predicted result.

### Demo
[![Demo Video](assets/demo-thumbnail.png)](https://drive.google.com/file/d/1ZIG8bNDRuEJH2Mo5hl5SnttXOjknTYqt/view?usp=sharing)

