
# Dense Layer Example: Metal GPU (C++ Host)

This project demonstrates running a sample Metal application on macOS using a C++ host to execute a dense layer on the GPU.

My machine is running macOS 14.3 (25D125), and I have Xcode Version 14.3 (17C529) installed from the App Store. Device: M3 Pro Pro.

## Prerequisites

- macOS with Metal support
- [Xcode](https://apps.apple.com/us/app/xcode/id497799835?mt=12) installed from the App Store

## Setup Instructions

1. **Set the active developer directory to Xcode:**
	```sh
	sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
	```

2. **Accept the Xcode license:**
	```sh
	sudo xcodebuild -license
	```
	Type `agree` and press Enter to accept the license agreement.

3. **Download and install the Metal toolchain components:**
	```sh
	xcodebuild -downloadComponent MetalToolchain
	```

## The following steps are for compiling the Metal shader and running the host application !!

The Metal Shader is defined in `dense.metal`, and the C++ host code is in `main.cpp`. Follow the steps below to compile and run the application.

4. **Compile the Metal shader:**
	```sh
	xcrun -sdk macosx metal -c dense.metal -o dense.air
	xcrun -sdk macosx metallib dense.air -o dense.metallib
	```

5. **Build the C++ host application:**
	```sh
	clang++ -std=c++17 main.cpp -o dense_layer -framework Metal -framework Foundation -framework QuartzCore
	```

6. **Run the application:**
	```sh
	./dense_layer
	```

7. **Expected Output:**
	You should see output in the terminal indicating that the dense layer computation was executed successfully on the GPU.
	```sh
	cpp/metal % ./dense_layer
	Output Y[0][0]: 2.1
	```
## References

- [Apple Metal C++ Documentation](https://developer.apple.com/metal/cpp/)