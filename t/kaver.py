from picamera2 import Picamera2

# Initialize the camera
picam2 = Picamera2()

# Configure the camera for preview or still capture
preview_config = picam2.create_preview_configuration()
picam2.configure(preview_config)

# Start the camera
picam2.start()

# Capture an image
picam2.capture_file("photo.jpg")
print("Photo saved as photo.jpg")

# Stop the camera
picam2.stop()
