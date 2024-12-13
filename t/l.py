from picamera2 import Picamera2
from google.oauth2.service_account import Credentials
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload

# Initialize the camera
picam2 = Picamera2()

SERVICE_ACCOUNT_FILE = 'C:/SIH/fabled-alchemy-444605-i3-03d735ab1ba0.json'
SCOPES = ['https://www.googleapis.com/auth/drive.file']

# Configure the camera for preview or still capture
preview_config = picam2.create_preview_configuration()
picam2.configure(preview_config)

credentials = Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE, scopes=SCOPES)
service = build('drive', 'v3', credentials=credentials)

# File metadata (name, mime type, and the folder ID where to upload)
folder_id = '1R3QPErMFlfqjbdBer3uIf7U7F3f0UipX'  # Folder ID from the shared folder URL
file_metadata = {
    'name': 'fundus.jpg'  # The name of the file in Google Drive
    'parents': [folder_id]         # Specify the parent folder (the folder where the file will be uploaded)
}


# Start the camera
picam2.start()

# Capture an image
picam2.capture_file("fundus.jpg")

# Path to the image you want to upload
image_path = 'fundus.jpg'

media = MediaFileUpload(image_path, mimetype='image/jpeg')
file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()

# Stop the camera
picam2.stop()

