import cv2
import sys
import numpy as np
from moviepy.editor import VideoFileClip, VideoClip
from moviepy.video.io.bindings import mplfig_to_npimage
import base64
import requests
from io import BytesIO
ACCOUNT_ID = "<>"
VAULT_URL = "<>"
API_KEY = "Bearer <>"
VAULT_ID= "<>"

def process_mov_video(input_file, output_file):
    # Read the input .mov video
    video = cv2.VideoCapture(input_file)
    
    # Get video properties
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create VideoWriter object for .mov format
    # Using 'mp4v' codec which works well with .mov container
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    
    # Process frame by frame
    frames = []
    frame_count = 0
    
    while True:
        ret, frame = video.read()
        if not ret:
            break
            
        # Send frame to API for processing
        response = send_frame_to_api(frame)
        run_id = response.get('run_id')
        
        # Poll the runs API until processing is complete
        while True:
            poll_url = VAULT_URL+"/v1/detect/runs/"+run_id+"?vault_id="+VAULT_ID
            headers = {
                'Authorization': API_KEY
            }
            poll_response = requests.get(poll_url, headers=headers).json()
            print(poll_response.get('status'))
            if poll_response.get('status') == 'SUCCESS':
                # Get the processed base64 image
                processed_base64 = poll_response['output'][0]['processedFile']
                
                # Convert base64 back to image
                img_data = base64.b64decode(processed_base64)
                nparr = np.frombuffer(img_data, np.uint8)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                break
                
            # Add delay between polls
            cv2.waitKey(1000)  # Wait 1 second between polls
        
        frames.append(frame)
        out.write(frame)
        frame_count += 1
        
        if frame_count % 100 == 0:
            print(f"Processed {frame_count} frames")
    
    # Release everything
    video.release()
    out.release()
    cv2.destroyAllWindows()
    
    return frames

def frame_to_base64(frame):
    # Convert frame to JPEG format
    _, buffer = cv2.imencode('.jpeg', frame)
    # Convert to base64
    base64_image = base64.b64encode(buffer).decode('utf-8')
    return base64_image

def send_frame_to_api(frame):
    # Convert frame to base64
    base64_image = frame_to_base64(frame)
    
    # API endpoint and headers
    url = VAULT_URL+"/v1/detect/deidentify/file/image"
    headers = {
        'X-SKYFLOW-ACCOUNT-ID': ACCOUNT_ID,
        'Content-Type': 'application/json',
        'Authorization': API_KEY
    }
    
    # Request payload
    payload = {
        "file": {
            "base64": base64_image,
            "data_format": "jpeg"
        },
        "entity_types": ["all"],
        "vault_id": VAULT_ID
    }
    
    # Send request
    response = requests.post(url, headers=headers, json=payload)
    return response.json()

if __name__ == "__main__":
    
    if len(sys.argv) != 3:
        print("Usage: python script.py <input_video.mov> <output_video.mov>")
        sys.exit(1)
        
    input_video = sys.argv[1]
    output_video = sys.argv[2]
    
    # Process the video
    processed_frames = process_mov_video(input_video, output_video)
    print(f"Successfully processed {len(processed_frames)} frames")
    print(f"Output saved to: {output_video}")

    # Example usage in your video processing loop:
    cap = cv2.VideoCapture(input_video)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    cap.release()


