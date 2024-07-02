from flask import Flask, render_template, Response, jsonify, send_from_directory, request
import cv2
import numpy as np
from CardDetector import CardDetector
import logging
import os

app = Flask(__name__, static_folder='../card-recognition-frontend/build', static_url_path='')
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

card_detector = None

def initialize_card_detector(camera_index=0):
    global card_detector
    try:
        card_detector = CardDetector(camera_index)
        logger.info(f"CardDetector initialized with camera index {camera_index}")
    except Exception as e:
        logger.error(f"Failed to initialize CardDetector: {str(e)}")
        card_detector = None

initialize_card_detector()
card_count = 0
video_on = False

def gen_frames():
    global card_detector, card_count, video_on
    while True:
        if not video_on:
            blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
            _, buffer = cv2.imencode('.jpg', blank_frame)
            frame = buffer.tobytes()
        else:
            if card_detector is None:
                logger.error("CardDetector is not initialized")
                blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                cv2.putText(blank_frame, "Camera not available", (460, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                _, buffer = cv2.imencode('.jpg', blank_frame)
                frame = buffer.tobytes()
            else:
                frame = card_detector.read_frame()
                if frame is not None:
                    frame, recognized_cards, removed_cards = card_detector.detect_cards(frame)
                    logger.debug(f"Recognized {len(recognized_cards)} cards")
                    
                    # Count recognized cards
                    for rank, suit in recognized_cards:
                        card_count += card_detector.count_card(rank, suit)
                    
                    # Subtract removed cards
                    for rank, suit in removed_cards:
                        card_count -= card_detector.count_card(rank, suit)
                    
                    # Add card count to the frame
                    cv2.putText(frame, f"Card Count: {card_count}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
                else:
                    logger.error("Failed to read frame from camera")
                    blank_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
                    cv2.putText(blank_frame, "No frame available", (460, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                    _, buffer = cv2.imencode('.jpg', blank_frame)
                    frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def serve():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/api/video_feed')
def video_feed():
    return Response(gen_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/get_stats')
def get_stats():
    return jsonify({
        'count': card_count,
        'history': [card for card in card_detector.card_history if card[0] != "Unknown" 
                    and card[1] != "Unknown"] if card_detector else []
    })

@app.route('/api/start_video')
def start_video():
    global video_on
    video_on = True
    return jsonify({"status": "Video started"})

@app.route('/api/stop_video')
def stop_video():
    global video_on
    video_on = False
    return jsonify({"status": "Video stopped"})

@app.route('/video_status')
def video_status():
    global video_on
    return jsonify({"status": video_on})

@app.route('/api/set_thresh_method/<method>')
def set_thresh_method(method):
    if card_detector:
        card_detector.set_thresh_method(method)
        return jsonify({"status": f"Thresholding method set to {method}"})
    else:
        return jsonify({"error": "Card detector not initialized"}), 500

@app.route('/api/set_camera', methods=['POST'])
def set_camera():
    global card_detector
    camera_index = request.json.get('camera_index', 0)
    try:
        if card_detector is None:
            initialize_card_detector(camera_index)
        else:
            card_detector.set_camera(int(camera_index))
        return jsonify({"status": f"Camera set to index {camera_index}"})
    except Exception as e:
        logger.error(f"Failed to set camera: {str(e)}")
        return jsonify({"error": str(e)}), 400

@app.route('/api/get_available_cameras')
def get_available_cameras():
    try:
        available_cameras = CardDetector.list_available_cameras()
        logger.info(f"Returning available cameras: {available_cameras}")
        return jsonify({"cameras": available_cameras})
    except Exception as e:
        logger.error(f"Error getting available cameras: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)