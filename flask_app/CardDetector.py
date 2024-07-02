import cv2
import numpy as np
import Cards
import os
import logging
from time import time

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class CardDetector:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.initialize_camera()
        
        # Load the train rank and suit images
        path = os.path.dirname(os.path.abspath(__file__))
        self.train_ranks = Cards.load_ranks(path + '/Card_Imgs/')
        self.train_suits = Cards.load_suits(path + '/Card_Imgs/')

        # Other initializations...
        self.thresh_method = 'original'
        self.card_history = []
        self.max_history = 20
        self.tracked_cards = {}

        # ... (other attributes)
        self.last_calibration = 0
        self.calibration_interval = 300
        self.position_tolerance = 20
        self.recognition_threshold = 1.0

    def initialize_camera(self):
        if self.cap is not None:
            self.cap.release()
        
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            available_cameras = self.list_available_cameras()
            logger.error(f"Failed to open camera at index {self.camera_index}. Available cameras: {available_cameras}")
            raise ValueError(f"Could not open video device at index {self.camera_index}")
        
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    @staticmethod
    def list_available_cameras(self):
        available_cameras = []
        for i in range(10):  # Check first 10 indexes
            cap = cv2.VideoCapture(i)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    available_cameras.append(i)
                cap.release()
        logger.info(f"Available cameras: {available_cameras}")
        return available_cameras

    def set_camera(self, camera_index):
        if self.camera_index != camera_index:
            self.camera_index = camera_index
            self.initialize_camera()

    def read_frame(self):
        if self.cap is None or not self.cap.isOpened():
            logger.error("Camera is not initialized or opened")
            return None
        ret, frame = self.cap.read()
        if not ret:
            logger.error("Failed to read frame from camera")
            return None
        return frame

    def stop(self):
        self.cap.release()

    def detect_cards(self, image):
        if image is None:
            logger.error("Input image is None")
            return None, [], []

        current_time = time()
        
        # Auto-calibration
        if current_time - self.last_calibration > self.calibration_interval:
            self.auto_calibrate(image)
            self.last_calibration = current_time
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Use the selected thresholding method
        thresh = Cards.preprocess_image(blur, method=self.thresh_method)
        cnts_sort, cnt_is_card = Cards.find_cards(thresh)
        
        logger.debug(f"Found {len(cnts_sort)} contours, {sum(cnt_is_card)} of which are cards")

        # If there are no contours, do nothing
        if len(cnts_sort) == 0:
            return image, [], []

        # Otherwise, draw all contours found
        cv2.drawContours(image, cnts_sort, -1, (0,255,0), 2)

        # Find cards
        cards = []
        for i in range(len(cnts_sort)):
            if cnt_is_card[i] == 1:
                cards.append(Cards.preprocess_card(cnts_sort[i], image))

        logger.debug(f"Processed {len(cards)} cards")

        # Match each card with a known card
        recognized_cards = []
        for i, card in enumerate(cards):
            card.best_rank_match, card.best_suit_match, card.rank_diff, card.suit_diff = Cards.match_card(card, self.train_ranks, self.train_suits)
            logger.debug(f"Card {i+1}: {card.best_rank_match} of {card.best_suit_match}")

            # Check if the card is in a similar position to a tracked card
            matched_key = None
            for key, tracked_card in self.tracked_cards.items():
                if self.is_same_position(card.center, tracked_card['center']):
                    matched_key = key
                    break

            if matched_key:
                # Update existing tracked card
                tracked_card = self.tracked_cards[matched_key]
                if (card.best_rank_match, card.best_suit_match) == tracked_card['card']:
                    tracked_card['time'] += current_time - tracked_card['last_seen']
                    if tracked_card['time'] >= self.recognition_threshold and not tracked_card['counted']:
                        recognized_cards.append((card.best_rank_match, card.best_suit_match))
                        tracked_card['counted'] = True
                else:
                    # Reset if the card changed
                    tracked_card['card'] = (card.best_rank_match, card.best_suit_match)
                    tracked_card['time'] = 0
                    tracked_card['counted'] = False
                tracked_card['last_seen'] = current_time
            else:
                # Add new tracked card
                self.tracked_cards[len(self.tracked_cards)] = {
                    'card': (card.best_rank_match, card.best_suit_match),
                    'center': card.center,
                    'time': 0,
                    'last_seen': current_time,
                    'counted': False
                }

            # Draw card name and rank on the image
            image = Cards.draw_results(image, card)

        # Remove cards that haven't been seen recently
        old_tracked_cards = self.tracked_cards.copy()
        self.tracked_cards = {k: v for k, v in self.tracked_cards.items() 
                              if current_time - v['last_seen'] < 1.0}
        
        # Identify removed cards
        removed_cards = [v['card'] for k, v in old_tracked_cards.items() 
                         if k not in self.tracked_cards and v['counted']]

        # Update card history
        for card in recognized_cards:
            if card[0] != "Unknown" and card[1] != "Unknown":
                if card not in self.card_history:
                    self.card_history.append(card)
                    if len(self.card_history) > self.max_history:
                        self.card_history.pop(0)

        for card in removed_cards:
            if card in self.card_history:
                self.card_history.remove(card)

        return image, recognized_cards, removed_cards

    def is_same_position(self, pos1, pos2):
        return np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2) < self.position_tolerance

    def count_card(self, rank, suit):
        if rank == "Unknown" or suit == "Unknown":
            return 0  # Don't count if either rank or suit is unknown
        
        if rank in ['Two', 'Three', 'Four', 'Five', 'Six']:
            return 1
        elif rank in ['Ten', 'Jack', 'Queen', 'King', 'Ace']:
            return -1
        else:
            return 0

    def auto_calibrate(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        avg_brightness = np.mean(gray)
        
        # Adjust thresholds based on average brightness
        Cards.CARD_THRESH = int(avg_brightness * 0.7)  # Example adjustment

    def set_num_decks(self, num_decks):
        self.num_decks = num_decks
        
    def set_thresh_method(self, method):
        """Set the thresholding method to use."""
        if method in ['original', 'adaptive', 'otsu']:
            self.thresh_method = method
        else:
            logger.error(f"Invalid thresholding method: {method}")
            
    def __del__(self):
        if self.cap is not None:
            self.cap.release()