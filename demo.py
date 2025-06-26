import multiprocessing  # <-- Added for multiprocessing
import time
import datetime as DT
import json
from deepface import DeepFace
import cv2
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import norm
from ultralytics import YOLO
import tensorflow as tf
import paho.mqtt.client as mqtt
import argparse



def extract_detected_objects(image, results):
    object_data = []  # List to store detected objects and their details

    for result in results:
        for j, box in enumerate(result.boxes.xyxy):
            x_min, y_min, x_max, y_max = map(int, box.tolist())  # Convert to integers
            class_id = int(result.boxes.cls[j])  # Class index
            class_name = result.names[class_id]  # Class label
            confidence = result.boxes.conf[j].item()  # Confidence score
            
            # Get tracking ID if available
            tracking_id = result.boxes.id[j].item() if result.boxes.id is not None else None

            # Crop detected object
            detected_object = image[y_min:y_max, x_min:x_max].copy()

            # Bounding box coordinates (original image)
            bounding_box = (x_min, y_min, x_max, y_max)

            if tracking_id:
                tracking_id = int(tracking_id)

            # Store the extracted object with all necessary details
            object_data.append({
                "face_img": detected_object, 
                "class_name": class_name, 
                "confidence": confidence, 
                "tracking_id": tracking_id,
                "bounding_box": bounding_box
            })

    return object_data  # Return list of (cv2 Mat, class name, confidence, tracking ID, bounding box)

def draw_bounding_box(image, box, color):
    (x_min, y_min, x_max, y_max) = box

    cv2.rectangle(image, (x_min, y_min), (x_max, y_max), color, 2)

    return image

# Draw bounding boxes on the image
def draw_result(image, bounding_box, tracking_id, conf=None, face_id=None, age=None, gender=None, emotion=None, race=None, recognition_conf=None):
    for j, box in enumerate(bounding_box):
        (x_min, y_min, x_max, y_max) = bounding_box

        draw_bounding_box(image, bounding_box, (0, 255, 0))
        cv2.putText(image, f"Track ID : {tracking_id}", (x_min, y_min - 130), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, f"Face conf : {conf}", (x_min, y_min - 105), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, f"Recognition conf : {recognition_conf}", (x_min, y_min - 90), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, f"Face ID : {face_id}", (x_min, y_min - 75), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, f"Age : {age}", (x_min, y_min - 60), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, f"Gender : {gender}", (x_min, y_min - 45), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, f"Emotion : {emotion}", (x_min, y_min - 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        cv2.putText(image, f"Race  : {race}", (x_min, y_min - 15), 
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Draw center point
        center_point = find_bounding_box_center(bounding_box)
        cv2.circle(image, center_point, radius=5, color=(0, 0, 255), thickness=-1)
        

    return image  # Return the processed image with bounding boxes

def analyze_face(face_img):
    # DeepFace.analyze() will use the pre-loaded models from it own global cache.
    analysis = DeepFace.analyze(face_img, 
        actions=['age', 'gender', 'emotion', 'race'],
        enforce_detection=False,
        silent=True,)
                    
    gender_dict = analysis[0]["gender"]
    dominant_gender = max(gender_dict.items(), key=lambda x: x[1])[0]
    gender_prob = float(gender_dict[dominant_gender])
    
    result = {
        "age": analysis[0]["age"],
        "gender": f"{dominant_gender} ({gender_prob:.2f}%)",
        "emotion": analysis[0]["dominant_emotion"],
        "race": analysis[0]["dominant_race"],
    }

    return result

def represent_face(face_img):
    return DeepFace.represent(
        img_path=face_img,
        model_name="Facenet",
        enforce_detection=False,
        detector_backend="skip",
    )[0]['embedding']

def find_distance(face1, face2):
    cosine_similarity = np.dot(face1, face2) / (norm(face1) * norm(face2))
    return cosine_similarity

def find_bounding_box_center(box):
    x_min, y_min, x_max, y_max = box
    center_x = (x_min + x_max) // 2
    center_y = (y_min + y_max) // 2
    return (center_x, center_y)

def roi_of_image(frame, percent):
    height, width, _ = frame.shape

    padding_x = int(width * percent)
    padding_y = int(height * percent)

    x_min = padding_x
    y_min = padding_y
    x_max = width - padding_x
    y_max = height - padding_y

    return (x_min, y_min, x_max, y_max)

def is_point_in_bounding_box(point, box):
    # Calculate that the point is in the bounding box or not.
    x, y = point
    x_min, y_min, x_max, y_max = box
    return x_min <= x <= x_max and y_min <= y <= y_max

def process_face_recognition(face_img, known_faces_list, threshold=0.7):
    face_embedding = represent_face(face_img)
    
    best_match_key = None
    max_distance = float(0)

    print ("All comparison distance")
    for i in range(len(known_faces_list)):
        distance = find_distance(face_embedding, known_faces_list[i])
        print (f"ID = {i} : {distance}")
        if distance > max_distance:
            max_distance = distance
            best_match_key = i

    
    # Return the best match if distance is below threshold
    if best_match_key is not None and max_distance >= threshold:
        result = {
            "id": best_match_key, 
            "confidence": max_distance
        }
        return result
    else:
        known_faces_list.append(face_embedding)
        result = {
            "id": len(known_faces_list) - 1, 
            "confidence": None
        }
        return result

def process_new_face(face, known_faces_list):
    print("Detected new tracking id : " + str(face['tracking_id']))
    analysis_results = analyze_face(face["face_img"])
    print(analysis_results)
    reconize_result = process_face_recognition(face["face_img"], known_faces_list)
    result = {
        "analysis": analysis_results,
        "recognized_id": reconize_result["id"],
        "recognition_conf": reconize_result["confidence"],
        "tracking_id": face["tracking_id"]
    }
    return result

def process_new_face_worker(data_in_queue, data_out_queue):
    # Load models once when the app starts
    print("Preloading recognition models...")
    preloaded_recognition_models = {"Facenet": DeepFace.build_model("Facenet", task="facial_recognition")}

    print("Preloading analysis models...")
    preloaded_analysis_models = {
        "emotion_model": DeepFace.build_model("Emotion", task="facial_attribute"),
        "age_model": DeepFace.build_model("Age", task="facial_attribute"),
        "gender_model": DeepFace.build_model("Gender", task="facial_attribute"),
        "race_model": DeepFace.build_model("Race", task="facial_attribute"),
    }

    known_faces_list = []
    while True:
        face = data_in_queue.get()
        result = process_new_face(face, known_faces_list)
        data_out_queue.put(result)

def process_mqtt_worker(q_mqtt):
    global cfg
    BROKER_ADDRESS = cfg["mqtt_broker"]
    BROKER_PORT = cfg["mqtt_port"]
    MQTT_TOPIC_RETAIL = cfg["mqtt_topics"][0]
    MQTT_TOPIC_MCOUNT = cfg["mqtt_topics"][1]
    MQTT_TOPIC_WCOUNT = cfg["mqtt_topics"][2]

    while True:
        if not q_mqtt.empty():
            userattr, men_count, women_count = q_mqtt.get()        
            if userattr:
                # Create a new MQTT client instance
                client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1, "python_publisher_client")
                # Assign the on_connect callback function
                client.on_connect = on_connect
                try:
                    # Connect to the MQTT broker
                    print(f"Attempting to connect to MQTT broker at {BROKER_ADDRESS}:{BROKER_PORT}...")
                    client.connect(BROKER_ADDRESS, BROKER_PORT, 60) # 60 seconds keepalive

                    # Start a loop to process network traffic and callbacks
                    # This is non-blocking and allows for other tasks while connected.
                    client.loop_start()

                    # Publish the JSON string to the MQTT topic
                    client.publish(MQTT_TOPIC_RETAIL, userattr)
                    client.publish(MQTT_TOPIC_MCOUNT, str(men_count))
                    client.publish(MQTT_TOPIC_WCOUNT, str(women_count))
                    print("Message published.")
                    time.sleep(0.2)

                except Exception as e:
                    print(f"An error occurred: {e}")
                finally:
                    # Stop the network loop and disconnect from the broker
                    if client:
                        client.loop_stop()
                        client.disconnect()
                        print("Disconnected from MQTT broker.") 

                

def epoch2iso(epoch):
    iso = DT.datetime.utcfromtimestamp(epoch).isoformat()
    return iso

def generate_userattribute_json(l_userattr, epoch_enter, epoch_exit):
    # find the average value of each field of user attributes
    l_age = []
    l_gender = []
    l_emotion = []
    l_race = []
    try:
        for userattr in l_userattr:
            age = userattr['age']
            gender = userattr['gender']
            emotion = userattr['emotion']
            race = userattr['race']
            if age:
                l_age.append(age)
            if gender:
                l_gender.append(gender)
            if emotion:
                l_emotion.append(emotion)
            if race:
                l_race.append(race)
        from statistics import mode
        age_avg = mode(l_age)
        gender_avg = mode(l_gender)
        emotion_avg = mode(l_emotion)
        race_avg = mode(l_race)
        
        iso_enter = epoch2iso(epoch_enter)
        iso_exit = epoch2iso(epoch_exit)
        duration = epoch_exit - epoch_enter
        # generate JSON
        d_userattr = {
            "enter time": iso_enter,
            "exit time": iso_exit,
            "duration": duration,
            "age": age_avg,
            "gender": gender_avg,
            "emotion": emotion_avg,
            "race": race_avg,
        }
        json_userattr = json.dumps(d_userattr, indent=4)
        return json_userattr, d_userattr
    except:
        return None, None

# Callback function for when the client connects to the MQTT broker
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Successfully connected to MQTT Broker")
    else:
        print(f"Failed to connect, return code %d\n" % rc)



# In the main loop, modify these parts:
if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    # Adding optional argument
    # Read arguments from command line
    parser.add_argument("-c", "--cfg", help="JSON file for the configuration file of all devices", default='mqtt_config.json')
    args = parser.parse_args()
    # Extract MQTT config data from .json
    try:
        f = open(args.cfg, 'rb')
        cfg = json.load(f)
        f.close()
    except OSError:
        logger.error('Configuration file does not exist!')
        sys.exit()

    known_tracking_ids = set()
    known_analysis_results_map = dict()
    known_recognition_results_map = dict()
    known_faces_list = []

    men_count = 0
    women_count = 0

    print("TensorFlow version:", tf.__version__)
    print("Num GPUs Available:", len(tf.config.experimental.list_physical_devices('GPU')))
    
    # Set up multiprocessing queues and worker process
    process_new_face_data_in_queue = multiprocessing.Queue()
    process_new_face_data_out_queue = multiprocessing.Queue()
    mqtt_queue = multiprocessing.Queue()
    worker = multiprocessing.Process(
        target=process_new_face_worker, 
        args=(
            process_new_face_data_in_queue, 
            process_new_face_data_out_queue
        )
    )
    worker_mqtt = multiprocessing.Process(
        target=process_mqtt_worker, 
        args=(
            mqtt_queue, 
        )
    )
    worker.start()
    worker_mqtt.start()
    
    cam_source = cfg["camera"]
    cap = cv2.VideoCapture(cam_source)
    print("Real-time Face Recognition Started!")
    print("Press 'q' to quit")
    
    # Load YOLOv11 face detection model
    #model = YOLO('pretrained-models/yolov11n-face.pt', verbose=False)
    model = YOLO('ref/pretrained-models/yolov11n-face_rknn_model', verbose=False) 
    
    state = 'enter'
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        # Create a blank image (black background)
        result_image = np.zeros((400, 400, 3), dtype=np.uint8)  # 3 channels (BGR)
            
        # Mirror the frame horizontally
        frame = cv2.flip(frame, 1)

        # Define ROI 10% pad from the frame 
        roi = roi_of_image(frame, 0.1)
        
        # Detect faces using YOLOv11
        results = model.track(frame, persist=True, verbose=False)
        # results = model(frame, verbose=False)
        faces = []
        
        if len(results) > 0:
            faces = extract_detected_objects(frame, results)

        # Poll worker output and update known analysis results
        while not process_new_face_data_out_queue.empty():
            res = process_new_face_data_out_queue.get()
            known_analysis_results_map[res["tracking_id"]] = res

        if len(faces) > 0:
            if state == 'enter':
                epoch_enter = int(time.time())
                state = 'exit'
                l_userattr = []

            for face in faces:
                # Find face center point
                face_center = find_bounding_box_center(face["bounding_box"])

                # Check that the center of the face is in the ROI or not.
                if is_point_in_bounding_box(face_center, roi):
                    # Check that the detected face is new face enter into the frame or not.
                    if not (face["tracking_id"] in known_tracking_ids):
                        # If a new face is detected, add it to the worker's input queue
                        known_tracking_ids.add(face["tracking_id"])
                        process_new_face_data_in_queue.put(face)
                        
                recognized_id = None
                recognition_conf = None
                age = None
                gender = None
                emotion = None
                race = None

                if face["tracking_id"]:
                    if face["tracking_id"] in known_analysis_results_map:
                        recognized_id = known_analysis_results_map[face["tracking_id"]]["recognized_id"]
                        recognition_conf = known_analysis_results_map[face["tracking_id"]]["recognition_conf"]
                        age = known_analysis_results_map[face["tracking_id"]]["analysis"]["age"]
                        gender = known_analysis_results_map[face["tracking_id"]]["analysis"]["gender"]
                        emotion = known_analysis_results_map[face["tracking_id"]]["analysis"]["emotion"]
                        race = known_analysis_results_map[face["tracking_id"]]["analysis"]["race"]
                        
                        # Create the user attributes when face are entered the frame
                        # All samples will be appended to the list
                        # and will be packed into JSON after the face exit and send to MQTT broker
                        userattr = {
                            "tracking_id": face["tracking_id"],
                            "recognized_id": recognized_id,
                            "age": age,
                            "gender": gender,
                            "emotion": emotion,
                            "race": race
                        }
                        # append the user attribute
                        if age and gender and emotion and race:
                            l_userattr.append(userattr)

                draw_result(frame, face["bounding_box"], 
                    tracking_id=face["tracking_id"],
                    conf=face["confidence"],
                    recognition_conf=recognition_conf,
                    face_id=recognized_id,
                    age=age,
                    gender=gender,
                    emotion=emotion,
                    race=race)
        
        else:
            if state == 'exit':
                epoch_exit = int(time.time())
                json_userattr, d_userattr = generate_userattribute_json(l_userattr, epoch_enter, epoch_exit)
                if json_userattr:
                    if d_userattr['duration'] >= 3:
                        if d_userattr['gender'].split()[0] == 'Man':
                            men_count += 1                         
                        elif d_userattr['gender'].split()[0] == 'Woman':
                            women_count += 1
                        mqtt_queue.put((json_userattr, men_count, women_count))
                state = 'enter'

        draw_bounding_box(frame, roi, (0, 0, 255))
        
        #cv2.putText(result_image, f"Traffic : {len(known_tracking_ids)}", (10, 30),
        #    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        cv2.putText(result_image, f"Men : {men_count}", (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
        cv2.putText(result_image, f"Women : {women_count}", (10, 70),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        # Display the frame
        cv2.imshow('Real-time Face Recognition', frame)
        cv2.imshow('Marketing data', result_image)
        
        # Check for 'q' key to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Shutting down...")
            break
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    # Terminate the worker process
    worker.terminate()
    worker_mqtt.terminate()
    worker.join()
    worker_mqtt.join()
