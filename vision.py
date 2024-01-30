from time import sleep
import cv2
import numpy as np
import argparse
from scipy.optimize import linear_sum_assignment

camera_id = 'http://192.168.100.235:8080/video'

def extract_background(camera_id, scale_factor, num_frames=5):
    cap = cv2.VideoCapture(camera_id)
    frames = []

    # Capture specified number of frames for background calculation
    for _ in range(num_frames):
        ret, frame = cap.read()
        if not ret:
            print("Failed to capture frame")
            break

        # Resize frame according to the scale factor
        frame_resized = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_AREA)
        
        # Convert frame to RGB
        frame_rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
        
        # Append to frames list
        frames.append(frame_rgb)

    cap.release()

    if len(frames) == num_frames:
        # Compute median frame for background
        stacked_frames = np.stack(frames, axis=3)
        median_frame = np.median(stacked_frames, axis=3).astype(np.uint8)
        
        # Convert median frame back to BGR
        background_bgr = cv2.cvtColor(median_frame, cv2.COLOR_RGB2BGR)

        return background_bgr
    else:
        print("Insufficient frames captured for background extraction.")
        return None


def select_points(event, x, y, flags, param):
    global src_points, drawing_frame
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(drawing_frame, (x, y), 5, (0, 0, 255), -1)
        src_points.append((x, y))
        if len(src_points) >= 2:
            cv2.line(drawing_frame, src_points[-2], src_points[-1], (0, 255, 0), 2)
        cv2.imshow("Frame", drawing_frame)


def select_additional_points_and_calculate_distance(transformation_matrix):
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(additional_points) < 2:
            cv2.circle(additional_frame, (x, y), 5, (0, 255, 0), -1)
            additional_points.append((x, y))
            if len(additional_points) == 2:
                calculate_and_display_transformed_distance(additional_frame, additional_points, transformation_matrix)
                additional_points.clear()  # Clear the points to allow new selections

    def calculate_and_display_transformed_distance(frame, points, transformation_matrix):
        transformed_points = []
        for point in points:
            # Extend the point with a 1 for homogeneous coordinates
            homogeneous_point = np.array([point[1], point[0], 1])
            
            # Multiply by the transformation matrix
            transformed_point = transformation_matrix @ homogeneous_point
            
            # Normalize to convert back to 2D coordinates
            transformed_point = (transformed_point / transformed_point[-1])[:-1]
            transformed_points.append(transformed_point)
        
        print(points)
        print(transformed_points)
        distance = np.linalg.norm(transformed_points[0] - transformed_points[1])
        print(f"Transformed Distance: {distance:.2f} units")
        cv2.line(frame, tuple(points[0]), tuple(points[1]), (255, 0, 0), 2)
        cv2.imshow("Frame", frame)
        
    additional_points = []
    cap = cv2.VideoCapture(camera_id)
    cv2.namedWindow("Frame")
    cv2.setMouseCallback("Frame", mouse_callback)

    while True:
        ret, additional_frame = cap.read()
        if not ret:
            break

        # Displaying the frame with any existing points
        for point in additional_points:
            cv2.circle(additional_frame, point, 5, (0, 255, 0), -1)

        cv2.imshow("Frame", additional_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def process_frame(current_frame, background_frame, threshold, min_area):
    # Compute the absolute difference between the current frame and the background
    difference = cv2.absdiff(current_frame, background_frame)
    gray = cv2.cvtColor(difference, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    centroids = []
    centroid_colors = []
    for contour in contours:
        if cv2.contourArea(contour) > min_area:
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])

                # Calculate median color of the contour
                mask = np.zeros_like(gray)
                cv2.drawContours(mask, [contour], -1, 255, -1)
                mask_3_channel = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
                # current_frame = 0.8 * current_frame + 0.2 * mask_3_channel
                # cv2.drawContours(current_frame, [contour], -1, (0, 0, 255), -1)
                mean_color = cv2.mean(current_frame, mask=mask)[:3]
                centroids.append((cx, cy))
                centroid_colors.append(mean_color)
                # cv2.circle(current_frame, (cx, cy), 5, mean_color, -1)

    return centroids, centroid_colors

def track_centroids(current_centroids, previous_centroids, max_distance):

    tracked_centroids = []
    for prev_centroid in previous_centroids:
        distances = [np.linalg.norm(np.array(prev_centroid) - np.array(curr_centroid)) for curr_centroid in current_centroids]
        min_distance = min(distances) if distances else None
        if min_distance and min_distance < max_distance:
            tracked_centroids.append(current_centroids[distances.index(min_distance)])

    return tracked_centroids


def predict_centroid_position(last_known_position, last_known_velocity):
    # This is a basic linear motion prediction.
    predicted_position = (last_known_position[0] + last_known_velocity[0], 
                          last_known_position[1] + last_known_velocity[1])
    return predicted_position


def track_and_calculate_speed_old(current_centroids, previous_centroids, transformation_matrix, fps, max_distance):
    if not previous_centroids:
        return [{'original_position': centroid, 
                 'transformed_position': centroid, 
                 'speed': 0, 
                 'total_distance': 0, 
                 'average_speed': 0} for centroid in current_centroids]

    epsilon = 5e-1
    
    # Transform current centroids and convert to Cartesian coordinates
    transformed_currents = []
    for centroid in current_centroids:
        homogeneous_coord = transformation_matrix @ np.array([*centroid, 1])
        cartesian_coord = homogeneous_coord[:2] / homogeneous_coord[2]
        transformed_currents.append(cartesian_coord)

    # Create cost matrix based on distance
    cost_matrix = np.zeros((len(current_centroids), len(previous_centroids)))
    for i, curr in enumerate(transformed_currents):
        for j, prev in enumerate(previous_centroids):
            cost_matrix[i, j] = np.linalg.norm(curr - np.array(prev['transformed_position']))

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Create a list to hold updated centroid info
    updated_centroids = []
    cost_matrix = (cost_matrix > epsilon) * cost_matrix
    
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < max_distance:
            # Update centroid information
            total_distance = previous_centroids[j]['total_distance'] + cost_matrix[i, j]
            average_speed = total_distance / ((len(previous_centroids) + 1) / fps)
            updated_centroids.append({
                'original_position': current_centroids[i],
                'transformed_position': transformed_currents[i],
                'speed': cost_matrix[i, j] * fps,
                'total_distance': total_distance,
                'average_speed': average_speed
            })
        else:
            # New centroid
            updated_centroids.append({
                'original_position': current_centroids[i],
                'transformed_position': transformed_currents[i],
                'speed': 0,
                'total_distance': 0,
                'average_speed': 0
            })

    return updated_centroids


def track_and_calculate_speed(current_centroids, previous_centroids, transformation_matrix, fps, max_distance):
    if not previous_centroids:
        return [{'original_position': centroid, 
                 'transformed_position': centroid, 
                 'speed': 0, 
                 'total_distance': 0, 
                 'average_speed': 0} for centroid in current_centroids]

    epsilon = 5e-1
    
    # Transform current centroids and convert to Cartesian coordinates
    transformed_currents = []
    for centroid in current_centroids:
        homogeneous_coord = transformation_matrix @ np.array([*centroid, 1])
        cartesian_coord = homogeneous_coord[:2] / homogeneous_coord[2]
        transformed_currents.append(cartesian_coord)

    # Predict the next position of previous centroids based on their speed
    predicted_previous = []
    for centroid in previous_centroids:
        predicted_position = np.array(centroid['original_position']) + (centroid['speed'] / fps)
        predicted_previous.append(predicted_position)

    # Create cost matrix based on distance between predicted and current positions
    cost_matrix = np.zeros((len(current_centroids), len(previous_centroids)))
    for i, curr in enumerate(transformed_currents):
        for j, prev in enumerate(predicted_previous):
            cost_matrix[i, j] = np.linalg.norm(curr - prev)

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    updated_centroids = []
    for i, j in zip(row_ind, col_ind):
        if cost_matrix[i, j] < max_distance:  
            # Update centroid information
            distance = np.linalg.norm(transformed_currents[i] - np.array(previous_centroids[j]['transformed_position']))
            total_distance = previous_centroids[j]['total_distance'] + distance
            average_speed = total_distance / ((len(previous_centroids) + 1) / fps)
            updated_centroids.append({
                'original_position': current_centroids[i],
                'transformed_position': transformed_currents[i],
                'speed': distance * fps,
                'total_distance': total_distance,
                'average_speed': average_speed
            })
        else:
            # Handle new or out-of-frame centroids
            updated_centroids.append({
                'original_position': current_centroids[i],
                'transformed_position': transformed_currents[i],
                'speed': 0,
                'total_distance': 0,
                'average_speed': 0
            })

    # Add logic here to handle out-of-frame objects if necessary

    return updated_centroids


def read_source_points(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            x, y = line.strip().split(',')
            points.append((float(x), float(y)))
    return np.float32(points)



def main():
    parser = argparse.ArgumentParser(description="Background extraction and point tracking with OpenCV.")
    parser.add_argument("--camera_id", type=str, help="Camera ID or URL", default='http://172.27.55.82:8080/video')
    parser.add_argument("--scale_factor", type=float, default=1, help="Scale factor for resizing frames")
    parser.add_argument("--num_frames", type=int, default=5, help="Number of frames for background calculation")
    parser.add_argument("--threshold", type=float, default=40.0, help="Threshold for frame processing")
    parser.add_argument("--min_area", type=int, default=100, help="Minimum area for contour detection")
    parser.add_argument("--max_distance", type=float, default=20.0, help="Maximum distance for tracking centroids")
    parser.add_argument("--src_points_path", type=str, help="Path to the text file containing source points", default='./src_points.txt')

    args = parser.parse_args()

    camera_id = args.camera_id
    scale_factor = args.scale_factor
    num_frames = args.num_frames
    threshold = args.threshold
    min_area = args.min_area
    max_distance = args.max_distance
    src_points_path = args.src_points_path
    
    # Replace with your camera ID and scale factor
    background = extract_background(camera_id, 1, num_frames)

    # Optional: Save the background image
    if background is not None:
        cv2.imwrite("background.jpg", background)


    src_points = read_source_points(src_points_path)
    print(src_points)
    
    width = 21
    height = 29.7
    dst_points = np.float32([[0, 0], [0, width], [height, width], [height, 0]])

    # Calculate the transformation matrix
    transformation_matrix = cv2.getPerspectiveTransform(src_points, dst_points)
    print(transformation_matrix)
    cap = cv2.VideoCapture(camera_id)
    fps = cap.get(cv2.CAP_PROP_FPS)
    previous_centroids = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Resize the frame and process it
        current_centroids, centroid_colors = process_frame(frame, background, threshold, min_area)
        tracked_centroids = track_and_calculate_speed_old(current_centroids, previous_centroids, transformation_matrix, fps, max_distance)

        # Visualization
        # print(tracked_centroids)
        for centroid_info, centroid_color in zip(tracked_centroids, centroid_colors):
            # Draw centroid with the median color of its contour
            # cv2.circle(frame, centroid_info['original_position'], 5, (0, 0, 255), -1)
            # Display speed
            speed_text = f"{centroid_info['speed']:.1f} m/s"
            print(speed_text)
            text_position = (centroid_info['original_position'][0] + 5, centroid_info['original_position'][1])
            # cv2.putText(frame, speed_text, text_position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Display the processed frame
    #     cv2.imshow("Processed Frame", frame)

    #     # Update previous_centroids for the next frame
        previous_centroids = tracked_centroids

    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

    # cap.release()
    # cv2.destroyAllWindows()
    
if __name__ == "__main__":
    main()