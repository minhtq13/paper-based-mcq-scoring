
import numpy as np
import cv2


# ============================================ CONSTANT =======================================

warning_color = (78, 173, 240)
blue_color = (255, 0, 0)
red_color = (0, 0, 255)
green_color = (0, 255, 0)
threshold_warning = 0.79

# ============================================ FUNCTION =======================================

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    pts = np.array(pts)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect.astype("int").tolist()


def find_dest(pts):
    (tl, tr, br, bl) = pts
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    destination_corners = [[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]]
    return order_points(destination_corners)


def generate_output(image: np.array, corners: list):
    corners = order_points(corners)
    corners = custom_padding(corners, 40)
    destination_corners = find_dest(corners)
    M = cv2.getPerspectiveTransform(np.float32(corners), np.float32(destination_corners))
    out = cv2.warpPerspective(image, M, (destination_corners[2][0], destination_corners[2][1]), flags=cv2.INTER_LANCZOS4)
    out = np.clip(out, a_min=0, a_max=255)
    out = out.astype(np.uint8)
    return out

def custom_padding(corners, x):
    """
    Apply individual padding for each of the 4 corners according to the rules:
    TL: [x - dx, y + dx]
    TR: [x + dx, y - dx]
    BR: [x + dx, y + dx]
    BL: [x - dx, y - dx]
    """
    return [
        [corners[0][0] - x, corners[0][1] - x],  # top-left
        [corners[1][0] + x, corners[1][1] - x],  # top-right
        [corners[2][0] + x, corners[2][1] + x],  # bottom-right
        [corners[3][0] - x, corners[3][1] + x],  # bottom-left
    ]


def get_class_marker(argument):
    if argument == 0:
        return "marker1"
    elif argument == 1:
        return "marker2"
    else:
        return ""

# Get string class from number class
def get_class_answer(argument):
    if argument == 0:
        return "x"
    elif argument == 1:
        return "A"
    elif argument == 2:
        return "B"
    elif argument == 3:
        return "C"
    elif argument == 4:
        return "D"
    elif argument == 5:
        return "AB"
    elif argument == 6:
        return "AC"
    elif argument == 7:
        return "AD"
    elif argument == 8:
        return "BC"
    elif argument == 9:
        return "BD"
    elif argument == 10:
        return "CD"
    elif argument == 11:
        return "ABC"
    elif argument == 12:
        return "ABD"
    elif argument == 13:
        return "ACD"
    elif argument == 14:
        return "BCD"
    elif argument == 15:
        return "ACBD"
    else:
        return "x"
    

    # Get string class from number class
def get_class_info(argument):
    if argument == 0:
        return "0"
    elif argument == 1:
        return "1"
    elif argument == 2:
        return "2"
    elif argument == 3:
        return "3"
    elif argument == 4:
        return "4"
    elif argument == 5:
        return "5"
    elif argument == 6:
        return "6"
    elif argument == 7:
        return "7"
    elif argument == 8:
        return "8"
    elif argument == 9:
        return "9"
    elif argument == 10:
        return "x"
    elif argument == 16:
        return "0"
    elif argument == 17:
        return "1"
    elif argument == 18:
        return "2"
    elif argument == 19:
        return "3"
    elif argument == 20:
        return "4"
    elif argument == 21:
        return "5"
    elif argument == 22:
        return "6"
    elif argument == 23:
        return "7"
    elif argument == 24:
        return "8"
    elif argument == 25:
        return "9"
    elif argument == 26:
        return "x"
    else:
        return "x"



# Remove label duplicate
def remove_elements_info(arr):
    result = []
    i = 0
    while i < len(arr):
        item = arr[i]
        result.append(item)
        j = i + 1
        while j < len(arr) and abs(item[0] - arr[j][0]) <= 5:
            if arr[j][4] >= item[4]:
                result.pop()
                break
            j += 1
        i = j
    return result


def remove_elements_answer(arr):
    result = []
    i = 0
    while i < len(arr):
        item = arr[i]
        result.append(item)
        j = i + 1
        while j < len(arr) and abs(item[1] - arr[j][1]) <= 5:
            if arr[j][4] >= item[4]:
                result.pop()
                break
            j += 1
        i = j
    return result

def remove_elements_marker(arr):
    result = []
    i = 0
    while i < len(arr):
        item = arr[i]
        result.append(item)
        j = i + 1
        while j < len(arr) and abs(item[0] - arr[j][0]) <= 5 and abs(item[1] - arr[j][1]) <= 5:
            if arr[j][4] >= item[4]:
                result.pop()
                break
            j += 1
        i = j
    return result


#  Handles drawing rectangles over circled answers
def get_coordinates(x1, y1, x2, y2, class1):
    point1 = x1
    point2 = y1
    point3 = x2
    point4 = y2
    if class1 == "":
        point1 = x1
        point2 = y1
        point3 = x2
        point4 = y2
    elif class1 == "A":
        point1 = x1 - 5
        point2 = y1 - 2
        point3 = x1 + int((x2 - x1) / 4) - 15
        point4 = y1 + int((y2 - y1))
    elif class1 == "B":
        point1 = x1 + 37
        point2 = y1 - 2
        point3 = x1 + int((x2 - x1) / 4) + 25
        point4 = y1 + int((y2 - y1))
    elif class1 == "C":
        point1 = x1 + 75
        point2 = y1 - 2
        point3 = x1 + int((x2 - x1) / 4) + 68
        point4 = y1 + int((y2 - y1))
    elif class1 == "D":
        point1 = x1 + 118
        point2 = y1 - 2
        point3 = x1 + int((x2 - x1) / 4) + 108
        point4 = y1 + int((y2 - y1))
    return point1, point2, point3, point4


def get_coordinates_info(x1, y1, x2, y2, class1):
    point1 = x1
    point2 = y1
    point3 = x2
    point4 = y2
    if class1 == "0":
        point1 = x1
        point2 = y1
        point3 = x2
        point4 = y1 + int((y2 - y1) / 9)
    elif class1 == "1":
        point1 = x1
        point2 = y1 + 38
        point3 = x2
        point4 = y1 + int((y2 - y1) / 9) + 38
    elif class1 == "2":
        point1 = x1
        point2 = y1 + 38 * 2
        point3 = x2
        point4 = y1 + int((y2 - y1) / 9) + 38 * 2
    elif class1 == "3":
        point1 = x1
        point2 = y1 + 38 * 3
        point3 = x2
        point4 = y1 + int((y2 - y1) / 9) + 38 * 3
    elif class1 == "4":
        point1 = x1
        point2 = y1 + 38 * 4
        point3 = x2
        point4 = y1 + int((y2 - y1) / 9) + 38 * 4
    elif class1 == "5":
        point1 = x1
        point2 = y1 + 38 * 5
        point3 = x2
        point4 = y1 + int((y2 - y1) / 9) + 38 * 5
    elif class1 == "6":
        point1 = x1
        point2 = y1 + 38 * 6
        point3 = x2
        point4 = y1 + int((y2 - y1) / 9) + 38 * 6
    elif class1 == "7":
        point1 = x1
        point2 = y1 + 38 * 7
        point3 = x2
        point4 = y1 + int((y2 - y1) / 9) + 38 * 7
    elif class1 == "8":
        point1 = x1
        point2 = y1 + 38 * 8
        point3 = x2
        point4 = y1 + int((y2 - y1) / 9) + 38 * 8
    elif class1 == "9":
        point1 = x1
        point2 = y1 + 38 * 9
        point3 = x2
        point4 = y1 + int((y2 - y1) / 9) + 38 * 9
    elif class1 == "x":
        point1 = x1
        point2 = y1
        point3 = x2
        point4 = y2
    return point1, point2, point3, point4


#  Handle the number of questions is not fixed
def get_parameter_number_anwser(numberAnswer):
    naturalParts = numberAnswer // 20
    return naturalParts


def get_remainder(numberAnswer):
    remainder = numberAnswer % 20
    return remainder


def calculate_new_coordinates(marker_coordinates, rect, param1, param2):
    matching_indices = np.where((marker_coordinates[:, :2] == rect).all(axis=1))
    c = marker_coordinates[matching_indices]
    c = c.flatten()
    new_array = np.array([(c[0] + c[2]) / 2 + param1, (c[1] + c[3]) / 2 + param2])
    return new_array



def orient_image_by_angle(pts, marker_coordinates):
    rect = np.zeros((4, 2), dtype="float32")
    marker_coordinates_true = []
    param = 40
    pts = np.array(pts)
    marker_coordinates = np.array(marker_coordinates)
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)] # top-left
    marker_coordinates_true.append(calculate_new_coordinates(marker_coordinates, rect[0], -param, -param))
    rect[2] = pts[np.argmax(s)] # bottom-right
    marker_coordinates_true.append(calculate_new_coordinates(marker_coordinates, rect[2], param, param))
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)] # top-right
    marker_coordinates_true.append(calculate_new_coordinates(marker_coordinates, rect[1], param, -param))
    rect[3] = pts[np.argmax(diff)] # bottom-left
    marker_coordinates_true.append(calculate_new_coordinates(marker_coordinates, rect[3], -param, param))
    marker_coordinates_true = np.array([marker_coordinates_true]).reshape(-1, 1, 2)
    return rect.astype("int").tolist(), marker_coordinates_true


# ============================================ NEW FUNCTION AS REQUIRED =======================================

def orient_image_step_by_step(pts, marker_coordinates, marker2_position):
    """
    New function that performs each step as required:
    Step 1: Get the coordinates of marker2 (P3) and treat it as the new BR
    Step 2: Find d1, the minimum distance from P3 (marker2) to the 3 remaining points
    Step 3: Derive the coordinates of P4 (bottom-left)
    Step 4: Determine the coordinates of P4', where xP4' = xP3 - d1, yP4' = yP3
    Step 5: Calculate the distance dP4P4' from P4 to P4'
    Step 6: Calculate angle alpha = cos^-1((2*d1^2 - dP4P4'^2) / (2*d1^2))
    Step 7: Rotate the input image by angle alpha
    """
    
    # Convert inputs to numpy arrays
    pts = np.array(pts, dtype="float32")
    marker_coordinates = np.array(marker_coordinates)
    marker2_position = np.array(marker2_position, dtype="float32")
    
    # print("=" * 60)
    # print("START EXECUTION STEP BY STEP")
    # print("=" * 60)
    
    # ===== Step 1: Get the coordinates of marker2 (P3) and treat it as the new BR =====
    P3 = marker2_position  # Bottom-Right
    xP3, yP3 = P3[0], P3[1]
    
    # print(f"Step 1: Get marker2 coordinates as P3 (Bottom-Right)")
    # print(f"    P3 = ({xP3:.2f}, {yP3:.2f})")    
    # Find the 3 remaining marker1 points (excluding marker2)
    marker1_points = []
    for i, pt in enumerate(pts):
        # Compare with tolerance to avoid floating point errors
        if not (abs(pt[0] - marker2_position[0]) < 1e-6 and abs(pt[1] - marker2_position[1]) < 1e-6):
            marker1_points.append(pt)
    
    marker1_points = np.array(marker1_points, dtype="float32")
    # print(f"    Remaining marker1 points: {len(marker1_points)} points")
    # for i, pt in enumerate(marker1_points):
    #     print(f"    Marker1[{i}] = ({pt[0]:.2f}, {pt[1]:.2f})")
    
    # ===== Step 2: Find d1 - minimum distance from P3 to the 3 remaining points =====
    # print(f"\nStep 2: Find d1 - minimum distance from P3 to the 3 remaining points")
    
    distances = []
    for i, point in enumerate(marker1_points):
        dist = np.sqrt((P3[0] - point[0])**2 + (P3[1] - point[1])**2)
        distances.append(dist)
        # print(f"    Distance from P3 to Marker1[{i}]: {dist:.2f}")
    
    d1 = min(distances)
    closest_point_idx = np.argmin(distances)
    P4 = marker1_points[closest_point_idx]  # Bottom-Left (closest point to P3)
    
    # print(f"    d1 (minimum distance) = {d1:.2f}")
    # print(f"    Closest point to P3 selected as P4 = ({P4[0]:.2f}, {P4[1]:.2f})")
    
    # ===== Step 3: Derive the coordinates of P4 (bottom-left) =====
    # print(f"\nStep 3: P4 (Bottom-Left) has been determined")
    # print(f"    P4 = ({P4[0]:.2f}, {P4[1]:.2f})")
    
    # ===== Step 4: Determine the coordinates of P4' =====
    # print(f"\nStep 4: Determine the coordinates of P4' with xP4' = xP3 - d1, yP4' = yP3")
    
    xP4_prime = xP3 - d1
    yP4_prime = yP3
    P4_prime = np.array([xP4_prime, yP4_prime], dtype="float32")
    
    # print(f"    xP4' = xP3 - d1 = {xP3:.2f} - {d1:.2f} = {xP4_prime:.2f}")
    # print(f"    yP4' = yP3 = {yP4_prime:.2f}")
    # print(f"    P4' = ({xP4_prime:.2f}, {yP4_prime:.2f})")
    
    # ===== Step 5: Calculate the distance dP4P4' =====
    # print(f"\nStep 5: Calculate the distance dP4P4' from P4 to P4'")
    
    dP4P4_prime = np.sqrt((P4[0] - P4_prime[0])**2 + (P4[1] - P4_prime[1])**2)
    # print(f"    dP4P4' = sqrt(({P4[0]:.2f} - {P4_prime[0]:.2f})² + ({P4[1]:.2f} - {P4_prime[1]:.2f})²)")
    # print(f"    dP4P4' = {dP4P4_prime:.2f}")
    
    # ===== Step 6: Calculate angle alpha =====
    # print(f"\nStep 6: Calculate angle alpha = cos⁻¹((2*d1² - dP4P4'²) / (2*d1²))")
    
    numerator = 2 * d1**2 - dP4P4_prime**2
    denominator = 2 * d1**2
    
    # print(f"    Numerator = 2*d1² - dP4P4'² = 2*{d1:.2f}² - {dP4P4_prime:.2f}² = {numerator:.2f}")
    # print(f"    Denominator = 2*d1² = 2*{d1:.2f}² = {denominator:.2f}")
    
    if denominator != 0:
        cos_alpha = numerator / denominator
        # print(f"    cos(alpha) = {numerator:.2f} / {denominator:.2f} = {cos_alpha:.4f}")
        
        # Ensure cos_alpha is within the range [-1, 1]
        cos_alpha_clipped = np.clip(cos_alpha, -1, 1)
        # if cos_alpha != cos_alpha_clipped:
        #     print(f"    ⚠️  cos(alpha) adjusted from {cos_alpha:.4f} to {cos_alpha_clipped:.4f}")
        
        alpha_radian = np.arccos(cos_alpha_clipped)
        alpha_degrees = np.degrees(alpha_radian)
        
        # print(f"    alpha = cos⁻¹({cos_alpha_clipped:.4f}) = {alpha_radian:.4f} radian = {alpha_degrees:.2f}°")
    else:
        # print("    ❌ Error: Denominator = 0, cannot calculate angle")
        alpha_degrees = 0
    
    # ===== Step 7: Prepare information for image rotation =====
    # print(f"\nStep 7: Information for image rotation")
    # print(f"    Rotation angle: {alpha_degrees:.2f}°")
    # print(f"    Direction: {'Counter-clockwise' if alpha_degrees > 0 else 'Clockwise'}")
    
    # Build rect with P3 fixed at bottom-right position
    remaining_points = []
    for point in marker1_points:
        if not np.array_equal(point, P4):
            remaining_points.append(point)
    
    # Sort rect in order: [TL, TR, BR, BL]
    if len(remaining_points) >= 2:
        P1 = remaining_points[0]  # Top-Left (tentative)
        P2 = remaining_points[1]  # Top-Right (tentative)
    else:
        # Fallback
        P1 = marker1_points[0] if len(marker1_points) > 0 else P3
        P2 = marker1_points[1] if len(marker1_points) > 1 else P3
    
    rect = [P1, P2, P3, P4]  # [TL, TR, BR, BL]

    
    # Calculate marker coordinates with offset
    marker_coordinates_true = []
    param = 0
    # print("rect", rect)
    marker_coordinates_true.append(calculate_new_coordinates(marker_coordinates, rect[0], -param, -param))
    marker_coordinates_true.append(calculate_new_coordinates(marker_coordinates, rect[1], param, -param))
    marker_coordinates_true.append(calculate_new_coordinates(marker_coordinates, rect[2], param, param))
    marker_coordinates_true.append(calculate_new_coordinates(marker_coordinates, rect[3], -param, param))
    
    marker_coordinates_true = np.array([marker_coordinates_true]).reshape(-1, 1, 2)
    # marker_coordinates_true = marker_coordinates_true.reshape(-1, 2).astype(int).tolist()
    # print(f"\nRESULT:")
    # print(f"    P1 (Top-Left): ({rect[0][0]:.2f}, {rect[0][1]:.2f})")
    # print(f"    P2 (Top-Right): ({rect[1][0]:.2f}, {rect[1][1]:.2f})")
    # print(f"    P3 (Bottom-Right): ({rect[2][0]:.2f}, {rect[2][1]:.2f}) ← Marker2")
    # print(f"    P4 (Bottom-Left): ({rect[3][0]:.2f}, {rect[3][1]:.2f})")
    # print(f"    Rotation angle: {alpha_degrees:.2f}°")
    # print("=" * 60)
    
    return marker_coordinates_true, alpha_degrees


# ============================================ ROTATE IMAGE BY ANGLE =======================================

def rotate_image_by_angle(image, angle_degrees, center=None):
    """
    Rotate the image by the calculated alpha angle
    
    Args:
        image: Input image (numpy array)
        angle_degrees: Rotation angle (degrees) - positive = counter-clockwise
        center: Center of rotation (x, y). If None, uses the image center
    
    Returns:
        rotated_image: The rotated image
    """
    if image is None:
        raise ValueError("Input image must not be None")
    
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Determine the center of rotation
    if center is None:
        center = (width // 2, height // 2)
    
    # Create the rotation matrix
    # Negative angle_degrees to rotate clockwise (commonly used for image correction)
    rotation_matrix = cv2.getRotationMatrix2D(center, -angle_degrees, 1.0)
    
    # Calculate the new image dimensions to avoid corner clipping
    cos_val = abs(rotation_matrix[0, 0])
    sin_val = abs(rotation_matrix[0, 1])
    
    new_width = int((height * sin_val) + (width * cos_val))
    new_height = int((height * cos_val) + (width * sin_val))
    
    # Adjust the rotation matrix so the image fits within the new frame (no clipping)
    rotation_matrix[0, 2] += (new_width / 2) - center[0]
    rotation_matrix[1, 2] += (new_height / 2) - center[1]
    
    # Perform the image rotation
    rotated_image = cv2.warpAffine(image, rotation_matrix, (new_width, new_height), 
                                   flags=cv2.INTER_LANCZOS4, 
                                   borderMode=cv2.BORDER_CONSTANT, 
                                   borderValue=(255, 255, 255))  # White background
    
    return rotated_image, rotation_matrix
