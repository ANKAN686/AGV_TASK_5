import cv2
import numpy as np

# 1. Load your specific map image
img = cv2.imread('aces_relations.png')

# 2. Convert from BGR (OpenCV default) to HSV color space to isolate the green easily
hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# 3. Define the HSV range for the green pixels we want to remove
lower_green = np.array([40, 40, 40]) 
upper_green = np.array([80, 255, 255])

# 4. Create a mask and apply it to replace those green pixels with pure white
green_mask = cv2.inRange(hsv_img, lower_green, upper_green)
img[green_mask > 0] = [255, 255, 255]

# 5. Save the cleaned map as a new file
cv2.imwrite('cleaned_map.png', img)
print("Map cleaned successfully!")

# --- PHASE 2: BUILDING THE NDT GRID ---
print("Building NDT Grid...")

# 1. Load the map you just cleaned, but in grayscale (easier to read)
map_img = cv2.imread('cleaned_map.png', cv2.IMREAD_GRAYSCALE)

# 2. Extract the coordinates of all the obstacle pixels (white pixels = 255)
# np.where returns the (y, x) coordinates of every white pixel
y_coords, x_coords = np.where(map_img == 255)
obstacle_points = np.column_stack((x_coords, y_coords))

# 3. Define the size of our NDT grid cells
# Let's say each cell is 20x20 pixels.
cell_size = 20
grid_cells = {}

# 4. Group the pixels into their specific grid cells
for point in obstacle_points:
    x, y = point
    # Integer division (//) figures out which cell index the point falls into
    cell_x = x // cell_size
    cell_y = y // cell_size
    cell_id = (cell_x, cell_y)
    
    if cell_id not in grid_cells:
        grid_cells[cell_id] = []
    grid_cells[cell_id].append(point)

# 5. Calculate the Mean (mu) and Covariance (Sigma) for each cell
ndt_map = {}

for cell_id, points in grid_cells.items():
    pts_array = np.array(points)
    
    # A cell needs at least 3 points to calculate a meaningful covariance matrix
    if len(pts_array) >= 3:
        # np.mean calculates the center [x, y] of the points
        mu = np.mean(pts_array, axis=0)
        
        # np.cov calculates the spread/shape. 
        # rowvar=False tells it our points are in rows, not columns
        sigma = np.cov(pts_array, rowvar=False)
        
        # Save our math to the final NDT map!
        ndt_map[cell_id] = {'mean': mu, 'cov': sigma}

print(f"Success! Created an NDT Map with {len(ndt_map)} valid cells.")


# --- PHASE 3: PARSING THE LiDAR DATA ---
print("Parsing the CARMEN log file...")

def parse_clf_dataset(filepath):
    scans = []
    
    with open(filepath, 'r') as f:
        for line in f:
            # We only care about the lines containing Front Laser data (FLASER)
            if line.startswith('FLASER'):
                parts = line.split()
                
                # The second item is the number of laser points in this specific scan
                num_readings = int(parts[1])
                
                # Extract the actual distance measurements
                ranges = np.array([float(x) for x in parts[2:2 + num_readings]])
                
                
                x = float(parts[2 + num_readings])
                y = float(parts[3 + num_readings])
                theta = float(parts[4 + num_readings])
                
                scans.append({
                    'ranges': ranges,
                    'pose': np.array([x, y, theta])
                })
                
    return scans


dataset_file = 'aces.clf' 
lidar_data = parse_clf_dataset(dataset_file)

print(f"Successfully extracted {len(lidar_data)} LiDAR scans!")


# --- PHASE 4: SCALED TRANSFORMATION ---
def transform_scan(ranges, pose, scale):
    """ Translates laser distances (meters) into map coordinates (pixels) """
    x, y, theta = pose
    angles = np.linspace(-np.pi/2, np.pi/2, len(ranges))
    
    local_x = (ranges * scale) * np.cos(angles)
    local_y = (ranges * scale) * np.sin(angles)
    
    global_x = x + local_x * np.cos(theta) - local_y * np.sin(theta)
    global_y = y + local_x * np.sin(theta) + local_y * np.cos(theta)
    
    return np.column_stack((global_x, global_y))

# --- PHASE 5: VISUAL DEBUGGER ---
def draw_calibration(map_path, scan_ranges, initial_pose, scale, offset_x, offset_y):
    img = cv2.imread(map_path)
    
    start_x = (initial_pose[0] * scale) + offset_x
    start_y = (initial_pose[1] * scale) + offset_y
    theta = initial_pose[2]
    
    test_pose = [start_x, start_y, theta]
    points = transform_scan(scan_ranges, test_pose, scale)
    
    cv2.circle(img, (int(start_x), int(start_y)), 8, (255, 0, 0), -1)
    
    for pt in points:
        px, py = int(pt[0]), int(pt[1])
        if 0 <= px < img.shape[1] and 0 <= py < img.shape[0]:
            cv2.circle(img, (px, py), 1, (0, 0, 255), -1)
            
    cv2.imwrite('calibration_view.png', img)
    print("Saved 'calibration_view.png'! Open it to see the perfect alignment.")

# --- PHASE 6: SIMULATION & OPTIMIZATION LOOP ---
print("\nStarting NDT Tracking...")

def ndt_score(pose, scan_ranges, ndt_map, cell_size, scale, offset_x, offset_y):
    # Translate pose to pixels
    px_pose = [(pose[0] * scale) + offset_x, (pose[1] * scale) + offset_y, pose[2]]
    points = transform_scan(scan_ranges, px_pose, scale)
    
    score = 0
    for pt in points:
        cell_id = (int(pt[0] // cell_size), int(pt[1] // cell_size))
        if cell_id in ndt_map:
            mu = ndt_map[cell_id]['mean']
            cov = ndt_map[cell_id]['cov']
            try:
                # The core NDT Gaussian probability formula
                inv_cov = np.linalg.inv(cov)
                diff = pt - mu
                exponent = -0.5 * np.dot(np.dot(diff.T, inv_cov), diff)
                score += np.exp(exponent)
            except np.linalg.LinAlgError:
                pass
    return score

def optimize_pose(initial_pose, scan_ranges, ndt_map, cell_size, scale, offset_x, offset_y, iterations=10):
    current_pose = np.copy(initial_pose)
    current_score = ndt_score(current_pose, scan_ranges, ndt_map, cell_size, scale, offset_x, offset_y)
    
    step_trans = 0.05 # Move by 5cm
    step_rot = 0.02   # Rotate by a small angle
    
    for _ in range(iterations):
        best_score = current_score
        best_pose = current_pose
        
        nudges = [
            np.array([step_trans, 0, 0]), np.array([-step_trans, 0, 0]),
            np.array([0, step_trans, 0]), np.array([0, -step_trans, 0]),
            np.array([0, 0, step_rot]), np.array([0, 0, -step_rot])
        ]
        
        for nudge in nudges:
            test_pose = current_pose + nudge
            score = ndt_score(test_pose, scan_ranges, ndt_map, cell_size, scale, offset_x, offset_y)
            if score > best_score:
                best_score = score
                best_pose = test_pose
                
        if best_score <= current_score:
            break # We found the peak!
            
        current_pose = best_pose
        current_score = best_score
        
    return current_pose

# THE EXACT ACES CALIBRATION
exact_scale = 20.0  
exact_offset_x = 334 
exact_offset_y = 331 

# --- PHASE 6: FAST-FORWARD SIMULATION ---
# Let's cover 2,000 steps of time, but only do the heavy math every 10th step!
trajectory = []
current_estimate = lidar_data[0]['pose'] 

print("Optimizing trajectory. Fast-forwarding through time...")
for i in range(0, 2000, 10): 
    scan = lidar_data[i]
    
    # Optimize the pose
    optimized_pose = optimize_pose(current_estimate, scan['ranges'], ndt_map, cell_size, exact_scale, exact_offset_x, exact_offset_y)
    trajectory.append(optimized_pose)
    
    # Calculate how far the internal wheels think they moved during our 10-step skip
    if i + 10 < len(lidar_data):
        odom_delta = lidar_data[i+10]['pose'] - lidar_data[i]['pose']
        current_estimate = optimized_pose + odom_delta

    # Print an update every 100 scans so we know it isn't frozen
    if i % 100 == 0:
        print(f"Time-jump: Processed up to scan {i} / 2000...")

# Draw the final tracked trajectory!
img = cv2.imread('cleaned_map.png')
for p in trajectory:
    px = int((p[0] * exact_scale) + exact_offset_x)
    py = int((p[1] * exact_scale) + exact_offset_y)
    cv2.circle(img, (px, py), 2, (255, 0, 0), -1) # Draw a blue path

cv2.imwrite('final_trajectory.png', img)
print("\nTask Complete! Saved 'final_trajectory.png'.")