# https://github.com/facebookresearch/projectaria_tools/blob/main/core/examples/mps_quickstart_tutorial.ipynb
# https://facebookresearch.github.io/projectaria_tools/docs/data_utilities/getting_started#step-5-run-machine-perception-services-mps-quickstart-tutorial
# Set this to the folder containing sample data. E.g /Users/myname/Downloads/MPS_data
import os
from projectaria_tools.core import data_provider, mps
from projectaria_tools.core.stream_id import StreamId
import numpy as np

import plotly.graph_objs as go
from matplotlib import pyplot as plt
import copy
import cv2
import json
import inspect

VIS = False
ROTATE90 = True
# Helper function to build the frustrum
def build_cam_frustum(transform_world_device):
    points = np.array([[0, 0, 0], [0.5, 0.5, 1], [-0.5, 0.5, 1], [-0.5, -0.5, 1], [0.5, -0.5, 1]]) * 0.6
    transform_world_rgb = transform_world_device.matrix() @ T_device_RGB.matrix()
    points_transformed = (transform_world_rgb[:3, :3] @ points.T).T + transform_world_rgb[:3, 3]
    return go.Mesh3d(x=points_transformed[:, 0], y=points_transformed[:, 1], z=points_transformed[:, 2],
        i=[0, 0, 0, 0, 1, 1], j=[1, 2, 3, 4, 2, 3], k=[2, 3, 4, 1, 3, 4], showscale=False, visible=False,
        colorscale="jet", intensity=points[:, 2], opacity=1.0, hoverinfo='none')

# Helper function to get nearest eye gaze for a timestamp
def get_nearest_eye_gaze(eye_gazes, query_timestamp_ns):
    return eye_gazes[min(range(len(eye_gazes)), key = lambda i: abs(eye_gazes[i].tracking_timestamp.total_seconds()*1e9 - query_timestamp_ns))]

def get_nearest_trajectory(trajectory, query_timestamp_ns):
    return trajectory[min(range(len(trajectory)), key = lambda i: abs(trajectory[i].tracking_timestamp.total_seconds()*1e9 - query_timestamp_ns))]


# scene_name = 'fountain1'
scene_name = 'cs-lounge-1'
mps_sample_path = "/home/jqiu/2TB_SSD1/Datasets/EgoNeRF_Dataset/"+scene_name
vrsfile = os.path.join(mps_sample_path, scene_name + ".vrs")
LOAD_TRAJECTORY = True  # False
# Create data provider and get T_device_rgb
provider = data_provider.create_vrs_data_provider(vrsfile)
# Since we want to display the position of the RGB camera, we are querying its relative location
# from the device and will apply it to the device trajectory.
T_device_RGB = provider.get_device_calibration().get_transform_device_sensor("camera-rgb")
print('T_device_RGB: ', T_device_RGB)

if LOAD_TRAJECTORY:
    # Trajectory and global points
    # closed_loop_trajectory = os.path.join(mps_sample_path, scene_name + "_" + "closed_loop_trajectory.csv")
    closed_loop_trajectory = os.path.join(mps_sample_path, "closed_loop_trajectory.csv")
    # global_points = os.path.join(mps_sample_path, "trajectory", "global_points.csv.gz")

    #Eye Gaze
    # eyegaze = os.path.join(mps_sample_path, scene_name + "_eyegaze.csv")
    eyegaze = os.path.join(mps_sample_path, "generalized_eye_gaze.csv")

    ## Load trajectory and global points
    mps_trajectory = mps.read_closed_loop_trajectory(closed_loop_trajectory)
    # points = mps.read_global_point_cloud(global_points, mps.StreamCompressionMode.GZIP)

    ## Load eyegaze
    eye_gazes = mps.read_eyegaze(eyegaze)

    print('mps_trajectory.shape, eye_gazes.shape: ', len(mps_trajectory), len(eye_gazes))
    # Load all world positions from the trajectory
    traj = np.empty([len(mps_trajectory), 3])
    for i in range(len(mps_trajectory)):
        traj[i, :] = mps_trajectory[i].transform_world_device.translation()

    # Subsample trajectory for quick display camera
    skip = 1000
    mps_trajectory_subset = mps_trajectory[::skip]
    steps = [None] * len(mps_trajectory_subset)

    # Load each pose as a camera frustum trace
    cam_frustums = [None] * len(mps_trajectory_subset)

    for i in range(len(mps_trajectory_subset)):
        pose = mps_trajectory_subset[i]
        cam_frustums[i] = build_cam_frustum(pose.transform_world_device)
        timestamp = pose.tracking_timestamp.total_seconds()
        step = dict(method="update",
                    args=[{"visible": [False] * len(cam_frustums) + [True] * 2}, {"title": "Trajectory and Point Cloud"}, ],
                    label=timestamp, )
        step["args"][0]["visible"][i] = True  # Toggle i'th trace to "visible"
        steps[i] = step
    cam_frustums[0].visible = True

    # Filter the point cloud by inv depth and depth and load
    threshold_invdep = 0.001
    threshold_dep = 0.15
    skip = 1
    # point_cloud = np.empty([len(points), 3])
    # j = 0
    # for i in range(0, len(points), skip):
    #     if (points[i].inverse_distance_std < threshold_invdep and points[i].distance_std < threshold_dep):
    #         point_cloud[j, :] = points[i].position_world
    #         j = j + 1
    # point_cloud = point_cloud[:j, :]

    # Create slider to allow scrubbing and set the layout
    sliders = [dict(currentvalue={"suffix": " s", "prefix": "Time :"}, pad={"t": 5}, steps=steps, )]
    layout = go.Layout(sliders=sliders,
                       scene=dict(bgcolor='lightgray', dragmode='orbit', aspectmode='data', xaxis_visible=False,
                                  yaxis_visible=False, zaxis_visible=False),
                       title='trajectory')

    # Plot trajectory and point cloud
    # We color the points by their z coordinate
    trajectory = go.Scatter3d(x=traj[:, 0], y=traj[:, 1], z=traj[:, 2], mode="markers",
                              marker={"size": 2, "opacity": 0.8, "color": "red"}, name="Trajectory", hoverinfo='none')
    # global_points = go.Scatter3d(x=point_cloud[:, 0], y=point_cloud[:, 1], z=point_cloud[:, 2], mode="markers",
    #                              marker={"size": 1.5, "color": point_cloud[:, 2], "cmin": -1.5, "cmax": 2,
    #                                      "colorscale": "viridis", },
    #                              name="Global Points", hoverinfo='none')

    # draw
    # plot_figure = go.Figure(data=cam_frustums + [trajectory, global_points], layout=layout)
    if VIS:
        plot_figure = go.Figure(data=cam_frustums + [trajectory], layout=layout)
        plot_figure.show()

# https://facebookresearch.github.io/projectaria_tools/docs/data_formats/aria_vrs/aria_vrs_format
# Sensor	Stream ID	Recordable type ID	label
# ET camera	211-1	EyeCameraRecordableClass	camera-et
# RGB camera	214-1	RgbCameraRecordableClass	camera-rgb
# Microphone	231-1	StereoAudioRecordableClass	mic
# Barometer	247-1	BarometerRecordableClass	baro
# GPS	281-1	GpsRecordableClass	gps
# Wi-Fi	282-1	WifiBeaconRecordableClass	wps
# Bluetooth	283-1	BluetoothBeaconRecordableClass	bluetooth
# SLAM/Mono Scene camera left	1201-1	SlamCameraData	camera-slam-left
# SLAM/Mono Scene camera right	1201-2	SlamCameraData	camera-slam-right
# IMU (1kHz)	1202-1	SlamImuData	imu-right
# IMU (800Hz)	1202-2	SlamImuData	imu-left
# Magnetometer	1203-1	SlamMagnetometerData	mag

rgb_stream_id = StreamId("214-1")
num_rgb_frames = provider.get_num_data(rgb_stream_id)
print('num_rgb_frames: ', num_rgb_frames)
et_stream_id = StreamId("211-1")
num_et_frames = provider.get_num_data(et_stream_id)
print('num_et_frames: ', num_et_frames)
camera_slam_left_stream_id = StreamId("1201-1")
num_camera_slam_left_frames = provider.get_num_data(camera_slam_left_stream_id)
print('num_camera_slam_left_frames: ', num_camera_slam_left_frames)
camera_slam_right_stream_id = StreamId("1201-2")
num_camera_slam_right_frames = provider.get_num_data(camera_slam_right_stream_id)
print('num_camera_slam_right_frames: ', num_camera_slam_right_frames)
# ---------------------------------
microphone_stream_id = StreamId("231-1")
num_microphone_frames = provider.get_num_data(microphone_stream_id)
print('num_microphone_frames: ', num_microphone_frames)
barometer_stream_id = StreamId("247-1")
num_barometer_frames = provider.get_num_data(barometer_stream_id)
print('num_barometer_frames: ', num_barometer_frames)
gps_stream_id = StreamId("281-1")
num_gps_frames = provider.get_num_data(gps_stream_id)
print('num_gps_frames: ', num_gps_frames)
wifi_stream_id = StreamId("282-1")
num_wifi_frames = provider.get_num_data(wifi_stream_id)
print('num_wifi_frames: ', num_wifi_frames)
bluetooth_stream_id = StreamId("283-1")
num_bluetooth_frames = provider.get_num_data(bluetooth_stream_id)
print('num_bluetooth_frames: ', num_bluetooth_frames)
imu1k_stream_id = StreamId("1202-1")
num_imu1k_frames = provider.get_num_data(imu1k_stream_id)
print('num_imu1k_frames: ', num_imu1k_frames)
imu800_stream_id = StreamId("1202-2")
num_imu800_frames = provider.get_num_data(imu800_stream_id)
print('num_imu800_frames: ', num_imu800_frames)
magnetometer_stream_id = StreamId("1203-1")
num_magnetometer_frames = provider.get_num_data(magnetometer_stream_id)
print('num_magnetometer_frames: ', num_magnetometer_frames)

# num_rgb_frames:  1261
# num_et_frames:  1261
# num_camera_slam_left_frames:  1261
# num_camera_slam_right_frames:  1261
# num_microphone_frames:  2953
# num_barometer_frames:  6276
# num_gps_frames:  127
# num_wifi_frames:  340
# num_bluetooth_frames:  1
# num_imu1k_frames:  126029
# num_imu800_frames:  102124
# num_magnetometer_frames:  1262

info_dict = {'gaze': [], # {'gaze_center_in_pixels': [], 'eye_gaze': [], 'gaze_center_in_camera': []},
             'trajecotry': [],
             'rgb_path': [],
             'et_path': [],
             'camera_slam_left_path': [],
             'camera_slam_right': [],
             }
rgb_folder = os.path.join(scene_name, 'rgb')
os.makedirs(rgb_folder, exist_ok=True)

et_folder = os.path.join(scene_name, 'et')
os.makedirs(et_folder, exist_ok=True)

camera_slam_left_folder = os.path.join(scene_name, 'camera_slam_left')
os.makedirs(camera_slam_left_folder, exist_ok=True)

camera_slam_right_folder = os.path.join(scene_name, 'camera_slam_right')
os.makedirs(camera_slam_right_folder, exist_ok=True)

for rgb_frames_i in range(num_rgb_frames):
    print('Processing {}/{}'.format(rgb_frames_i, num_rgb_frames))
    rgb_frame = provider.get_image_data_by_index(rgb_stream_id, (int)(rgb_frames_i))
    assert rgb_frame[0] is not None, "no rgb frame"

    image = rgb_frame[0].to_numpy_array()
    capture_timestamp_ns = rgb_frame[1].capture_timestamp_ns
    eye_gaze = get_nearest_eye_gaze(eye_gazes, capture_timestamp_ns)
    # print('eye_gaze: ', dir(eye_gaze))  # 'depth', 'pitch', 'pitch_high', 'pitch_low', 'tracking_timestamp', 'yaw', 'yaw_high', 'yaw_low'
    current_mps_trajectory = get_nearest_trajectory(mps_trajectory, capture_timestamp_ns)
    current_translation = current_mps_trajectory.transform_world_device.translation()  # 3,
    current_rotation = current_mps_trajectory.transform_world_device.rotation_matrix()  # 3 x 3
    # print('current_translation, current_rotation: ', current_translation, current_rotation)
    current_mps_trajectory = np.concatenate((current_translation, current_rotation.reshape(-1))).tolist()

    # get projection function
    device_calibration = provider.get_device_calibration()
    cam_calibration = device_calibration.get_camera_calib(provider.get_label_from_stream_id(rgb_stream_id))
    assert cam_calibration is not None, "no camera calibration"

    depth_m = 1.0 # Select a fixed depth of 1m
    gaze_center_in_cpf = mps.get_eyegaze_point_at_depth(eye_gaze.yaw, eye_gaze.pitch, depth_m)
    transform_cpf_sensor = provider.get_device_calibration().get_transform_cpf_sensor(provider.get_label_from_stream_id(rgb_stream_id))
    gaze_center_in_camera = transform_cpf_sensor.inverse().matrix()@ np.hstack((gaze_center_in_cpf, 1)).T
    gaze_center_in_camera = gaze_center_in_camera[:3] / gaze_center_in_camera[3:]
    gaze_center_in_pixels = cam_calibration.project(gaze_center_in_camera)

    info_dict['gaze'].append(copy.deepcopy({'gaze_center_in_pixels': copy.deepcopy(gaze_center_in_pixels).tolist(),
                                            # 'eye_gaze': copy.deepcopy(eye_gaze),
                                            'gaze_center_in_camera': copy.deepcopy(gaze_center_in_camera).tolist()
                                            }))
    rgb_path = os.path.join(rgb_folder, '{:09d}.png'.format(rgb_frames_i))
    if ROTATE90:
        image = np.rot90(image, -1)
    cv2.imwrite(rgb_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    info_dict['rgb_path'].append(rgb_path)

    et_frame = provider.get_image_data_by_index(et_stream_id, (int)(rgb_frames_i))
    assert et_frame[0] is not None, "no et frame"
    et_np = et_frame[0].to_numpy_array()
    et_path = os.path.join(et_folder, '{:09d}.png'.format(rgb_frames_i))
    cv2.imwrite(et_path, cv2.cvtColor(et_np, cv2.COLOR_RGB2BGR))
    info_dict['et_path'].append(et_path)

    camera_slam_left_frame = provider.get_image_data_by_index(camera_slam_left_stream_id, (int)(rgb_frames_i))
    assert camera_slam_left_frame[0] is not None, "no et frame"
    camera_slam_left_np = camera_slam_left_frame[0].to_numpy_array()
    camera_slam_left_path = os.path.join(camera_slam_left_folder, '{:09d}.png'.format(rgb_frames_i))
    if ROTATE90:
        camera_slam_left_np = np.rot90(camera_slam_left_np, -1)
    cv2.imwrite(camera_slam_left_path, cv2.cvtColor(camera_slam_left_np, cv2.COLOR_RGB2BGR))
    info_dict['camera_slam_left_path'].append(camera_slam_left_path)

    camera_slam_right_frame = provider.get_image_data_by_index(camera_slam_right_stream_id, (int)(rgb_frames_i))
    assert camera_slam_right_frame[0] is not None, "no et frame"
    camera_slam_right_np = camera_slam_right_frame[0].to_numpy_array()
    camera_slam_right_path = os.path.join(camera_slam_right_folder, '{:09d}.png'.format(rgb_frames_i))
    if ROTATE90:
        camera_slam_right_np = np.rot90(camera_slam_right_np, -1)
    cv2.imwrite(camera_slam_right_path, cv2.cvtColor(camera_slam_right_np, cv2.COLOR_RGB2BGR))
    info_dict['camera_slam_left_path'].append(camera_slam_right_path)
    # Draw a cross at the projected gaze center location
    if VIS:
        fig, ax = plt.subplots()
        if gaze_center_in_pixels is not None:
            ax.imshow(image);
            ax.plot(gaze_center_in_pixels[0], gaze_center_in_pixels[1], '+', c="red", mew=1, ms=20)
            plt.title('gaze center in pixels')
            plt.show()
        else:
            print(f"Eye gaze center projected to {gaze_center_in_pixels}, which is out of camera sensor plane.")

json_path = os.path.join(scene_name, 'info_dict.json')
with open(os.path.join(scene_name, 'info_dict.json'), "w") as outfile:
    json.dump(info_dict, outfile)

print('info_dict.json saved to {}'.format(json_path))
