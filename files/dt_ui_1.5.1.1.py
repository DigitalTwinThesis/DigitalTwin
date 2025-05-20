#final?
from PyQt6.QtWidgets import *
from PyQt6.QtCore import *
from PyQt6.QtGui import *
from robodk.robolink import * # RoboDK API
from robodk.robomath import *  # Math functions for transformations
from time import *
from threading import Thread # Import the Thread class from the threading module
import casadi as ca  # Import CasADi functions
import numpy as np  # Import NumPy functions
import pandas as pd  # Import Pandas functions
import matplotlib.pyplot as plt  # Import Matplotlib for plotting
import matplotlib.dates as mdates  # Import Matplotlib for date formatting
from datetime import datetime  # Import datetime for timestamps
import socket
import cv2

HOME = 'Home'  # Target name for the robot home position
SCRIPT = 'history'
MSTEP = 10  # Step size for movement in mm or degrees
DSTEP = 3   
USER_MAN_SPEED = 20
USER_AUTO_SPEED = 100
IP = "192.168.0.102"
PORT = 30002

base = __file__.rsplit("\\", 1)[0]  # Windows-only compatibitily!
JOINTS_PATH = f"{base}\\joints.csv"
HISTORY_PATH = f"{base}\\joint_usage_history.csv"
GRIPPER_FOLDER_PATH = f"{base}\\local_files"
GRIPPER_OPEN_PATH = f"{base}\\gripper_open.script"
GRIPPER_OPEN_CELL_PATH = f"{base}\\gripper_open_cell.script"
GRIPPER_CLOSE_BOX_PATH = f"{base}\\gripper_close_box.script"
GRIPPER_CLOSE_CELL_PATH = f"{base}\\gripper_close_cell.script"
GRIPPER_CLOSE_LID_PATH = f"{base}\\gripper_close_lid.script"
RESET_IMAGE_PATH = f"{base}\\reset_cell.png"
ROBOTIC_STATION_PATH = f"{base}\\robotic_station.rdk"

###################################################################################################################
#################### VISION SYSTEM ################################################################################
###################################################################################################################
class Vision():
    def __init__(self):
        
        self.cap = None

        # Define battery molds (rectangles)
        rectwidth = 35
        recthigh = 230
        self.cell_mold = [
            (710, 535, rectwidth, recthigh), (650, 535, rectwidth, recthigh),
            (590, 535, rectwidth, recthigh), (530, 535, rectwidth, recthigh),
            (465, 535, rectwidth, recthigh), (400, 535, rectwidth, recthigh),
            (340, 535, rectwidth, recthigh), (270, 535, rectwidth, recthigh),
            (210, 535, rectwidth, recthigh), (145, 535, rectwidth, recthigh),
            (80, 535, rectwidth, recthigh),  (25, 535, rectwidth, recthigh)
        ]

        self.box_mold = [
            (600, 25, 40, 200), (440, 25, 45, 200),
            (290, 30, 45, 200)
        ]

        self.lid_mold = [
            (580, 270, 40, 200) , (430, 270, 45, 200),
            (280, 270, 45, 200)
        ]

        self.cell_detected = 0
        self.cell_mold_detected = []
        self.box_mold_detected = []
        self.lid_mold_detected = []
        self.cell_mold_empty = []
        self.box_mold_empty = []
        self.lid_mold_empty = []

        self.cap = cv2.VideoCapture(0)


    def start_up(self):
        ret, frame = self.cap.read()

        if not ret:
            print("Error in the image.")
            self.cap.release()
            exit()

        # Perspective correction
        initialpoint = np.float32([[90, 30], [530, 190], [15, 320], [590, 420]])
        finalpoint = np.float32([[0, 0], [800, 0], [0, 800], [800, 800]])
        M = cv2.getPerspectiveTransform(initialpoint, finalpoint)
        self.dst = cv2.warpPerspective(frame, M, (800, 800)) 

    # Show image
    def view (self):
        
        cv2.imshow('Estantería con detección', self.dst)
        cv2.waitKey(0)


    # Function to detect red colour in a region
    def detect_cell_colour(self, cell_rect):
        hsv = cv2.cvtColor(cell_rect, cv2.COLOR_BGR2HSV)

        # Red ranges (two zones in HSV)
        low_red1 = np.array([0, 100, 100])
        high_red1 = np.array([10, 255, 255])
        low_red2 = np.array([160, 100, 100])
        high_red2 = np.array([180, 255, 255])

        mask1 = cv2.inRange(hsv, low_red1, high_red1)
        mask2 = cv2.inRange(hsv, low_red2, high_red2)
        red_mask = cv2.bitwise_or(mask1, mask2)

        return cv2.countNonZero(red_mask)
    
     # Function to detect white colour in a region
    def detect_white_colour(self, rect):
        hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)

        low_white = np.array([0, 0, 200])
        high_white = np.array([180, 30, 255])

        white_mask = cv2.inRange(hsv, low_white, high_white)

        return cv2.countNonZero(white_mask)
    
    # Function to detect grey colour in a region
    def detect_grey_colour(self, rect):
        hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)

        low_grey = np.array([100, 20, 120])
        high_grey = np.array([120, 110, 170])

        grey_mask = cv2.inRange(hsv, low_grey, high_grey)

        return cv2.countNonZero(grey_mask)
    
    # Function to detect orange colour in a region
    def detect_orange_colour(self, rect):
        hsv = cv2.cvtColor(rect, cv2.COLOR_BGR2HSV)

        low_orange = np.array([10, 50, 200])
        high_orange = np.array([25, 255, 255])

        orange_mask = cv2.inRange(hsv, low_orange, high_orange)

        return cv2.countNonZero(orange_mask)
    
    def detect_cell(self):
        self.cell_mold_detected = []
        for i, (x, y, w, h) in enumerate(self.cell_mold):
            rect = self.dst[y:y+h, x:x+w]
            red_area = self.detect_cell_colour(rect)
            white_area = self.detect_white_colour(rect)

            # Detect battery only if both red and white areas are present
            if red_area > 50 and white_area > 50:
                self.cell_detected += 1
                self.cell_mold_detected.append(i + 1)  # Save mold number
                cv2.rectangle(self.dst, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(self.dst, 'Cell', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            else:
                self.cell_mold_empty.append(i + 1)  # Save empty molds
                cv2.rectangle(self.dst, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(self.dst, 'Empty', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

    def detect_box(self):
        self.box_mold_detected = []
        for i, (x, y, w, h) in enumerate(self.box_mold):
            rect = self.dst[y:y+h, x:x+w]
            
            if i == 0:  # Primer rectángulo: debe detectar naranja
                orange_area = self.detect_orange_colour(rect)
                if orange_area > 6000:
                    self.box_mold_detected.append(i + 1)  # Save mold number
                    cv2.rectangle(self.dst, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Naranja
                    cv2.putText(self.dst, 'Orange Box', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    self.box_mold_empty.append(i + 1)  # Save empty molds
                    cv2.rectangle(self.dst, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(self.dst, 'Empty', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            elif i == 1:  # Segundo rectángulo: debe detectar gris
                grey_area = self.detect_grey_colour(rect)
                if grey_area > 500:
                    self.box_mold_detected.append(i + 1)  # Save mold number
                    cv2.rectangle(self.dst, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Gris
                    cv2.putText(self.dst, 'Grey Box', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    self.box_mold_empty.append(i + 1)  # Save empty molds
                    cv2.rectangle(self.dst, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(self.dst, 'Empty', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            elif i == 2:  # Tercer rectángulo: debe detectar blanco
                white_area = self.detect_white_colour(rect)
                if white_area > 6000:
                    self.box_mold_detected.append(i + 1)  # Save mold number
                    cv2.rectangle(self.dst, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Blanco
                    cv2.putText(self.dst, 'White Box', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    self.box_mold_empty.append(i + 1)  # Save empty molds
                    cv2.rectangle(self.dst, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(self.dst, 'Empty', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            
            else:
                self.box_mold_empty.append(i + 1)  # Save mold number
                cv2.rectangle(self.dst, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(self.dst, 'Empty', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)


    def detect_lid(self):
        self.lid_mold_detected = []
        for i, (x, y, w, h) in enumerate(self.lid_mold):
            rect = self.dst[y:y+h, x:x+w]

            # Detect color based on the specific rectangle
            if i == 0:  # Primer rectángulo: debe detectar naranja
                orange_area = self.detect_orange_colour(rect)
                if orange_area > 6000:
                    self.lid_mold_detected.append(i + 1)  # Save lid number
                    cv2.rectangle(self.dst, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Naranja
                    cv2.putText(self.dst, 'Orange Lid', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    self.lid_mold_empty.append(i + 1)  # Save empty lids
                    cv2.rectangle(self.dst, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(self.dst, 'Empty', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            elif i == 1:  # Segundo rectángulo: debe detectar gris
                grey_area = self.detect_grey_colour(rect)
                if grey_area > 500:
                    self.lid_mold_detected.append(i + 1)  # Save lid number
                    cv2.rectangle(self.dst, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Gris
                    cv2.putText(self.dst, 'Grey Lid', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    self.lid_mold_empty.append(i + 1)  # Save empty lids
                    cv2.rectangle(self.dst, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(self.dst, 'Empty', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            elif i == 2:  # Tercer rectángulo: debe detectar blanco
                white_area = self.detect_white_colour(rect)
                if white_area > 6000:
                    self.lid_mold_detected.append(i + 1)  # Save lid number
                    cv2.rectangle(self.dst, (x, y), (x+w, y+h), (0, 255, 0), 2)  # Blanco
                    cv2.putText(self.dst, 'White Lid', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                else:
                    self.lid_mold_empty.append(i + 1)  # Save empty lids
                    cv2.rectangle(self.dst, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(self.dst, 'Empty', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

            else:
                self.lid_mold_empty.append(i + 1)  # Save empty lids
                cv2.rectangle(self.dst, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(self.dst, 'Empty', (x, y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

###################################################################################################################
#################### PREDICTIVE MAINTENANCE #######################################################################
###################################################################################################################
class Maintenance():
    def __init__(self, program=None):
        self.df = None
        self.historical_df = None
        self.program = program

    def extract_data(self):
        msg, joint_list, status = self.program.InstructionListJoints(
            mm_step=1,
            deg_step=1,
            flags=4,
            time_step=0.1,
            save_to_file= JOINTS_PATH
        )
        
    def read_csv(self):
        # Load CSV file
        self.df = pd.read_csv(JOINTS_PATH)
        self.df.columns = self.df.columns.str.strip()  # Remove leading/trailing whitespace

        # Check if the CSV file is empty
        self.time_col = next((col for col in self.df.columns if 'TIME' in col.upper()), None)
        if not self.time_col:
            raise ValueError("Time column not found in the CSV file.")

        ########################################
        # Check if the time column is valid
        is_valid_time = self.df[self.time_col].is_monotonic_increasing and self.df[self.time_col].nunique() > 2
        use_index_time = not is_valid_time
        if use_index_time:
            self.df['Time_Index'] = self.df.index
            self.time_col = 'Time_Index'
        
        # Sort and remove duplicates
        self.df = self.df.sort_values(by=self.time_col).drop_duplicates(subset=self.time_col).reset_index(drop=True)

        # Smooth the data
        for i in range(1, 7):
            self.df[f'J{i}'] = self.df[f'J{i}'].rolling(window=3, min_periods=1).mean()
        ########################################
        
        # Mechanical limits for each joint
        # Source: Universal Robots datasheet
        self.joint_limits = {
            'J1': (-360, 360),
            'J2': (-360, 360),
            'J3': (-360, 360),
            'J4': (-360, 360),
            'J5': (-360, 360),
            'J6': (-360, 360)
        }

    def robot_joints_range(self):
        plt.figure(figsize=(8, 6))
        for i, joint in enumerate(self.joint_limits.keys()):
            used_min = self.df[joint].min()
            used_max = self.df[joint].max()
            mech_min, mech_max = self.joint_limits[joint]

            plt.barh(i, mech_max - mech_min, left=mech_min, color='lightgray', label='Mechanical Limits' if i == 0 else "")
            plt.barh(i, used_max - used_min, left=used_min, color='green', label='Joint Values' if i == 0 else "")

            usage_pct = 100 * (used_max - used_min) / (mech_max - mech_min)
            plt.text(mech_max + 10, i, f"{usage_pct:.2f}%", va='center')

        plt.yticks(range(6), [f'J{i+1}' for i in range(6)])
        plt.gca().invert_yaxis()
        plt.xlabel("Joint Value (deg or mm)")
        plt.title("Robot Joints Range")
        plt.legend()
        plt.tight_layout()
        plt.show()

    def robot_joints_usage(self):
        fig, axes = plt.subplots(6, 1, figsize=(10, 10), sharex=False)
        fig.suptitle("Robot Joints Utilization", fontsize=14)

        for i, ax in enumerate(axes):
            joint = f'J{i+1}'
            ax.hist(self.df[joint], bins=30)
            ax.set_ylabel("Count")
            ax.set_xlabel(f"{joint} Value (deg or mm)")

        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def robot_joints_over_time(self):
        fig, axes = plt.subplots(6, 1, figsize=(10, 10), sharex=True)
        fig.suptitle("Joints Over Time", fontsize=14)

        for i, ax in enumerate(axes):
            joint = f'J{i+1}'
            ax.plot(self.df[self.time_col], self.df[joint], label=f'Joint {i+1}', linewidth=1)
            ax.set_ylabel("Joint (deg/s)")
            ax.legend()

        axes[-1].set_xlabel("Time (s)" if self.time_col == 'Time_Index' else "Sample Index")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()
    
    def robot_joints_velocity(self):
        fig, axes = plt.subplots(6, 1, figsize=(10, 10), sharex=True)
        fig.suptitle("Joint Speeds Over Time", fontsize=14)

        for i, ax in enumerate(axes):
            joint = f'J{i+1}'
            speed_col = f'SPEED_{joint}'
            if speed_col in self.df.columns:
                ax.plot(self.df[self.time_col], self.df[speed_col], label=f'Speed {joint}', color='blue')
                ax.set_ylabel("deg/s")
                ax.legend()
            else:
                ax.text(0.5, 0.5, "N/A", ha='center')

        axes[-1].set_xlabel("Time (s)" if self.time_col == 'Time_Index' else "Sample Index")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def robot_joints_acceleration(self):
        fig, axes = plt.subplots(6, 1, figsize=(10, 10), sharex=True)
        fig.suptitle("Joint Accelerations Over Time", fontsize=14)

        for i, ax in enumerate(axes):
            joint = f'J{i+1}'
            acc_col = f'ACCEL_{joint}'
            if acc_col in self.df.columns:
                ax.plot(self.df[self.time_col], self.df[acc_col], label=f'Accel {joint}', color='purple')
                ax.set_ylabel("deg/s²")
                ax.legend()
            else:
                ax.text(0.5, 0.5, "N/A", ha='center')

        axes[-1].set_xlabel("Time (s)" if self.time_col == 'Time_Index' else "Sample Index")
        plt.tight_layout(rect=[0, 0, 1, 0.96])
        plt.show()

    def historical_data(self):
        joint_usage = {'Timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

        for i in range(1, 7):
            joint = f'J{i}'
            speed_col = f'SPEED_{joint}'
            acc_col = f'ACCEL_{joint}'

            # Calculate total movement
            if joint in self.df.columns:
                diffs = self.df[joint].diff().abs().dropna()
                total_movement = diffs.sum()
                joint_usage[joint] = round(total_movement, 2)
            else:
                joint_usage[joint] = None

            # Average speed
            if speed_col in self.df.columns:
                joint_usage[speed_col] = round(self.df[speed_col].abs().mean(), 2)
            else:
                joint_usage[speed_col] = None

            # Average acceleration
            if acc_col in self.df.columns:
                joint_usage[acc_col] = round(self.df[acc_col].abs().mean(), 2)
            else:
                joint_usage[acc_col] = None
        
        try:
            self.historical_df = pd.read_csv(HISTORY_PATH)
            self.historical_df = pd.concat([self.historical_df, pd.DataFrame([joint_usage])], ignore_index=True)

        except FileNotFoundError:
            self.historical_df = pd.DataFrame([joint_usage])
        self.historical_df.to_csv(HISTORY_PATH, index=False)

    def historical_analysis(self):
        # Load historical data
        self.historical_df = pd.read_csv(HISTORY_PATH)
        self.historical_df['Timestamp'] = pd.to_datetime(self.historical_df['Timestamp'])

        # Total historical movement per joint
        plt.figure(figsize=(12, 6))
        for i in range(1, 7):
            col = f'J{i}'
            if col in self.historical_df.columns:
                plt.plot(range(1, len(self.historical_df) + 1), self.historical_df[col].cumsum(), label=f'J{i}')
        plt.title("Evolution of Historical Total Movement per Joint")
        plt.xlabel("Sample number")
        plt.ylabel("Cummulative Movement (deg)")
        plt.legend()
        plt.grid(True)
        #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        plt.tight_layout()
        plt.show()

        # Average historical speed
        plt.figure(figsize=(12, 6))
        for i in range(1, 7):
            col = f'SPEED_J{i}'
            if col in self.historical_df.columns:
                plt.plot(range(1, len(self.historical_df) + 1), self.historical_df[col], label=f'J{i}')

        plt.title("Evolution of Historical Average Speed per Joint")
        plt.xlabel("Sample number")
        plt.ylabel("Average speed (deg/s)")
        plt.legend()
        plt.grid(True)
        #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        plt.tight_layout()
        plt.show()

        # Average historical acceleration
        plt.figure(figsize=(12, 6))
        for i in range(1, 7):
            col = f'ACCEL_J{i}'
            if col in self.historical_df.columns:
                plt.plot(range(1, len(self.historical_df) + 1), self.historical_df[col], label=f'J{i}')
        plt.title("Evolution of Historical Average Acceleration per Joint")
        plt.xlabel("Sample number")
        plt.ylabel("Average acceleration (deg/s²)")
        plt.legend()
        plt.grid(True)
        #plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d\n%H:%M'))
        plt.tight_layout()
        plt.show()

###################################################################################################################
#################### PATH AND TRAJECTORY CALCULATION ##############################################################
###################################################################################################################
class Path():
    def __init__(self, parent):
        self.stop = False
        self.resume = False
        self.running = False
        self.ui = parent
        self.watching = True
        self.calculation_method = 0  # Default calculation method
        self.mode = 0
        self.cells = []
        self.box = 0
        self.assembled = [[],[],[]]
        if parent.RDK.Item(SCRIPT, ITEM_TYPE_PROGRAM).Valid():
            test_code = parent.RDK.Item(SCRIPT, ITEM_TYPE_PROGRAM)
        else:
            test_code = parent.RDK.AddProgram(SCRIPT)
        self.maintenance = Maintenance(test_code)
        print("[INFO] Starting vision system...")
        self.vision = Vision()
        print("[INFO] Vision system started.")

        self.disassembly = False

################################# PATH SCRIPTS #####################################################################
    def main_script(self):
        try:
            """Main script to execute the robot program."""
            # Set robot speed and acceleration
            self.ui.robot.setSpeed(50)
            self.ui.robot.setAcceleration(100)
            self.ui.robot.setSpeedJoints(15)
            self.ui.robot.setAccelerationJoints(30)

            # If automatic assembly, use vision system to detect box and cells
            if self.mode == 1 or self.mode == 3:
                print("[INFO] Assembly starting...")
                # Call vision system to detect box and cells
                self.vision.start_up()
                self.vision.detect_box()
                if not self.vision.box_mold_detected:
                    print("[ERROR] Box not detected.")
                    self.box = 0
                    return
                self.vision.detect_lid()
                if not self.vision.lid_mold_detected:
                    print("[ERROR] Lid not detected.")
                    self.box = 0
                    return
                self.vision.detect_cell()
                if not self.vision.cell_mold_detected:
                    print("[ERROR] Cells not detected.")
                    self.cells = []
                    return
            
                #self.vision.view()
                # Check if box and lid match
                if self.mode == 1:
                    for detected in self.vision.box_mold_detected:
                        if detected in self.vision.lid_mold_detected:
                            self.box = detected
                            break
                        else:
                            self.box = 0
                    
                    if len(self.vision.cell_mold_detected[:4]) != 4:
                        print("[WARNING] Not enough cells detected")
                        self.cells = []
                        return
                    else:    
                        self.cells = self.vision.cell_mold_detected[:4]  # Get first 4 detected cells
                    
                elif self.mode == 3:
                    if self.box not in self.vision.box_mold_detected or self.box not in self.vision.lid_mold_detected:
                        print("[ERROR] Box or lid selected were not found.")
                        self.box = []
                        return
                
                if len(self.assembled[self.box-1]) != 0:
                    print(f"[WARNING] Selected box {self.box} has already been assembled with cells {self.assembled[self.box-1]}, please return the box and cells to their previous assembled state, or resart and reset the cell.")
                    return
                else:
                    print(f"[INFO] Box and lid correctly selected: {self.box}.")
                    
                for cell in self.cells:
                    if cell not in self.vision.cell_mold_detected:
                        print(f"[ERROR] Selected cell {cell} not detected.")
                        self.cells = []
                        return
                    for finished in self.assembled:
                        if cell in finished:
                            print(f"[ERROR] Cell {cell} is already part of an assembled box, please return the box and cells to their previous assembled state, or resart and reset the cell")
                            self.cells = []
                            return             

                # If no match found, return error
                if self.box == 0:
                    print("[ERROR] There are no matching box/lid.")
                    return     
                
            box = self.box      # Box number to assemble (1, 2, or 3)
            cells = self.cells  # Cells to place in the box (4 cells in total, from 1 to 12)

            if self.disassembly == False:
                # check if the box is already assembled
                # Perform assembly
                print(f"[INFO] Performing assembly of box {box} and cells {cells}")
                self.place_box(box)
                for i in range(1, 5):
                    self.place_cell(cells[i-1], i)
                # Keep track of the assembled box and its cells
                self.assembled[box-1] = cells
                self.place_full_box(box)
                self.place_lid(box)
                print(f"[INFO] Box assembled successfully with cells: {self.assembled[box-1]}.")
            elif self.disassembly == True:
                # check if the box is already assembled
                if len(self.assembled[box-1]) == 0:
                    print("[WARNING] Selected box has not been assembled.")
                    return
                else:
                    # Perform disassembly
                    print(f"[INFO] Performing disassembly of box {box} and cells {cells}")
                    cells = self.assembled[box-1]
                    self.disassemble_lid(box)
                    self.disassemble_full_box(box)
                    for i in range(1, 5):
                        self.disassemble_cell(cells[i-1], i)
                    # Empty the assembled box
                    self.assembled[box-1] = []
                    self.disassemble_box(box)
                    # Replace virtual box and its cells
                    self.ui.virtual_replace(box, cells)
                    
            
            print("[INFO] Main script execution complete.")
            self.running = False
        except Exception as e:
            print(f"[WARNING] {e}")
    
    def script_execution(self, targets):
        '''Execute the script with the given targets'''
        RUN_ON_ROBOT = True

        if self.ui.RDK.RunMode() != RUNMODE_SIMULATE:
            RUN_ON_ROBOT = False

        # Set mode to run on the physical robot
        if RUN_ON_ROBOT:
            self.ui.RDK.setRunMode(RUNMODE_RUN_ROBOT)

        # Set reference and tool
        reference_frame = self.ui.RDK.Item('UR10e Base', ITEM_TYPE_FRAME)
        tool = self.ui.RDK.Item('tcp', ITEM_TYPE_TOOL)

        try:
            self.ui.robot.setPoseFrame(reference_frame)
            self.maintenance.program.setPoseFrame(reference_frame)
            self.ui.robot.setPoseTool(tool)
            self.maintenance.program.setPoseTool(tool)
        except:
            pass

        # Follow path
        for name in targets:
            target = self.ui.RDK.Item(name, ITEM_TYPE_TARGET)
            if not target.Valid():
                print(f"[WARNING] Target '{name}' not found!")
                continue
            if not self.stop:
                print(f"[INFO] Moving to {name}...")
                self.perform_path(target)
                print(f"[INFO] Reached {name}")
            else:
                # Wait for the stop condition to be released
                print("[INFO] Waiting...")
                while ((not self.resume) or (not self.running))  and (self.stop):
                    sleep(2)
                if self.running:
                    raise Exception("Paused assembly process ended.")
                print(f"[INFO] Moving to {name}...")
                self.perform_path(target)
                print(f"[INFO] Reached {name}")

################################ ASSEMBLY AND DISASSEMBLY PATHS ####################################################
    def place_box(self, box_number):
        self.ui.gripper.gripper_open()
        targets = ['Home']

        #go to take a box
        targets.append(f"box{box_number}pos")
        targets.append(f"box{box_number}go")
        self.script_execution(targets)
        targets = []
        self.ui.gripper.gripper_close_box(box_number)
        #self.ui.robot.WaitFinished()

        #self.virtual_gripper_update(f'Attach box{box_number}')
        #self.ui.robot.WaitFinished()
        #sleep(3)

        #take out the box and put it in the assembly station
        targets.append(f"box{box_number}up") 
        targets.append(f"box{box_number}out")
        targets.append('box_to_assembly')
        targets.append(f"box_assembly{box_number}approach")
        #self.script_execution(targets)
        #targets = []
        #self.calculation_method = 3
        targets.append(f"box_assembly{box_number}down")
        self.script_execution(targets)
        targets = []
        self.ui.gripper.gripper_open()
        #self.calculation_method = 0
        #self.virtual_gripper_update(f"Detach")
        targets.append('assembly_out')
        targets.append('assembly_up')
        self.script_execution(targets)
        targets = []
        
    def place_cell(self, cell_number, position_in_box):
        #self.ui.gripper.gripper_open_cell()
        targets = ['Home']
        #go to take a cell
        #targets.append(f"cellsposition")
        targets.append(f"cell{cell_number}pos")
        targets.append(f"cell{cell_number}go")
        self.script_execution(targets)
        targets = []
        self.ui.gripper.gripper_close_cell(cell_number)
        #self.virtual_gripper_update(f"Attach cell{cell_number}")
        #take out the cell and put it in the box
        targets.append(f"cell{cell_number}up")
        targets.append(f"cell{cell_number}out")
        targets.append('Home')
        targets.append(f"place_cell{position_in_box}pos")
        #self.script_execution(targets)
        #targets = []
        #self.calculation_method = 3
        targets.append(f"place_cell{position_in_box}down")
        self.script_execution(targets)
        targets = []
        self.ui.gripper.gripper_open_cell()
        #self.virtual_gripper_update(f"Detach cell {cell_number}")
        #self.calculation_method = 0  
     
        targets.append(f"place_cell{position_in_box}pos")
        self.script_execution(targets)
        targets = []

    def place_full_box(self, shelf_position):
        self.ui.gripper.gripper_open()
        targets = ['Home']
        #go to take the full box
        targets.append('full_box_position') 
        targets.append('grab_full_box')
        self.script_execution(targets)
        self.ui.gripper.gripper_close_box(self.box)
        targets = []
        targets.append('grab_full_box_approach')
        #self.virtual_gripper_update(f"Attach box {box_number}")
        
        #put full box in the shelf
        targets.append('full_box_up')
        targets.append('go_to_shelf') 
        targets.append(f"place{shelf_position}pos")
        targets.append(f"place{shelf_position}approach")
        targets.append(f"place{shelf_position}down")
        self.script_execution(targets)
        targets = []
        self.ui.gripper.gripper_open()
        #self.virtual_gripper_update(f"Detach box {box_number}")
        #change the grip of the lid
        targets.append(f"place{shelf_position}out")
        targets.append('Home')
        self.script_execution(targets)
        targets = []
        
    def place_lid(self, lid_number):   
        self.ui.gripper.gripper_open()
        targets = ['Home']
        
        #go to take a lid
        #targets.append(f"lidsposition")
        targets.append(f"lid{lid_number}pos")
        targets.append(f"lid{lid_number}go")
        self.script_execution(targets)
        self.ui.gripper.gripper_close_lid(lid_number)
        #self.virtual_gripper_update(f"Attach lid{lid_number}")
        targets = []

        #take out the lid and put it in the assembly station
        #self.calculation_method = 3
        targets.append(f"lid{lid_number}up")
        targets.append(f"lid{lid_number}out1")
        targets.append(f"lid{lid_number}out")
        targets.append('lid_to_assembly')
        targets.append('lid_in_assembly')
        targets.append(f"lid_assembly{lid_number}approach")
        #self.script_execution(targets)
        #targets = []
        #self.calculation_method = 0
        targets.append(f"lid_assembly{lid_number}down")
        self.script_execution(targets)
        self.ui.gripper.gripper_open()
        #self.virtual_gripper_update(f"Detach lid {lid_number}")
        targets = []

        #change the grip of the lid
        targets.append('new_grab')
        targets.append('new_grab_approach')
        targets.append('new_grab_down')
        self.script_execution(targets)
        self.ui.gripper.gripper_close_lid(lid_number)
        #self.virtual_gripper_update(f"Attach lid{lid_number}")
        targets = []

        #put lid on the box
        #self.calculation_method = 3
        targets.append('lid_to_shelf')
        #self.script_execution(targets)
        #targets = []
        #self.calculation_method = 0
        targets.append(f"lid{lid_number}shelf")
        #self.script_execution(targets)
        #targets = []
        #self.calculation_method = 3
        targets.append(f"lid{lid_number}down")
        self.script_execution(targets)
        targets = []
        #self.calculation_method = 0
        self.ui.gripper.gripper_open()
        #self.virtual_gripper_update(f"Detach lid {lid_number}")

        #leave lid on the shelf
        targets.append(f"lid{lid_number}shelf")
        targets.append('Home')
        self.script_execution(targets)
        targets = []
    
    def disassemble_box(self, box_number):
        self.ui.gripper.gripper_open()
        targets = ['Home']
        #take box from assembly
        targets.append('assembly_up')
        targets.append('assembly_out')
        targets.append(f"box_assembly{box_number}down")
        self.script_execution(targets)
        targets = []
        self.ui.gripper.gripper_close_box(box_number)
        #self.virtual_gripper_update(f"Attach box{id_box}")

        #take box and put it in the shelf
        targets.append(f"box_assembly{box_number}approach")
        targets.append('box_to_assembly')
        targets.append(f"box{box_number}out")
        targets.append(f"box{box_number}up")
        targets.append(f"box{box_number}go")
        self.script_execution(targets)
        targets = []
        self.ui.gripper.gripper_open()
        #self.virtual_gripper_update(f"Detach")

        #leave box
        targets.append(f"box{box_number}pos")
        targets.append('Home')
        self.script_execution(targets)
        targets = []  

    def disassemble_cell(self, cell_number, position_in_box):
        targets = ['Home']
        targets.append(f"place_cell{position_in_box}pos")
        targets.append(f"place_cell{position_in_box}down")
        self.script_execution(targets)
        targets = []
        self.ui.gripper.gripper_close_cell(cell_number)
        #self.virtual_gripper_update(f"Attach cell{cell_number}")
        
        #take out the cell and put it in the shelf
        targets.append(f"place_cell{position_in_box}pos")
        targets.append('Home')
        targets.append(f"cell{cell_number}out")
        targets.append(f"cell{cell_number}up")
        targets.append(f"cell{cell_number}go")
        self.script_execution(targets)
        self.ui.gripper.gripper_open_cell()
        targets = []
        #self.virtual_gripper_update(f"Detach")

        #leave cell
        targets.append(f"cell{cell_number}pos")
        targets.append('Home')
        self.script_execution(targets)
        targets = []

    def disassemble_full_box(self, shelf_position):
        self.ui.gripper.gripper_open()
        targets = []
        #take box from final shelf
        targets.append(f"place{shelf_position}out")
        targets.append(f"place{shelf_position}down")
        self.script_execution(targets)
        targets = []
        self.ui.gripper.gripper_close_box(shelf_position)
        #self.virtual_gripper_update(f"Attach box{id_box}")
        
        #put full box in the shelf
        targets.append(f"place{shelf_position}approach")
        targets.append(f"place{shelf_position}pos")
        targets.append('go_to_shelf')
        targets.append('full_box_up')
        targets.append('grab_full_box_approach')
        targets.append('grab_full_box')
        self.script_execution(targets)
        targets = []
        self.ui.gripper.gripper_open()
        #self.virtual_gripper_update(f"Detach")
        
        #leave full box in assembly
        targets.append('full_box_position')
        targets.append('Home')
        self.script_execution(targets)
        self.ui.gripper.gripper_open_cell()

    def disassemble_lid(self, lid_number):
        self.ui.gripper.gripper_open()

        #take lid from the shelf
        targets = ['Home']
        targets.append(f"lid{lid_number}shelf")
        targets.append(f"lid{lid_number}down")
        self.script_execution(targets)
        targets = []
        self.ui.gripper.gripper_close_lid(lid_number)
        #self.virtual_gripper_update(f"Attach lid{id_box}")

        #put lid on the box
        targets.append(f"lid{lid_number}shelf")
        targets.append('lid_to_shelf')
        targets.append('new_grab')
        targets.append('new_grab_approach')
        targets.append('new_grab_down')
        self.script_execution(targets)
        targets = []
        self.ui.gripper.gripper_open()
        #self.virtual_gripper_update(f"Detach")
        
        #change the grip of the lid
        targets.append('new_grab')
        targets.append(f"lid_assembly{lid_number}down")
        self.script_execution(targets)
        targets = []
        self.ui.gripper.gripper_close_lid(lid_number)
        #self.virtual_gripper_update(f"Attach lid{id_box}")

        #take out the from assembly and put it in the shelf
        targets.append(f"lid_assembly{lid_number}approach")
        targets.append('lid_in_assembly')
        targets.append('lid_to_assembly')
        targets.append(f"lid{lid_number}out")
        targets.append(f"lid{lid_number}out1")
        targets.append(f"lid{lid_number}up")
        targets.append(f"lid{lid_number}go")
        self.script_execution(targets)
        targets = []
        self.ui.gripper.gripper_open()
        #self.virtual_gripper_update(f"Detach")
        
        #leave lid
        targets.append(f"lid{lid_number}pos")
        targets.append('Home')
        self.script_execution(targets)
        targets = []
     
################################# PATH CALCULATIONS ################################################################
    def perform_path(self, target, obstacle_pos=None, obstacle_radius=0.1, N=20, avoid_obstacles=False, min_time=False, low_energy=False):
        '''Selection of the path calculation method.'''
        # 0: MoveJ
        # 1: Joint path
        # 2: CasADi path
        
        if self.calculation_method == 0:
            self.moveJ_path(target)
        elif self.calculation_method == 1:
            self.joint_path(target)
        elif self.calculation_method == 2:
            self.casadi_path(target, obstacle_pos, obstacle_radius, N, avoid_obstacles, min_time, low_energy)

#   0 
    def moveJ_path(self, target):       
        """Executes a joint path on the robot."""
        # Add the momentum to the maintenance program
        self.maintenance.program.MoveJ(target)
        # Perform the movement on the robot
        self.ui.robot.MoveJ(target)

#   1
    def joint_path(self, target, num_steps=20):
        """Executes a joint path on the robot."""
        #obtain current position
        current_joints = self.ui.robot.Joints().tolist()
        target = target.Joints().tolist()
        trip = []
        for i in range(1,num_steps +1):
            #t= i / num_steps
            #sigm_factor = 1 / (1 + np.exp(-10 * (t - 0.5)))
            #interp = [s + (e-s)*sigm_factor for s,e in zip(current_joints,target)] 
            interp = [s + (e-s)*i/num_steps for s,e in zip(current_joints,target)]
            trip.append(interp) 
        
        for name in trip:
            self.maintenance.program.MoveL(name)
            self.ui.robot.MoveL(name)
            #sleep(0.01)

#   2
    def casadi_path(self, target_item, obstacle_pos=None, obstacle_radius=0.1, N=20, avoid_obstacles=False, min_time=False, low_energy=False):
        """
        Trajectory optimization using CasADi with options for:
        - Obstacle avoidance
        - Time minimization
        - Energy minimization (smooth motion)
        Trajectory is computed in Cartesian XYZ space.
        """

        # Variables and optimization setup
        # Time step variable if optimizing time
        dt = ca.MX.sym('dt') if min_time else 0.05
        # Trajectory of XYZ points
        X = ca.MX.sym('X', 3, N)  

        # Cost function
        cost = 0
        if low_energy:
            # Minimize acceleration for smooth movement
            for k in range(1, N-1):
                acc = X[:, k+1] - 2 * X[:, k] + X[:, k-1]
                cost += ca.sumsqr(acc)
        if min_time:
            # Minimize total time
            cost += dt * N

        # Constraints Initialization
        g = []  # List of constraints
        lb = []  # Lower bounds
        ub = []  # Upper bounds

        # Extract start and end positions (XYZ only)
        current_xyz = Pose_2_TxyzRxyz(self.ui.robot.Pose())[0:3]
        target_xyz = Pose_2_TxyzRxyz(target_item.Pose())[0:3]

        # Start position is the current pose
        g += [X[:, 0] - ca.DM(current_xyz)]
        lb += [0.0] * 3
        ub += [0.0] * 3

        # End position is the target pose
        g += [X[:, -1] - ca.DM(target_xyz)]
        lb += [0.0] * 3
        ub += [0.0] * 3

        # Limit max translation per step (for smoothness)
        max_step = 30.0  # mm per step
        for k in range(1, N):
            step = X[:, k] - X[:, k-1]
            g += [step]
            lb += [-max_step] * 3
            ub += [max_step] * 3

        # Add obstacle avoidance constraint (distance^2 > radius^2)
        if avoid_obstacles and obstacle_pos:
            for k in range(N):
                obstacle_xyz = ca.DM([obstacle_pos[0, 3], obstacle_pos[1, 3], obstacle_pos[2, 3]])
                dist_sq = ca.sumsqr(X[:, k] - obstacle_xyz)
                g += [dist_sq]
                lb += [obstacle_radius**2]
                ub += [1e3]  # No upper bound

        # Add time constraints
        if min_time:
            g += [dt]
            lb += [1e-3]
            ub += [0.5]

        # Build NLP problem
        opt_vars = ca.vertcat(ca.reshape(X, -1, 1), dt) if min_time else ca.reshape(X, -1, 1)
        nlp = {'x': opt_vars, 'f': cost, 'g': ca.vertcat(*g)}

        solver = ca.nlpsol('solver', 'ipopt', nlp, {
            "ipopt.print_level": 0,
            "print_time": 0
        })

        # Initial Guess
        guess = np.linspace(current_xyz, target_xyz, N).flatten().tolist()
        if min_time:
            guess += [0.1]  # Initial dt guess

        # Solve the optimization problem
        sol = solver(x0=guess, lbg=lb, ubg=ub)
        result = np.array(sol['x'].full()).flatten()

        # Extract trajectory from solution
        traj = result[:-1].reshape((N, 3)).T if min_time else result.reshape((N, 3)).T
        dt_val = float(result[-1]) if min_time else dt

        # Keep target orientation constant
        target_pose = target_item.Pose()

        # EXECUTE TRAJECTORY
        for i in range(traj.shape[1]):
            x, y, z = traj[:, i]

            # Copy target pose and update position
            pose = Mat(target_pose)
            pose[0:3, 3] = [x, y, z]  # Update position only

            joints = self.ui.robot.SolveIK(pose)
            if joints is not None:
                # Add the movement to the maintenance program
                self.maintenance.program.MoveJ(pose)
                # Perform the movement on the robot
                self.ui.robot.MoveJ(pose, blocking = False)
                if not isinstance(dt_val, ca.MX):
                    sleep(float(dt_val))  # Add delay if not blocking
            else:
                print(f"[WARNING] Skipping unreachable point at step {i}")

        print("[INFO] CasADi trajectory executed.")

###################################################################################################################
###################################### GRIPPER  ###################################################################
###################################################################################################################
class Gripper():
    def __init__(self, parent):
        ''' Initialize the gripper. '''
        self.parent = parent
        self.RDK = parent.RDK
        self.robot = parent.robot
        self.gripper_open_command = None
        self.gripper_open_cell_command = None
        self.gripper_close_box_command = None
        self.gripper_close_cell_command = None
        self.gripper_close_lid_command = None
        self.moving_gripper = False
        self.socket = None
        self.virtual_gripper = self.RDK.Item('Gripper', ITEM_TYPE_ROBOT)
        self.gripper_setup()

    def gripper_setup(self):
        '''Sets up the gripper by reading the gripper commands from files,
        if the files are not found, it creates the programs from RoboDK.
        and finally connects to the socket.'''
        setup = False
        setup2 = False
        while not setup:
            try:
                # Read gripper commands from files
                with open(GRIPPER_OPEN_PATH, 'r') as f:
                    self.gripper_open_command = f.read()

                with open(GRIPPER_OPEN_CELL_PATH, 'r') as f:
                    self.gripper_open_cell_command = f.read()

                with open(GRIPPER_CLOSE_BOX_PATH, 'r') as f:
                    self.gripper_close_box_command = f.read()

                with open(GRIPPER_CLOSE_CELL_PATH, 'r') as f:
                    self.gripper_close_cell_command = f.read()
                
                with open(GRIPPER_CLOSE_LID_PATH, 'r') as f:
                    self.gripper_close_lid_command = f.read()
                
                setup = True
            except:
                # If files are not found, create the programs in RoboDK
                program = self.RDK.Item('gripper_open', ITEM_TYPE_PROGRAM)
                program.MakeProgram(GRIPPER_FOLDER_PATH)
                program = self.RDK.Item('gripper_open_cell', ITEM_TYPE_PROGRAM)
                program.MakeProgram(GRIPPER_FOLDER_PATH)
                program = self.RDK.Item('gripper_close_box', ITEM_TYPE_PROGRAM)
                program.MakeProgram(GRIPPER_FOLDER_PATH)
                program = self.RDK.Item('gripper_close_cell', ITEM_TYPE_PROGRAM)
                program.MakeProgram(GRIPPER_FOLDER_PATH)
                program = self.RDK.Item('gripper_close_lid', ITEM_TYPE_PROGRAM)
                program.MakeProgram(GRIPPER_FOLDER_PATH)
         # Connect to the gripper socket
        while not setup2:
            try:
                print(f"[INFO] Attempting to connect to the gripper at {IP}:{PORT}...")
                # Setup the socket connection for the gripper
                self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.socket.connect((IP, PORT))
                print(f"[INFO] Connected to gripper at {IP}:{PORT} successfully.")
                setup2 = True
            except (ConnectionRefusedError, TimeoutError, OSError) as e:
                print(f"[WARNING] Connection attempt failed: {e}")
                if self.parent.ask_retry_connection('gripper') == 2:
                    raise SystemExit("[ERROR] Connection attempt aborted.")

    def gripper_open(self):
        '''Moves the virtual gripper to the open position,
        sends program to detach the virtual gripper from the robot
        and sends the command to open the gripper.'''
        if self.moving_gripper:
            print("[ERROR] Gripper is busy!")
            return
        self.moving_gripper = True
        # Set the run mode to simulate, to avoid executing the program on the real robot
        self.RDK.setRunMode(RUNMODE_SIMULATE)
        # Move the virtual gripper to the open position
        self.virtual_gripper.MoveJ([70])
        # Run the program to detach the gripper in the simulation
        prog = self.RDK.Item('Detach', ITEM_TYPE_PROGRAM)
        if prog.Valid():
            prog.RunCode()
        # Set the run mode back to run robot
        self.RDK.setRunMode(RUNMODE_RUN_ROBOT)
        print("[INFO] Opening Gripper")
        # Send the command to open the gripper
        
        self.socket.send(self.gripper_open_command.encode('utf-8'))
        # Wait a second before reconnecting to ensure the command is sent correctly
        sleep(1)
        # Reconnect to the robot
        success = self.robot.Connect(IP)
        print("[INFO] Gripper Open")
        while not success:
            success = self.robot.Connect(IP)
            self.moving_gripper = True
        self.moving_gripper = False

    def gripper_open_cell(self):
        '''Moves the virtual gripper to the open position after cell,
        sends program to detach the virtual cell from the gripper
        and sends the command to open the gripper.'''
        if self.moving_gripper:
            print("[ERROR] Gripper is busy!")
            return
        self.moving_gripper = True
        # Set the run mode to simulate, to avoid executing the program on the real robot
        self.RDK.setRunMode(RUNMODE_SIMULATE)
        # Move the virtual gripper to the open position after cell
        self.virtual_gripper.MoveJ([38])
        # Run the program to detach the cell in the simulation
        prog = self.RDK.Item('Detach', ITEM_TYPE_PROGRAM)
        if prog.Valid():
            prog.RunCode()
            print("[INFO] Virtual cell detached")
        # Set the run mode back to run robot
        self.RDK.setRunMode(RUNMODE_RUN_ROBOT)
        print("[INFO] Opening Gripper")
        # Send the command to open the gripper
        self.socket.send(self.gripper_open_cell_command.encode('utf-8'))
        # Wait a second before reconnecting to ensure the command is sent correctly
        sleep(1)
        # Reconnect to the robot
        success = self.robot.Connect(IP)
        print("[INFO] Gripper Open")
        while not success:
            success = self.robot.Connect(IP)
            self.moving_gripper = True
        self.moving_gripper = False

    def gripper_close_box(self, box_num):
        '''Moves the virtual gripper to the box position,
        sends program to attach the virtual box to the gripper
        and sends the command to close the gripper.'''
        if self.moving_gripper:
            print("[ERROR] Gripper is busy!")
            return
        self.moving_gripper = True
        # Set the run mode to simulate, to avoid executing the program on the real robot
        self.RDK.setRunMode(RUNMODE_SIMULATE)
        # Move the virtual gripper to the box position
        self.virtual_gripper.MoveJ([63])
        # Run the program to attach the box in the simulation
        prog = self.RDK.Item(f'Attach box{box_num}', ITEM_TYPE_PROGRAM)
        if prog.Valid():
            prog.RunCode()
            print("[INFO] Virtual box attached")
        # Set the run mode back to run robot
        # Keep track of the assembled box and its cells
        if self.parent.scripts.assembled[box_num-1] != 0:
            for i in self.parent.scripts.assembled[box_num-1]:
                prog = self.RDK.Item(f'Attach cell{i}', ITEM_TYPE_PROGRAM)
                if prog.Valid():
                    prog.RunCode()
                    print("[INFO] Virtual cell attached")
        self.RDK.setRunMode(RUNMODE_RUN_ROBOT)
        print("[INFO] Gripping Box")
        # Send the command to close the gripper
        self.socket.send(self.gripper_close_box_command.encode('utf-8'))
        # Wait a second before reconnecting to ensure the command is sent correctly
        sleep(1)
        # Reconnect to the robot
        success = self.robot.Connect(IP)
        print("[INFO] Box Gripped")
        while not success:
            success = self.robot.Connect(IP)
            self.moving_gripper = True
        self.moving_gripper = False

    def gripper_close_cell(self, cell_num):
        '''Moves the virtual gripper to the cell position,
        sends program to attach the virtual cell to the gripper
        and sends the command to close the gripper.'''
        if self.moving_gripper:
            print("[ERROR] Gripper is busy!")
            return
        self.moving_gripper = True
        # Set the run mode to simulate, to avoid executing the program on the real robot
        self.RDK.setRunMode(RUNMODE_SIMULATE)
        # Move the virtual gripper to the cell position
        self.virtual_gripper.MoveJ([25])
        # Run the program to attach the cell in the simulation
        prog = self.RDK.Item(f'Attach cell{cell_num}', ITEM_TYPE_PROGRAM)
        if prog.Valid():
            prog.RunCode()
            print("[INFO] Virtual cell attached")
        # Set the run mode back to run robot
        self.RDK.setRunMode(RUNMODE_RUN_ROBOT)
        print("[INFO] Gripping cell")
        # Send the command to close the gripper
        self.socket.send(self.gripper_close_cell_command.encode('utf-8'))
        # Wait a second before reconnecting to ensure the command is sent correctly
        sleep(1)
        # Reconnect to the robot
        success = self.robot.Connect(IP)
        print("[INFO] cell Gripped")
        while not success:
            success = self.robot.Connect(IP)
            self.moving_gripper = True
        self.moving_gripper = False

    def gripper_close_lid(self, lid_num):
        '''Moves the virtual gripper to the lid position,
        sends program to attach the virtual lid to the gripper
        and sends the command to close the gripper.'''
        if self.moving_gripper:
            print("[ERROR] Gripper is busy!")
            return
        self.moving_gripper = True
        # Set the run mode to simulate, to avoid executing the program on the real robot
        self.RDK.setRunMode(RUNMODE_SIMULATE)
        # Move the virtual gripper to the lid position
        self.virtual_gripper.MoveJ([17])
        # Run the program to attach the lid in the simulation
        prog = self.RDK.Item(f'Attach lid{lid_num}', ITEM_TYPE_PROGRAM)
        if prog.Valid():
            prog.RunCode()
            print("[INFO] Virtual lid attached")
        # Set the run mode back to run robot
        self.RDK.setRunMode(RUNMODE_RUN_ROBOT)
        print("[INFO] Gripping Lid")
        # Send the command to close the gripper
        self.socket.send(self.gripper_close_lid_command.encode('utf-8'))
        # Wait a second before reconnecting to ensure the command is sent correctly
        sleep(1)
        # Reconnect to the robot
        success = self.robot.Connect(IP)
        print("[INFO] Lid Gripped")
        while not success:
            success = self.robot.Connect(IP)
            self.moving_gripper = True
        self.moving_gripper = False

###################################################################################################################
#################### ROBOT CONTROL AND UI #########################################################################
###################################################################################################################
class RobotControlUI(QWidget):
    def __init__(self, robot, rdk):
        ''' Initialize the Robot Control UI. '''
        super().__init__()

        self.robot = robot
        self.RDK = rdk  # Store RoboDK instance

        self.manual_mode = True  # Manual mode starts as active

        self.movement_timer = QTimer()
        self.movement_timer.timeout.connect(self.perform_movement)

        self.rotation_timer = QTimer()
        self.rotation_timer.timeout.connect(self.perform_rotation)

        self.movement_direction = None
        self.rotation_direction = None

        self.manual_speed = USER_MAN_SPEED

        # Ensure that the physical cell and the virtual cell are in the correct, same state
        ready = self.ask_reset()
        if ready == 2:
            raise SystemExit("[ERROR] Cell is not in the correct state!")
        
        self.gripper = Gripper(self)  # Initialize the gripper

        # Initialize the robot and connect to it
        self.initUI()
        
        # Reset the virtual model
        prog = self.RDK.Item(f'Replace All', ITEM_TYPE_PROGRAM)
        if prog.Valid():
            print("[INFO] Resetting virtual model...")
            prog.RunCode()
        
        self.RDK.setSimulationSpeed(1) # Set simulation speed to ensure the simulation runs at real time

        # Ensure the virtual robot is in the home position
        home = self.RDK.Item(HOME, ITEM_TYPE_TARGET)
        if home.Valid():
            self.robot.MoveJ(home, blocking=False)

        self.connect_to_robot() # Connect to the robot
        
        self.scripts = Path(self)

################################# UI SETUP #######################################################################
    def initUI(self):
        ''' Create and initialize the UI components and layout. '''
        self.setWindowTitle('UR10e Robot Control')
        self.setGeometry(100, 100, 450, 700)

        layout = QGridLayout()

        # Mode Selection Section
        self.btn_manual_mode = QPushButton("Manual Mode")
        self.btn_manual_mode.setStyleSheet("background-color: #4CAF50; color: white; font-size: 14px; padding: 5px;")
        self.btn_manual_mode.clicked.connect(self.activate_manual_mode)
        layout.addWidget(self.btn_manual_mode, 0, 0, 1, 2)

        self.btn_auto_mode = QPushButton("Automatic Mode")
        self.btn_auto_mode.setStyleSheet("background-color: #D32F2F; color: white; font-size: 14px; padding: 5px;")
        self.btn_auto_mode.clicked.connect(self.activate_auto_mode)
        layout.addWidget(self.btn_auto_mode, 0, 2, 1, 2)

        # Manual Mode Controls
        self.label_move = QLabel('TCP Position Control')
        layout.addWidget(self.label_move, 1, 1, 1, 3)

        self.manual_buttons = []
        self.auto_buttons = []
        self.move_buttons = []
        self.rotate_buttons = []
        self.gripper_buttons = []
        self.maintenance_buttons = []

        self.create_movement_button(layout, '↑ Y+', 2, 1, 0, 1, 0)  # Move +Y
        self.create_movement_button(layout, '← X-', 3, 0, -1, 0, 0)  # Move -X
        self.create_movement_button(layout, '→ X+', 3, 2, 1, 0, 0)  # Move +X
        self.create_movement_button(layout, '↓ Y-', 4, 1, 0, -1, 0)  # Move -Y
        self.create_movement_button(layout, 'Up Z+', 2, 3, 0, 0, 1)  # Move +Z
        self.create_movement_button(layout, 'Down Z-', 4, 3, 0, 0, -1)  # Move -Z

        # Spacer
        layout.addWidget(QLabel(''), 5, 1)

        # Rotation Controls
        self.label_rotate = QLabel('TCP Orientation Control')
        layout.addWidget(self.label_rotate, 6, 1, 1, 3)

        self.create_rotation_button(layout, '↻ X+', 7, 1, 1, 0, 0)  # Rotate +X
        self.create_rotation_button(layout, '↺ X-', 9, 1, -1, 0, 0)  # Rotate -X
        self.create_rotation_button(layout, '↻ Y+', 8, 0, 0, 1, 0)  # Rotate +Y
        self.create_rotation_button(layout, '↺ Y-', 8, 2, 0, -1, 0)  # Rotate -Y
        self.create_rotation_button(layout, '↻ Z+', 7, 3, 0, 0, 1)  # Rotate +Z
        self.create_rotation_button(layout, '↺ Z-', 9, 3, 0, 0, -1)  # Rotate -Z

        # Spacer
        layout.addWidget(QLabel(''), 10, 1)

        # Speed control label and slider
        self.label_speed = QLabel(f"Manual Speed: {self.manual_speed}")
        self.slider_speed = QSlider(Qt.Orientation.Horizontal)
        self.slider_speed.setRange(1, 100)
        self.slider_speed.setValue(self.manual_speed)
        self.slider_speed.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.slider_speed.setTickInterval(10)
        self.slider_speed.valueChanged.connect(self.update_speed)
        layout.addWidget(self.label_speed, 10, 2, 1, 2)
        layout.addWidget(self.slider_speed, 11, 2, 1, 2)

        # Move to Home Button
        self.btn_home = QPushButton("Move to Home")
        self.btn_home.setStyleSheet("font-size: 14px; padding: 10px;")
        self.btn_home.clicked.connect(self.move_to_home)
        layout.addWidget(self.btn_home, 12, 1, 1, 3)
        self.manual_buttons.append(self.btn_home)

        # Automatic Mode Controls
        self.btn_execute_program = QPushButton("Execute Program")
        self.btn_execute_program.setStyleSheet("font-size: 14px; padding: 10px; background-color: #1976D2; color: white;")
        self.btn_execute_program.clicked.connect(self.execute_robot_program)
        layout.addWidget(self.btn_execute_program, 13, 1, 1, 3)
        self.auto_buttons.append(self.btn_execute_program)

        self.btn_pause_program = QPushButton("Pause Program")
        self.btn_pause_program.setStyleSheet("font-size: 14px; padding: 10px; background-color: #FFC107; color: black;")
        self.btn_pause_program.clicked.connect(self.pause_robot_program)
        layout.addWidget(self.btn_pause_program, 15, 1, 1, 3)
        self.auto_buttons.append(self.btn_pause_program)

        self.btn_resume_program = QPushButton("Resume Program")
        self.btn_resume_program.setStyleSheet("font-size: 14px; padding: 10px; background-color: #4CAF50; color: white;")
        self.btn_resume_program.clicked.connect(self.resume_robot_program)
        layout.addWidget(self.btn_resume_program, 14, 1, 1, 3)
        self.auto_buttons.append(self.btn_resume_program)

        self.btn_emergency_stop = QPushButton("STOP Program")
        self.btn_emergency_stop.setStyleSheet("font-size: 14px; padding: 10px; background-color: #D32F2F; color: white;")
        self.btn_emergency_stop.clicked.connect(self.emergency_stop)
        layout.addWidget(self.btn_emergency_stop, 16, 1, 1, 3)
        ###### Gripper control section #######

        # Open gripper button
        self.btn_gripper_open = QPushButton("Open Gripper")
        self.btn_gripper_open.setStyleSheet("background-color: #009688; color: white; font-size: 12px; padding: 5px;")
        self.btn_gripper_open.clicked.connect(self.gripper.gripper_open)
        layout.addWidget(self.btn_gripper_open, 12, 0, 1, 1)
        self.gripper_buttons.append(self.btn_gripper_open)
        
        # Open gripper for cell button
        self.btn_gripper_open_cell = QPushButton("Open Cell Gripper")
        self.btn_gripper_open_cell.setStyleSheet("background-color: #009688; color: white; font-size: 12px; padding: 5px;")
        self.btn_gripper_open_cell.clicked.connect(self.gripper.gripper_open_cell)
        layout.addWidget(self.btn_gripper_open_cell, 13, 0, 1, 1)
        self.gripper_buttons.append(self.btn_gripper_open_cell)

        # Box gripper button
        self.btn_gripper_close_box = QPushButton("Grip Box")
        self.btn_gripper_close_box.setStyleSheet("background-color: #009688; color: white; font-size: 12px; padding: 5px;")
        self.btn_gripper_close_box.clicked.connect(self.gripper.gripper_close_box)
        layout.addWidget(self.btn_gripper_close_box, 14, 0, 1, 1)
        self.gripper_buttons.append(self.btn_gripper_close_box)
        
        # cell gripper button
        self.btn_gripper_close_cell = QPushButton("Grip Cell")
        self.btn_gripper_close_cell.setStyleSheet("background-color: #009688; color: white; font-size: 12px; padding: 5px;")
        self.btn_gripper_close_cell.clicked.connect(self.gripper.gripper_close_cell)
        layout.addWidget(self.btn_gripper_close_cell, 15, 0, 1, 1)
        self.gripper_buttons.append(self.btn_gripper_close_cell)

        # Lid gripper button
        self.btn_gripper_close_lid = QPushButton("Grip Lid")
        self.btn_gripper_close_lid.setStyleSheet("background-color: #009688; color: white; font-size: 12px; padding: 5px;")
        self.btn_gripper_close_lid.clicked.connect(self.gripper.gripper_close_lid)
        layout.addWidget(self.btn_gripper_close_lid, 16, 0, 1, 1)
        self.gripper_buttons.append(self.btn_gripper_close_lid)

        # Show maintenance charts
        self.btn_maintenance = QPushButton("Charts")
        self.btn_maintenance.setStyleSheet("background-color: #009688; color: white; font-size: 12px; padding: 5px;")
        self.btn_maintenance.clicked.connect(self.start_maintenance)
        layout.addWidget(self.btn_maintenance, 10, 0, 1, 1)
        self.maintenance_buttons.append(self.btn_maintenance)

        # Maintenance script clear
        self.btn_maintenanceclr = QPushButton("Clr Charts")
        self.btn_maintenanceclr.setStyleSheet("background-color: #009688; color: white; font-size: 12px; padding: 5px;")
        self.btn_maintenanceclr.clicked.connect(self.clear_maintenance)
        layout.addWidget(self.btn_maintenanceclr, 11, 0, 1, 1)
        self.maintenance_buttons.append(self.btn_maintenanceclr)

        self.setLayout(layout)
        self.update_button_states()  # Initialize button states based on mode

################################# BUTTON FUNCTIONS ###############################################################
    def update_speed(self, value):
        """Update the current robot speed from slider."""
        self.manual_speed = value
        self.label_speed.setText(f"Speed: {value}")
    
    def start_maintenance(self):
        try:
            # Start the maintenance program
            self.scripts.maintenance.extract_data()
            self.scripts.maintenance.read_csv()
            self.scripts.maintenance.robot_joints_range()
            self.scripts.maintenance.robot_joints_usage()
            self.scripts.maintenance.robot_joints_over_time()
            self.scripts.maintenance.robot_joints_velocity()
            self.scripts.maintenance.robot_joints_acceleration()
            self.scripts.maintenance.historical_data()
            self.scripts.maintenance.historical_analysis()
        except:
            pass

    def clear_maintenance(self):
        try:
            # Delete the maintenance program and re-add it
            self.scripts.maintenance.program.Delete()
            self.scripts.maintenance.program = self.RDK.AddProgram(SCRIPT)
        except:
            pass
            
    def create_movement_button(self, layout, text, row, col, dx, dy, dz):
        """Helper function to create movement buttons."""
        btn = QPushButton(text)
        btn.setStyleSheet("font-size: 16px; padding: 10px;")
        # Connect button press to start movement
        btn.pressed.connect(lambda: self.start_movement(dx, dy, dz))
        btn.released.connect(self.stop_movement)
        layout.addWidget(btn, row, col)
        # Store the button in the list for later access
        self.move_buttons.append(btn)

    def create_rotation_button(self, layout, text, row, col, rx, ry, rz):
        """Helper function to create rotation buttons."""
        btn = QPushButton(text)
        btn.setStyleSheet("font-size: 16px; padding: 10px;")
        # Connect button press to start rotation
        btn.pressed.connect(lambda: self.start_rotation(rx, ry, rz))
        btn.released.connect(self.stop_rotation)
        layout.addWidget(btn, row, col)
        # Store the button in the list for later access
        self.rotate_buttons.append(btn)

    def activate_manual_mode(self):
        """Activates manual mode and disables automatic mode."""
        if self.robot.Busy() or self.scripts.running or self.scripts.resume:
            print("[ERROR] Cannot change to manual mode, robot is busy or program is running!")
            return
        self.manual_mode = True
        self.update_button_states()

    def activate_auto_mode(self):
        """Activates automatic mode and disables manual mode."""
        self.manual_mode = False
        self.update_button_states()

    def update_button_states(self):
        """Enables/disables buttons based on the active mode."""
        for btn in self.move_buttons + self.rotate_buttons + self.manual_buttons:
            btn.setEnabled(self.manual_mode)
            if not self.manual_mode:
                btn.setStyleSheet("background-color: #A5A5A5; color: black; font-size: 16px; padding: 10px;")
            else:
                btn.setStyleSheet("background-color: #7AA7D6; color: white; font-size: 16px; padding: 10px;")

        self.slider_speed.setEnabled(self.manual_mode)
        for btn in self.auto_buttons:
            btn.setEnabled(not self.manual_mode)
        # Update mode button colors
        if self.manual_mode:
            self.btn_manual_mode.setStyleSheet("background-color: #4CAF50; color: white;")
            self.btn_auto_mode.setStyleSheet("background-color: #D32F2F; color: white;")
            self.btn_execute_program.setStyleSheet("background-color: #A5A5A5; color: black; font-size: 14px; padding: 10px;")
            self.btn_pause_program.setStyleSheet("background-color: #A5A5A5; color: black; font-size: 14px; padding: 10px;")
            self.btn_resume_program.setStyleSheet("background-color: #A5A5A5; color: black; font-size: 14px; padding: 10px;")
        else:
            self.btn_manual_mode.setStyleSheet("background-color: #D32F2F; color: white;")
            self.btn_auto_mode.setStyleSheet("background-color: #4CAF50; color: white;")
            self.btn_execute_program.setStyleSheet("background-color: #7AA7D6; color: white; font-size: 14px; padding: 10px;")
            self.btn_pause_program.setStyleSheet("background-color: #ECAB50; color: white; font-size: 14px; padding: 10px;")
            self.btn_resume_program.setStyleSheet("background-color: #7AA7D6; color: white; font-size: 14px; padding: 10px;")
        for btn in self.gripper_buttons:
            btn.setEnabled(self.manual_mode)
            if self.manual_mode:
                btn.setStyleSheet("background-color: #009688; color: white; font-size: 12px;")
            else:
                btn.setStyleSheet("background-color: #A5A5A5; color: black; font-size: 12px;")
        for btn in self.maintenance_buttons:
            btn.setEnabled(not self.manual_mode)
            if not self.manual_mode:
                btn.setStyleSheet("background-color: #009688; color: white; font-size: 12px;")
            else:
                btn.setStyleSheet("background-color: #A5A5A5; color: black; font-size: 12px;")
   
################################# ROBOT CONNECTION AND MOVEMENT ###################################################
    def connect_to_robot(self):
        """Attempts to connect to the UR10e robot if not already connected."""
        if self.robot.ConnectedState() == 1:
            print("[INFO] Robot is already connected.")
        else:
            # Attempt to connect to the robot
            print(f"[INFO] Attempting to connect to the robot at {IP}...")
            success = self.robot.Connect(IP)  # Connect to the robot
            
            #self.robot.setJoints(real_joints)
            if success:
                print("[INFO] Successfully connected to the robot.")
            else:
                print("[ERROR] Failed to connect to the robot. Check network and robot settings.")
                if self.ask_retry_connection('robot') == 1:
                    try:
                        self.connect_to_robot()
                    except:
                        raise SystemExit("[ERROR] Connection attempt aborted.")
                else:
                    raise SystemExit("[ERROR] Connection attempt aborted.")
                
    def move_to_home(self):
        """Moves the robot to the predefined home position."""
        if not self.manual_mode:
            return
        elif self.robot.Busy():
            print("[ERROR] Robot is busy!")
            return
        target = self.RDK.Item(HOME, ITEM_TYPE_TARGET)
        if not target.Valid():
            print("[ERROR] Target not found in RoboDK!")
            return
        print("[INFO] Moving robot to home position...")
        self.robot.setSpeed(100)
        # Move to home position without blocking the UI thread
        
        threading.Thread(target=self.safe_movement, args=(2,target)).start()
        
        print("[INFO] Robot has reached home.")

    def start_movement(self, dx, dy, dz):
        """Starts continuous movement when a button is pressed."""
        if not self.manual_mode:
            return
        elif self.robot.Busy():
            print("[ERROR] Robot is busy!")
            return
        self.robot.setSpeed(self.manual_speed)
        self.movement_direction = (dx, dy, dz)
        self.movement_timer.start(30)

    def stop_movement(self):
        """Stops movement when the button is released."""
        self.movement_timer.stop()
        self.movement_direction = None

    def start_rotation(self, rx, ry, rz):
        """Starts continuous rotation when a button is pressed."""
        if not self.manual_mode:
            return
        elif self.robot.Busy():
            print("[ERROR] Robot is busy!")
            return
        self.robot.setSpeed(self.manual_speed)
        self.rotation_direction = (rx, ry, rz)
        self.rotation_timer.start(30)

    def stop_rotation(self):
        """Stops rotation when the button is released."""
        self.rotation_timer.stop()
        self.rotation_direction = None
    
    def perform_movement(self):
        """Executes movement every timer tick while button is held."""
        try:
            if self.movement_direction and self.manual_mode:
                # Apply translation matrix to the current pose
                dx, dy, dz = self.movement_direction
                pose = self.robot.Pose()
                pose = pose * transl(dx * MSTEP, dy * MSTEP, dz * MSTEP)  # Apply translation matrix
                # Move the robot to the new pose without blocking on the UI thread
                threading.Thread(target=self.safe_movement, args=(1,pose)).start()
        except:
            pass

    def perform_rotation(self):
        """Executes rotation every timer tick while button is held."""
        try:
            if self.rotation_direction and self.manual_mode:
                # Apply rotation matrix to the current pose
                rx, ry, rz = self.rotation_direction
                pose = self.robot.Pose()
                pose = pose * rotx(rx * DSTEP * 3.1416 / 180) * roty(ry * DSTEP * 3.1416 / 180) * rotz(rz * DSTEP * 3.1416 / 180)  # Apply rotation
                # Move the robot to the new pose without blocking on the UI thread
                threading.Thread(target=self.safe_movement, args=(2,pose)).start()
        except:
            pass

    def safe_movement(self, method, target):
        try:
            if method == 1:
                self.robot.MoveL(target, blocking=False)
            elif method == 2:
                self.robot.MoveJ(target, blocking=False)
        except:
            print(f"[ERROR] There was a problem executing the movement")

################################## ROBOT PROGRAM EXECUTION ########################################################
    def execute_robot_program(self):
        """Executes the predefined robot program once."""
        if self.manual_mode:
            return  # Only runs in automatic mode
        elif self.robot.Busy() or self.scripts.running or self.scripts.resume:
            # Check if the robot is busy or if a script is already running
            print("[ERROR] Robot is busy!")
            return
        print("[INFO] Executing SCRIPT...")
        self.scripts.running = True
        self.scripts.stop = False
        self.scripts.resume = False

        # Ask for mode selection
        self.scripts.mode = self.ask_mode()
        if self.scripts.mode == 1:      # Automatic assembly
            self.scripts.disassembly = False
        elif self.scripts.mode == 2:    # Automatic disassembly
            self.scripts.disassembly = True
            # Select box for automatic disassembly
            self.scripts.box = self.ask_selection()
        elif self.scripts.mode == 3:    # Manual assembly
            self.scripts.disassembly = False
            # Select box and cells for manual assembly
            self.scripts.box = self.ask_selection()
            self.scripts.cells = []
            for i in range(1, 5):
                self.scripts.cells.append(self.ask_cell_selection(self.scripts.cells)) 
        else:
            print("[ERROR] Invalid mode selected.")
            return
        # Start the script in a separate thread
        try:
            threading.Thread(target=self.scripts.main_script, args=()).start()
        except:
            print("[ERROR] There was a problem executing the program, please try again.")
        print("[INFO] Program sent for execution.")

    def pause_robot_program(self):
        """Stops the execution of the predefined robot program."""
        if self.manual_mode or self.scripts.stop  or ((not self.scripts.running) and (not self.scripts.resume)):
            return  # Only runs in automatic mode and while the program is running
        print("[INFO] Pausing SCRIPT...")
        self.scripts.running = False
        self.scripts.stop = True
        self.scripts.resume = False
        print("[INFO] Program execution paused.")

    def resume_robot_program(self):
        """Resumes execution of the predefined robot program."""
        if self.manual_mode  or (not self.scripts.running and not self.scripts.resume and not self.scripts.stop):
            return  # Only runs in automatic mode
        elif self.robot.Busy() or self.scripts.running or self.scripts.resume:
            print("[ERROR] Robot is already running a program!")
            return
        print(f"[INFO] Resuming SCRIPT...")
        self.scripts.running = False
        self.scripts.stop = False
        self.scripts.resume = True
        print("[INFO] Program resumed.")

    def emergency_stop(self):   
        """Stops the robot immediately."""
        print("[INFO] Emergency stop activated.")
        #self.RDK.RunCode("stopj(6.0)", True)
        self.scripts.stop = True
        self.scripts.resume = False
        self.scripts.running = False
        self.movement_timer.stop()
        self.rotation_timer.stop()
        #self.robot.Stop()
        #self.robot.Disconnect()
        #self.gripper.socket.close()
        raise SystemExit("[INFO] Emergency stop activated. Please restart the system.")

################################### DIALOG WINDOWS ###############################################################
    def ask_mode(self):
        '''Creates a dialog window to select the mode of execution.'''
        dialog = QDialog()
        dialog.setWindowTitle("Select Mode")

        selection = {"value": None}

        label = QLabel("Select assembly or disassembly")

        # Create buttons for each mode
        btn1 = QPushButton("Automatic Assembly")
        btn2 = QPushButton("Automatic Disassembly")
        btn3 = QPushButton("Manual Assembly")

        def select(value):
            selection["value"] = value
            dialog.accept()

        # Connect buttons to the selection function
        btn1.clicked.connect(lambda: select(1))
        btn2.clicked.connect(lambda: select(2))
        btn3.clicked.connect(lambda: select(3))

        layout = QVBoxLayout()
        layout.addWidget(label)

        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(btn1)
        button_layout.addWidget(btn2)
        button_layout.addWidget(btn3)

        layout.addLayout(button_layout)
        dialog.setLayout(layout)

        # Execute the dialog and get the result
        result = dialog.exec()
        return selection["value"] if result == QDialog.DialogCode.Accepted else None

    def ask_cell_selection(self, selected_cell):
        '''Creates a dialog window to select a cell for assembly.'''
        dialog = QDialog()
        dialog.setWindowTitle("Select cell for assembly")

        selection = {"value": None}

        label = QLabel("Select cell for assembly")
        layout = QVBoxLayout()
        layout.addWidget(label)

        # Create buttons for each cell
        for i in range(1, 13):
            btn = QPushButton(str(i))
            if i in selected_cell:
                btn.setEnabled(False)
            else:
                btn.clicked.connect(lambda _, value=i: select(value))
                layout.addWidget(btn)

        def select(value):
            selection["value"] = value
            dialog.accept()

        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()

        layout.addLayout(button_layout)
        dialog.setLayout(layout)

        # Execute the dialog and get the result
        result = dialog.exec()
        return selection["value"] if result == QDialog.DialogCode.Accepted else None

    def ask_selection(self):
        '''Creates a dialog window to select a box for assembly or disassembly.'''
        dialog = QDialog()
        dialog.setWindowTitle("Select Box")

        selection = {"value": None}

        label = QLabel("Select box for assembly")

        # Create buttons for each box
        btn1 = QPushButton("1")
        btn2 = QPushButton("2")
        btn3 = QPushButton("3")

        def select(value):
            selection["value"] = value
            dialog.accept()

        # Connect buttons to the selection function
        btn1.clicked.connect(lambda: select(1))
        btn2.clicked.connect(lambda: select(2))
        btn3.clicked.connect(lambda: select(3))

        layout = QVBoxLayout()
        layout.addWidget(label)

        # Create a horizontal layout for the buttons
        button_layout = QHBoxLayout()
        button_layout.addWidget(btn1)
        button_layout.addWidget(btn2)
        button_layout.addWidget(btn3)

        layout.addLayout(button_layout)
        dialog.setLayout(layout)

        # Execute the dialog and get the result
        result = dialog.exec()
        return selection["value"] if result == QDialog.DialogCode.Accepted else None

    def ask_reset(self):
        '''Creates a dialog window to display the image of the resetted cell
        and prompts the user to ensure that the cell and image are in the same state.'''
        dialog = QDialog()
        dialog.setWindowTitle("Initial State")

        selection = {"value": None}

        label1 = QLabel("Compare this image to the physical cell.")

        # image
        image_label = QLabel()
        pixmap = QPixmap(RESET_IMAGE_PATH)
        image_label.setPixmap(pixmap)
        image_label.setScaledContents(True)
        image_label.setFixedSize(400, 300)

        label2 = QLabel("Are the cell and image in the same state?")
        label3 = QLabel("If not, change the cell to match the image and confirm.")
        label4 = QLabel("Select the option that applies.")
        # Create buttons for each mode
        btn1 = QPushButton("Confirm")
        btn2 = QPushButton("Cancel")

        def select(value):
            selection["value"] = value
            dialog.accept()

        # Connect buttons to the selection function
        btn1.clicked.connect(lambda: select(1))
        btn2.clicked.connect(lambda: select(2))

        layout = QVBoxLayout()
        layout.addWidget(label1)
        layout.addWidget(image_label)
        layout.addWidget(label2)
        layout.addWidget(label3)
        layout.addWidget(label4)

        button_layout = QHBoxLayout()
        button_layout.addWidget(btn1)
        button_layout.addWidget(btn2)

        layout.addLayout(button_layout)
        dialog.setLayout(layout)

        # Execute the dialog and get the result
        result = dialog.exec()
        return selection["value"] if result == QDialog.DialogCode.Accepted else None

    def ask_retry_connection(self, object):
        '''Creates a dialog window to notify the failed connection to the robot
        and prompts the user to retry the connection.'''
        dialog = QDialog()
        dialog.setWindowTitle("Connection Failed")

        selection = {"value": None}

        label1 = QLabel(f"Failed to connect to {object}.")
        label2 = QLabel(f"Check network and {object} settings.")
        label3 = QLabel("Retry connection?")

        # Create buttons for each mode
        btn1 = QPushButton("Retry")
        btn2 = QPushButton("Dismiss")

        def select(value):
            selection["value"] = value
            dialog.accept()

        # Connect buttons to the selection function
        btn1.clicked.connect(lambda: select(1))
        btn2.clicked.connect(lambda: select(2))

        layout = QVBoxLayout()
        layout.addWidget(label1)
        layout.addWidget(label2)
        layout.addWidget(label3)

        button_layout = QHBoxLayout()
        button_layout.addWidget(btn1)
        button_layout.addWidget(btn2)

        layout.addLayout(button_layout)
        dialog.setLayout(layout)

        # Execute the dialog and get the result
        result = dialog.exec()
        return selection["value"] if result == QDialog.DialogCode.Accepted else None

################################ VIRTUAL REPLACEMENT ########################################################
    def virtual_replace(self, box, cells):
        """Replaces the virtual objects in the simulation."""
        # replace the virtual cells, box and lid
        for cell_num in cells:
            prog = self.RDK.Item(f'Replace cell{cell_num}', ITEM_TYPE_PROGRAM)
            if prog.Valid():
                print("[INFO] Replacing virtual cell")
                prog.RunCode()
        prog = self.RDK.Item(f'Replace lid{box}', ITEM_TYPE_PROGRAM)
        if prog.Valid():
            print("[INFO] Replacing virtual lid")
            prog.RunCode()
        prog = self.RDK.Item(f'Replace box{box}', ITEM_TYPE_PROGRAM)
        if prog.Valid():
            print("[INFO] Replacing virtual box")
            prog.RunCode()
        print("[INFO] Finished replacing virtual objects")

###################################################################################################################
#################### MAIN SETUP AND EXECUTION #####################################################################
###################################################################################################################
def RunUI():
    ## Initialize RoboDK and the robot
    RDK = Robolink()
    try:
        robot = RDK.ItemUserPick('Select a robot', ITEM_TYPE_ROBOT)
        if not robot.Valid():
            print("Error: Robot not found in RoboDK!")
            raise FileNotFoundError

        ## Initialize the UI
        app = QApplication([])
        window = RobotControlUI(robot, RDK)
        window.show()
        app.exec()
    except FileNotFoundError:
        print("Opening station...")
        RDK.AddFile(ROBOTIC_STATION_PATH)
        robot = RDK.ItemUserPick('Select a robot', ITEM_TYPE_ROBOT)
        if not robot.Valid():
            print("Error: Robot not found in RoboDK!")
            raise SystemError

        ## Initialize the UI
        app = QApplication([])
        window = RobotControlUI(robot, RDK)
        window.show()
        app.exec()

# Run the UI externally
if __name__ == "__main__":
    RunUI()
