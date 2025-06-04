from robodk.robolink import * # RoboDK API
from time import *
import socket

IP = "192.168.0.102"
PORT = 30002

base = __file__.rsplit("\\", 1)[0]  # Windows-only compatibitily!
GRIPPER_FOLDER_PATH = f"{base}\\local_files"
GRIPPER_OPEN_PATH = f"{base}\\gripper_open.script"
GRIPPER_OPEN_CELL_PATH = f"{base}\\gripper_open_cell.script"
GRIPPER_CLOSE_BOX_PATH = f"{base}\\gripper_close_box.script"
GRIPPER_CLOSE_CELL_PATH = f"{base}\\gripper_close_cell.script"
GRIPPER_CLOSE_LID_PATH = f"{base}\\gripper_close_lid.script"

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
                try:
                    if self.parent.ask_retry_connection('gripper') == 2:
                        raise SystemExit("[ERROR] Connection attempt aborted.")
                except:
                    return

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
