# DigitalTwin
Connected Digital Twin for Robotic Battery Assembly and Disassembly
Authors: Alberto Prats Ferrandis and Selene Delgado Pastor.

This repository contains the files necessary to set up and execute the developed digital twin for robotic battery assembly and disassembly.

The dt_ui.py file reads all of the files included here, so do not change their names or the UI might not function as intended.
The content is the following:
- dt_ui.py :    Main Python file that contains the UI, open the file in VS Code and execute in terminal, it will open RoboDK and the station automatically, given that you have a valid RoboDK license.
- Gripper .urp and .script files :    Gripper control files created by the UI to send to the UR10e robot through socket, they can be deleted, as the UI checks for their existance and creates them again if they are not found.
- reset_station.png :    Image of how the Robotic station should look before the program is started. The image is used by the UI file.
- joints.csv :    Csv file containing the joint values of the performed paths, this file can be deleted as the UI creates it every time maintenance is called.
- joint_usage_history.csv :    Csv file containing historical joint values of the performed paths, this file is saved and edited each time maintenance is called, to reset the history, simply delete this file, as the UI will create a new one if it does not exist whenever maintenance is called.
- robotic_station.rdk :    RoboDK file containing the virtual robotic station. This file cannot be opened without the RoboDk license, as it contains too may objects for the free/deactivated license. This file is opened automatically when the UI is started, so ensure you have the RoboDK license active by the time you want to execute the UI.

Use of this code or structure is encouraged, with the right considerations. Read the thesis before expanding on these files. Any additional work must reference the thesis paper in this repository.
