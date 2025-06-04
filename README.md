# DigitalTwin
Connected Digital Twin for Robotic Battery Assembly and Disassembly

Authors: Alberto Prats Ferrandis and Selene Delgado Pastor.

This repository contains the files necessary to set up and execute the developed digital twin for robotic battery assembly and disassembly. This code has been developed for a thesis project.

This project presents the design and implementation of a digital twin-based user interface for the real-time control of a UR10e robotic arm in an industrial assembly context, specifically targeting robotic cell box assembly and disassembly. The user interface, developed using the PyQt6 framework and integrated through RoboDK, enables manual and automated control, speed and gripper adjustments, and visual feedback of the robotic process. This work aims to show the benefits of the implementation of digital twins, and ultimately demonstrates a practical and flexible solution for enhancing human-robot interaction in smart manufacturing, bringing the feasibility and effectiveness of combining digital twins with the industry, ensuring optimal control, advanced user interface design, and the ability to control complex systems.

The ui.py file reads all of the files included here, so do not change their names or the UI might not function as intended.
The content is the following:
- ui.py :    Main Python file that contains the UI, open the file in VS Code and execute in terminal, it will open RoboDK and the station automatically, given that you have a valid RoboDK license. Execute this file to use the UI, has a manual robot control in addition to the assembly and disassembly processes.
- robotcode.py :     Python file containing the work of Paula Bolivar Perez and Aitor Sagarna Zabala, controls the robot movement to perform the assembly and disassembly processes. Contains the vision system. Execute this file to use the simple terminal-based execution of the assembly and disassembly processes. Does not have manual robot control.
- gripper.py :      Python file containing the code that controls both the virtual and physical grippers. Executing this code does not perform any actions.
- Gripper .urp and .script files :    Gripper control files created by the UI to send to the UR10e robot through socket, they can be deleted, as the UI checks for their existance and creates them again if they are not found.
- reset_station.png :    Image of how the Robotic station should look before the program is started. The image is used by the UI file.
- joints.csv :    Csv file containing the joint values of the performed paths, this file can be deleted as the UI creates it every time maintenance is called.
- joint_usage_history.csv :    Csv file containing historical joint values of the performed paths, this file is saved and edited each time maintenance is called, to reset the history, simply delete this file, as the UI will create a new one if it does not exist whenever maintenance is called.
- robotic_station.rdk :    RoboDK file containing the virtual robotic station. This file cannot be opened without the RoboDk license, as it contains too may objects for the free/deactivated license. This file is opened automatically when the UI is started, so ensure you have the RoboDK license active by the time you want to execute the UI.

Use of this code or structure is encouraged, with the right considerations stated below. Read the thesis before expanding on these files. Any additional work must reference the thesis paper in this repository. Any use of the results, virtual model or code must make reference to at least one of the authors.

The software developed and presented within this thesis has been tested and validated exclusively within the defined scope and under the specific conditions described throughout the project. Any modification of the source code, execution outside the recommended hardware or software environment, or use beyond the scenarios and safety assumptions established in the thesis falls entirely outside the intended design. Therefore, the author does not assume any responsibility or liability for potential malfunctions, unintended behaviors, or damages resulting from such uses or ocurring outside the authors' supervision. The use of the software developed must be executed with constant supervision of a trained individual. It is the responsibility of any third party using this code to ensure proper validation and risk assessment before deployment.


