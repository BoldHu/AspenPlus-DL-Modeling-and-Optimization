import win32com.client
from export import exporter
from aspen import aspen_model

# Step 1: Connect to Aspen Plus
aspen = win32com.client.Dispatch("Apwn.Document")

# Step 2: Open the Aspen Plus model file
model_path = r'D:\python_workspace\AspenPlus Optimization\PD-P1-CNR-1.bkp'
aspen.InitFromArchive2(model_path)

# Step 5: Close the Aspen Plus connection
aspen.Close()

# Release the COM object
del aspen
