import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
from canny_edge import can_main_window
import os

def gui_main_window():

    pressure_filename = ""
    video_filename = ""
    def pressure_cmd():
        nonlocal pressure_filename
        pressure_filename = filedialog.askopenfilename()
        pres_label.configure(text = str(pressure_filename))
        if pressure_filename == "":
            return
    filepath = ""
    def vid_cmd():
        nonlocal video_filename
        nonlocal filepath
        video_filename = filedialog.askopenfilename()
        filepath = os.path.dirname(os.path.abspath(video_filename))
        print(filepath)
        vid_label.configure(text = str(video_filename))
        if video_filename == "":
            return

    def run_cmd():
        try:
            low_entryP = int(id_entry.get())
            high_entryP = int(id_entry2.get())
        except ValueError:
            low_entryP=50
            high_entryP=90

        outputs = can_main_window(video_filename, pressure_filename, low_entryP, high_entryP, filepath)

        comp_labelLH.configure(text=str(outputs[0]))
        comp_label5090.configure(text=str(outputs[1]))
        comp_label80120.configure(text=str(outputs[2]))
        comp_label110150.configure(text=str(outputs[3]))


    ## Create root/base window
    root = tk.Tk()
    root.title("Compliance Analysis")
    root.geometry("1200x400")



    ## Buttons
    ttk.Button(root, text="Select Pressure Data", command=pressure_cmd).grid(column = 2, row = 2, sticky=tk.W)
    ttk.Button(root, text="Choose Video File", command=vid_cmd).grid(column=2, row=3, sticky=tk.W)
    ttk.Button(root, text="Run Analysis", command=run_cmd).grid(column=2, row=6, sticky=tk.W)

    ## Pressure and Video File Labels
    ttk.Label(root, text="Pressure File:").grid(column=3, row=2, sticky=tk.W)
    pres_label = ttk.Label(root, text="No File Selected")
    pres_label.grid(column=4, row=2, sticky=tk.W)

    ttk.Label(root, text="Video File:").grid(column=3, row=3, sticky=tk.W)
    vid_label = ttk.Label(root, text="No File Selected")
    vid_label.grid(column=4, row=3, sticky=tk.W)

    ## Desired Pressure Range Entry
    ttk.Label(root, text="Low Pressure (mmHg):").grid(column=2, row=4, sticky=tk.W)
    id_entry = tk.StringVar()
    id_entry.set("Enter Low Pressure")
    ttk.Entry(root, textvariable=id_entry).grid(column=3, row=4, sticky=tk.W)

    ttk.Label(root, text="High Pressure (mmHg):").grid(column=2, row=5, sticky=tk.W)
    id_entry2 = tk.StringVar()
    id_entry2.set("Enter High Pressure")
    ttk.Entry(root, textvariable=id_entry2).grid(column=3, row=5, sticky=tk.W)


    ## Compliance 50/90 Label
    ttk.Label(root, text="Compliance Whole Vessel 50-90 mmHg:").grid(column=2, row=7,sticky=tk.W)
    comp_label5090 = ttk.Label(root, text="No Analysis Run")
    comp_label5090.grid(column=3, row=7, sticky=tk.W)

    ## Compliance 80/120 Label
    ttk.Label(root, text="Compliance Whole Vessel 80-120 mmHg:").grid(column=2, row=8, sticky=tk.W)
    comp_label80120 = ttk.Label(root, text="No Analysis Run")
    comp_label80120.grid(column=3, row=8, sticky=tk.W)

    ## Compliance 110/150 Label
    ttk.Label(root, text="Compliance Whole Vessel 110-150 mmHg:").grid(column=2, row=9, sticky=tk.W)
    comp_label110150 = ttk.Label(root, text="No Analysis Run")
    comp_label110150.grid(column=3, row=9, sticky=tk.W)

    ## Compliance Low/High Label
    ttk.Label(root, text="Compliance Whole Vessel Low-High mmHg:").grid(column=2, row=10, sticky=tk.W)
    comp_labelLH = ttk.Label(root, text="No Analysis Run")
    comp_labelLH.grid(column=3, row=10, sticky=tk.W)

    ## Start GUI
    root.mainloop()

if __name__ == '__main__':
    gui_main_window()