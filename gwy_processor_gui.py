import os
import tkinter as tk
from tkinter import ttk, filedialog, simpledialog, messagebox
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk

# Import your existing gwyfile loader and processing functions
import gwy_loader
import gwy_processing


class GwyProcessorApp(tk.Tk):
    """
    A GUI application for processing Gwyddion (.gwy) files.
    """

    def __init__(self):
        super().__init__()
        self.title("Gwyddion Python Processor")
        self.geometry("1200x800")

        # Data state management
        self.filepath = None
        self.all_channels = {}
        self.current_channel_name = None
        self.original_data = None
        self.processing_history = []  # Stores (data, log_message) tuples
        self.current_data = None
        self.x_real = 1.0
        self.y_real = 1.0

        # --- GUI Layout ---
        self.create_menu()
        self.create_main_layout()

    def create_menu(self):
        """Creates the main application menu."""
        menu_bar = tk.Menu(self)
        self.config(menu=menu_bar)

        file_menu = tk.Menu(menu_bar, tearoff=0)
        menu_bar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Open Gwyddion File...", command=self.open_file)
        file_menu.add_command(label="Batch Process Folder...", command=self.open_batch_process_dialog)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.quit)

    def create_main_layout(self):
        """Creates the main panels for controls, plotting, and logging."""
        main_frame = ttk.Frame(self, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Configure grid layout
        main_frame.grid_columnconfigure(0, weight=1, minsize=250)
        main_frame.grid_columnconfigure(1, weight=4)
        main_frame.grid_rowconfigure(0, weight=3)
        main_frame.grid_rowconfigure(1, weight=1)

        # --- Left: Control Panel ---
        control_panel = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 10))
        self.create_control_widgets(control_panel)

        # --- Right: Plotting Area ---
        plot_frame = ttk.LabelFrame(main_frame, text="Image Display", padding="10")
        plot_frame.grid(row=0, column=1, rowspan=2, sticky="nsew")
        self.create_plot_widgets(plot_frame)

        # --- Bottom-Left: Log Area ---
        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding="10")
        log_frame.grid(row=1, column=0, sticky="nsew", padx=(0, 10), pady=(10, 0))
        self.log_text = tk.Text(log_frame, height=10, state="disabled", wrap="word")
        self.log_text.pack(fill=tk.BOTH, expand=True)

    def create_control_widgets(self, parent):
        """Creates all the buttons and entry fields for processing."""
        parent.grid_columnconfigure(0, weight=1)

        # Channel Selection
        ttk.Label(parent, text="Select Channel:").grid(row=0, column=0, sticky="w", pady=(0, 5))
        self.channel_var = tk.StringVar()
        self.channel_combo = ttk.Combobox(parent, textvariable=self.channel_var, state="readonly")
        self.channel_combo.grid(row=1, column=0, sticky="ew", pady=(0, 15))
        self.channel_combo.bind("<<ComboboxSelected>>", self.on_channel_select)

        # Processing buttons
        ttk.Button(parent, text="Undo Last Change", command=self.undo_last_change).grid(row=2, column=0, sticky="ew", pady=5)
        ttk.Separator(parent, orient="horizontal").grid(row=3, column=0, sticky="ew", pady=10)

        # Background subtraction
        ttk.Button(parent, text="Level by Plane Fit", command=lambda: self.apply_processing(gwy_processing.level_by_plane_fit, "Leveled by plane fit")).grid(row=4, column=0, sticky="ew", pady=2)
        
        poly_frame = ttk.Frame(parent)
        poly_frame.grid(row=5, column=0, sticky="ew", pady=2)
        poly_frame.grid_columnconfigure(0, weight=2)
        poly_frame.grid_columnconfigure(1, weight=1)
        self.poly_order_var = tk.StringVar(value="2")
        ttk.Button(poly_frame, text="Level by Polynomial (Order):", command=self.apply_poly_fit).pack(side="left", fill="x", expand=True)
        ttk.Entry(poly_frame, textvariable=self.poly_order_var, width=3).pack(side="right")

        # Row alignment
        ttk.Button(parent, text="Align Rows (Median of Diffs)", command=lambda: self.apply_processing(gwy_processing.align_rows, "Aligned rows (median of diffs)", method='median_diff')).grid(row=6, column=0, sticky="ew", pady=2)

        # FFT Filtering
        fft_frame = ttk.Frame(parent)
        fft_frame.grid(row=7, column=0, sticky="ew", pady=2)
        fft_frame.grid_columnconfigure(0, weight=2)
        fft_frame.grid_columnconfigure(1, weight=1)
        self.fft_cutoff_var = tk.StringVar(value="10.0")
        ttk.Button(fft_frame, text="FFT Lowpass Filter (Cutoff):", command=self.apply_fft_filter).pack(side="left", fill="x", expand=True)
        ttk.Entry(fft_frame, textvariable=self.fft_cutoff_var, width=5).pack(side="right")

        # Percentile filter
        perc_frame = ttk.Frame(parent)
        perc_frame.grid(row=8, column=0, sticky="ew", pady=2)
        self.min_perc_var = tk.StringVar(value="0.5")
        self.max_perc_var = tk.StringVar(value="99.5")
        ttk.Button(perc_frame, text="Filter by Percentile:", command=self.apply_percentile_filter).pack(side="left")
        ttk.Entry(perc_frame, textvariable=self.min_perc_var, width=4).pack(side="left", padx=2)
        ttk.Label(perc_frame, text="-").pack(side="left")
        ttk.Entry(perc_frame, textvariable=self.max_perc_var, width=4).pack(side="left", padx=2)

    def create_plot_widgets(self, parent):
        """Creates the Matplotlib figure and canvas."""
        self.fig = plt.Figure(figsize=(7, 7), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        toolbar = NavigationToolbar2Tk(self.canvas, parent)
        toolbar.update()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    # --- Core Functionality ---

    def open_file(self):
        """Opens a file dialog to select a .gwy file."""
        filepath = filedialog.askopenfilename(
            title="Select Gwyddion File",
            filetypes=(("Gwyddion files", "*.gwy"), ("All files", "*.*"))
        )
        if not filepath:
            return

        self.filepath = filepath
        try:
            self.all_channels = gwy_loader.load_gwy(self.filepath)
            channel_names = list(self.all_channels.keys())
            self.channel_combo['values'] = channel_names
            if channel_names:
                self.channel_combo.set(channel_names[0])
                self.on_channel_select()
            self.log("File loaded: " + os.path.basename(filepath))
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load file: {e}")
            self.log(f"Error loading {os.path.basename(filepath)}: {e}")

    def on_channel_select(self, event=None):
        """Handles selection of a new data channel."""
        self.current_channel_name = self.channel_var.get()
        if not self.current_channel_name:
            return

        channel_obj = self.all_channels[self.current_channel_name]
        self.original_data = channel_obj.data.copy()
        self.x_real = channel_obj.xreal
        self.y_real = channel_obj.yreal

        # Reset history and set initial state
        self.processing_history = []
        self.update_data(self.original_data, "Loaded channel: " + self.current_channel_name)

    def update_data(self, new_data, log_message):
        """Updates the current data, history, log, and plot."""
        if self.current_data is not None:
            # Don't add to history if it's the initial load
            if self.processing_history:
                self.processing_history.append((self.current_data.copy(), self.last_log_message))
            elif len(self.processing_history) == 0 and self.original_data is not None:
                 self.processing_history.append((self.original_data.copy(), "Initial state"))

        self.current_data = new_data
        self.last_log_message = log_message
        self.log(log_message)
        self.plot_data()

    def apply_processing(self, func, log_message, **kwargs):
        """Generic wrapper to apply a processing function."""
        if self.current_data is None:
            messagebox.showwarning("Warning", "No data loaded.")
            return
        
        try:
            processed_data = func(self.current_data, **kwargs)
            self.update_data(processed_data, log_message)
        except Exception as e:
            messagebox.showerror("Processing Error", f"An error occurred: {e}")
            self.log(f"Error during '{log_message}': {e}")

    def apply_poly_fit(self):
        """Applies polynomial fit with order from the entry box."""
        try:
            order = int(self.poly_order_var.get())
            self.apply_processing(gwy_processing.level_by_polynomial, f"Leveled by polynomial (order {order})", order=order)
        except ValueError:
            messagebox.showerror("Error", "Polynomial order must be an integer.")

    def apply_fft_filter(self):
        """Applies FFT lowpass filter with cutoff from the entry box."""
        try:
            cutoff = float(self.fft_cutoff_var.get())
            dx = self.x_real / self.current_data.shape[1]
            dy = self.y_real / self.current_data.shape[0]
            self.apply_processing(gwy_processing.filter_by_2d_fft, f"FFT lowpass filter (cutoff {cutoff})", cutoff_freq=cutoff, dx=dx, dy=dy, window='hanning')
        except ValueError:
            messagebox.showerror("Error", "FFT cutoff frequency must be a number.")

    def apply_percentile_filter(self):
        """Applies percentile filter with values from entry boxes."""
        try:
            min_p = float(self.min_perc_var.get())
            max_p = float(self.max_perc_var.get())
            self.apply_processing(gwy_processing.filter_by_percentile, f"Filtered to percentile range ({min_p}% - {max_p}%)", min_percentile=min_p, max_percentile=max_p)
        except ValueError:
            messagebox.showerror("Error", "Percentiles must be numbers.")

    def undo_last_change(self):
        """Reverts the data to its state before the last operation."""
        if not self.processing_history:
            messagebox.showinfo("Info", "No history to undo.")
            return

        previous_data, log_message = self.processing_history.pop()
        self.current_data = previous_data
        self.log(f"--- UNDO: Reverted '{self.last_log_message}' ---")
        self.last_log_message = log_message
        self.plot_data()

    def plot_data(self):
        """Clears the plot and redraws the current data."""
        if self.current_data is None:
            return

        self.ax.clear()
        
        # Use a copy for manipulation
        plot_data = self.current_data.copy()
        
        # Set baseline to zero for consistent visualization
        plot_data = gwy_processing.set_baseline_to_zero(plot_data)

        # Convert units for display
        data_nm = plot_data * 1e9
        x_um = self.x_real * 1e6
        y_um = self.y_real * 1e6

        im = self.ax.imshow(
            data_nm,
            extent=(0, x_um, 0, y_um),
            cmap=gwy_processing.get_gwyddion_cmap(),
            origin="upper",
            aspect="equal"
        )
        self.ax.set_title(self.current_channel_name)
        self.ax.set_xlabel("x (µm)")
        self.ax.set_ylabel("y (µm)")

        # Add or update the colorbar
        if hasattr(self, 'cbar'):
            self.cbar.update_normal(im)
        else:
            self.cbar = self.fig.colorbar(im, ax=self.ax)
            self.cbar.set_label("Height (nm)")

        self.canvas.draw()

    def log(self, message):
        """Adds a message to the log display."""
        self.log_text.config(state="normal")
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.log_text.config(state="disabled")

    # --- Batch Processing ---

    def open_batch_process_dialog(self):
        """Opens the dialog to configure and run batch processing."""
        dialog = BatchDialog(self)
        self.wait_window(dialog)

        if dialog.operations:
            folder_path = filedialog.askdirectory(title="Select Folder to Batch Process")
            if not folder_path:
                return
            
            self.run_batch_process(folder_path, dialog.operations, dialog.channel_name.get())

    def run_batch_process(self, folder_path, operations, channel_name):
        """Executes the batch processing job."""
        gwy_files = [f for f in os.listdir(folder_path) if f.endswith('.gwy')]
        if not gwy_files:
            messagebox.showinfo("Info", "No .gwy files found in the selected folder.")
            return

        output_folder = os.path.join(folder_path, "processed")
        os.makedirs(output_folder, exist_ok=True)
        self.log(f"--- Starting Batch Process on {len(gwy_files)} files ---")

        for filename in gwy_files:
            filepath = os.path.join(folder_path, filename)
            try:
                all_channels = gwy_loader.load_gwy(filepath)
                if channel_name not in all_channels:
                    self.log(f"Skipping {filename}: Channel '{channel_name}' not found.")
                    continue

                data = all_channels[channel_name].data.copy()
                
                # Apply sequence of operations
                for op, params in operations:
                    data = op(data, **params)
                
                # Create a new GwyDataField to save
                original_field = all_channels[channel_name]
                processed_field = gwy_loader.GwyDataField(
                    data=data,
                    xreal=original_field.xreal, yreal=original_field.yreal,
                    si_unit_xy=original_field.si_unit_xy,
                    si_unit_z=original_field.si_unit_z
                )

                # Create a new container and save the file
                container = gwy_loader.GwyContainer()
                container['/0/data'] = processed_field
                container['/0/data/title'] = channel_name + " (Processed)"

                output_filename = os.path.splitext(filename)[0] + "_processed.gwy"
                output_path = os.path.join(output_folder, output_filename)
                container.tofile(output_path)
                self.log(f"Processed and saved: {output_filename}")

            except Exception as e:
                self.log(f"Error processing {filename}: {e}")
        
        self.log("--- Batch Process Finished ---")
        messagebox.showinfo("Success", f"Batch processing complete. Files saved in '{output_folder}'.")


class BatchDialog(simpledialog.Dialog):
    """
    A dialog for defining a sequence of batch processing operations.
    """
    def __init__(self, parent):
        self.operations = []
        self.op_listbox = None
        self.channel_name = tk.StringVar(value="Height [Fwd]")
        super().__init__(parent, "Batch Process Setup")

    def body(self, master):
        master.grid_columnconfigure(0, weight=1)
        master.grid_columnconfigure(1, weight=1)

        ttk.Label(master, text="Target Channel Name:").grid(row=0, column=0, columnspan=2, sticky='w', pady=5)
        ttk.Entry(master, textvariable=self.channel_name).grid(row=1, column=0, columnspan=2, sticky='ew', pady=(0, 10))

        ttk.Label(master, text="Available Operations:").grid(row=2, column=0, sticky='w')
        ttk.Label(master, text="Processing Sequence:").grid(row=2, column=1, sticky='w')

        # --- Left: Available operations ---
        left_frame = ttk.Frame(master)
        left_frame.grid(row=3, column=0, padx=(0, 5), sticky='nsew')
        
        self.available_ops = {
            "Level by Plane Fit": (gwy_processing.level_by_plane_fit, {}),
            "Align Rows (Median)": (gwy_processing.align_rows, {'method': 'median_diff'}),
            "Poly Fit (Order 1)": (gwy_processing.level_by_polynomial, {'order': 1}),
            "Poly Fit (Order 2)": (gwy_processing.level_by_polynomial, {'order': 2}),
            "Poly Fit (Order 3)": (gwy_processing.level_by_polynomial, {'order': 3}),
        }
        self.available_listbox = tk.Listbox(left_frame)
        for op_name in self.available_ops:
            self.available_listbox.insert(tk.END, op_name)
        self.available_listbox.pack(fill='both', expand=True)
        
        ttk.Button(left_frame, text="Add ->", command=self.add_op).pack(pady=5)

        # --- Right: Selected operations ---
        right_frame = ttk.Frame(master)
        right_frame.grid(row=3, column=1, padx=(5, 0), sticky='nsew')
        
        self.op_listbox = tk.Listbox(right_frame)
        self.op_listbox.pack(fill='both', expand=True)
        
        ttk.Button(right_frame, text="<- Remove", command=self.remove_op).pack(pady=5)

        return self.available_listbox # initial focus

    def add_op(self):
        """Adds a selected operation to the sequence."""
        selection_indices = self.available_listbox.curselection()
        if not selection_indices:
            return
        
        op_name = self.available_listbox.get(selection_indices[0])
        self.op_listbox.insert(tk.END, op_name)

    def remove_op(self):
        """Removes an operation from the sequence."""
        selection_indices = self.op_listbox.curselection()
        if not selection_indices:
            return
        self.op_listbox.delete(selection_indices[0])

    def apply(self):
        """Called when OK is pressed. Finalizes the operation list."""
        op_names = self.op_listbox.get(0, tk.END)
        self.operations = [self.available_ops[name] for name in op_names]


if __name__ == "__main__":
    app = GwyProcessorApp()
    app.mainloop()