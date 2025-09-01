# Import required libraries
import tkinter as tk  # For creating GUI
from tkinter import ttk, filedialog, messagebox  # Additional GUI components
import numpy as np  # For array operations
from PIL import Image, ImageTk, ImageDraw  # For image processing and display
from osgeo import gdal, osr  # GDAL library for GeoTIFF handling
import os  # For file operations

class GeoTIFFViewer:
    def __init__(self, root):
        # Initialize the main window
        self.root = root
        self.root.title("GeoTIFF Viewer - Computer Vision Assignment")
        self.root.geometry("1000x700")
        
        # Initialize variables to store image data
        self.dataset = None  # GDAL dataset object
        self.image_array = None  # Image data as numpy array
        self.geotransform = None  # Geographic transformation parameters
        self.projection = None  # Coordinate system information
        self.display_image = None  # Image for display in GUI
        self.photo_image = None  # PhotoImage object for tkinter
        self.canvas_image = None  # Canvas image reference
        
        # Variables for zoom and pan functionality
        self.zoom_factor = 1.0  # Current zoom level
        self.pan_start_x = 0  # Starting X position for panning
        self.pan_start_y = 0  # Starting Y position for panning
        self.image_offset_x = 0  # Image offset in X direction
        self.image_offset_y = 0  # Image offset in Y direction
        
        # List to store marked locations
        self.marked_points = []
        
        # Create the GUI interface
        self.create_gui()
        
    def create_gui(self):
        # Create main frame to hold all components
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Create control panel at the top
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=(0, 10))
        
        # Load image button
        load_button = ttk.Button(control_frame, text="Load GeoTIFF Image", command=self.load_image)
        load_button.pack(side=tk.LEFT, padx=(0, 10))
        
        # Mouse coordinates display
        coord_frame = ttk.Frame(control_frame)
        coord_frame.pack(side=tk.LEFT, padx=(0, 20))
        
        # Pixel coordinates labels and text boxes
        ttk.Label(coord_frame, text="Pixel X:").grid(row=0, column=0, sticky=tk.W)
        self.pixel_x_var = tk.StringVar()  # Variable to hold pixel X coordinate
        ttk.Entry(coord_frame, textvariable=self.pixel_x_var, width=8, state='readonly').grid(row=0, column=1, padx=(5, 10))
        
        ttk.Label(coord_frame, text="Pixel Y:").grid(row=0, column=2, sticky=tk.W)
        self.pixel_y_var = tk.StringVar()  # Variable to hold pixel Y coordinate
        ttk.Entry(coord_frame, textvariable=self.pixel_y_var, width=8, state='readonly').grid(row=0, column=3, padx=(5, 10))
        
        # Geographic coordinates labels and text boxes
        ttk.Label(coord_frame, text="Longitude:").grid(row=1, column=0, sticky=tk.W)
        self.lon_var = tk.StringVar()  # Variable to hold longitude
        ttk.Entry(coord_frame, textvariable=self.lon_var, width=12, state='readonly').grid(row=1, column=1, padx=(5, 10))
        
        ttk.Label(coord_frame, text="Latitude:").grid(row=1, column=2, sticky=tk.W)
        self.lat_var = tk.StringVar()  # Variable to hold latitude
        ttk.Entry(coord_frame, textvariable=self.lat_var, width=12, state='readonly').grid(row=1, column=3, padx=(5, 10))
        
        # Marking location input frame
        mark_frame = ttk.Frame(control_frame)
        mark_frame.pack(side=tk.LEFT)
        
        # Input fields for marking specific locations
        ttk.Label(mark_frame, text="Mark Location:").grid(row=0, column=0, columnspan=2, sticky=tk.W)
        ttk.Label(mark_frame, text="Longitude:").grid(row=1, column=0, sticky=tk.W)
        self.mark_lon_var = tk.StringVar()  # Variable for longitude input
        ttk.Entry(mark_frame, textvariable=self.mark_lon_var, width=12).grid(row=1, column=1, padx=(5, 0))
        
        ttk.Label(mark_frame, text="Latitude:").grid(row=2, column=0, sticky=tk.W)
        self.mark_lat_var = tk.StringVar()  # Variable for latitude input
        ttk.Entry(mark_frame, textvariable=self.mark_lat_var, width=12).grid(row=2, column=1, padx=(5, 0))
        
        # Mark button to add cross at specified location
        mark_button = ttk.Button(mark_frame, text="Mark Location", command=self.mark_location)
        mark_button.grid(row=3, column=0, columnspan=2, pady=(5, 0))
        
        # Create canvas for image display
        self.canvas = tk.Canvas(main_frame, bg='white', width=800, height=600)
        self.canvas.pack(fill=tk.BOTH, expand=True)
        
        # Bind mouse events to canvas for interactivity
        self.canvas.bind('<Motion>', self.on_mouse_move)  # Track mouse movement
        self.canvas.bind('<MouseWheel>', self.on_mouse_wheel)  # Handle zoom with mouse wheel
        self.canvas.bind('<Button-1>', self.on_mouse_press)  # Handle mouse press for panning
        self.canvas.bind('<B1-Motion>', self.on_mouse_drag)  # Handle mouse drag for panning
        
    def load_image(self):
        # Open file dialog to select GeoTIFF file
        file_path = filedialog.askopenfilename(
            title="Select GeoTIFF Image",
            filetypes=[("GeoTIFF files", "*.tif *.tiff"), ("All files", "*.*")]
        )
        
        # Check if user selected a file
        if not file_path:
            return
            
        try:
            # Load the GeoTIFF file using GDAL
            self.dataset = gdal.Open(file_path)
            if self.dataset is None:
                messagebox.showerror("Error", "Could not open the selected file")
                return
            
            # Get image dimensions
            width = self.dataset.RasterXSize  # Image width in pixels
            height = self.dataset.RasterYSize  # Image height in pixels
            
            # Get geotransformation parameters (maps pixel coordinates to geographic coordinates)
            self.geotransform = self.dataset.GetGeoTransform()
            
            # Get coordinate system information
            self.projection = self.dataset.GetProjection()
            
            # Read image data from first band (assuming RGB or grayscale)
            band = self.dataset.GetRasterBand(1)  # Get first band
            self.image_array = band.ReadAsArray()  # Read band data as numpy array
            
            # Normalize image data for display (convert to 0-255 range)
            if self.image_array.dtype != np.uint8:
                # Normalize to 0-255 range for display
                img_min = np.min(self.image_array)
                img_max = np.max(self.image_array)
                self.image_array = ((self.image_array - img_min) / (img_max - img_min) * 255).astype(np.uint8)
            
            # Convert numpy array to PIL Image for display
            if len(self.image_array.shape) == 2:  # Grayscale image
                pil_image = Image.fromarray(self.image_array, 'L')
            else:  # Color image
                pil_image = Image.fromarray(self.image_array, 'RGB')
            
            # Store original image for zoom/pan operations
            self.original_image = pil_image
            
            # Display the image on canvas
            self.display_image_on_canvas(pil_image)
            
            # Reset zoom and pan parameters
            self.zoom_factor = 1.0
            self.image_offset_x = 0
            self.image_offset_y = 0
            
            # Clear any existing marked points
            self.marked_points = []
            
            messagebox.showinfo("Success", f"Image loaded successfully!\nDimensions: {width} x {height}")
            
        except Exception as e:
            # Show error message if loading fails
            messagebox.showerror("Error", f"Failed to load image: {str(e)}")
    
    def display_image_on_canvas(self, pil_image):
        # Resize image to fit canvas while maintaining aspect ratio
        canvas_width = self.canvas.winfo_width()
        canvas_height = self.canvas.winfo_height()
        
        # Calculate scaling factor to fit image in canvas
        if canvas_width > 1 and canvas_height > 1:  # Make sure canvas is initialized
            img_width, img_height = pil_image.size
            scale_x = canvas_width / img_width
            scale_y = canvas_height / img_height
            scale = min(scale_x, scale_y) * 0.9  # Use 90% of available space
            
            # Apply zoom factor
            scale *= self.zoom_factor
            
            # Calculate new image size
            new_width = int(img_width * scale)
            new_height = int(img_height * scale)
            
            # Resize image for display
            self.display_image = pil_image.resize((new_width, new_height), Image.Resampling.LANCZOS)
            
            # Convert to PhotoImage for tkinter
            self.photo_image = ImageTk.PhotoImage(self.display_image)
            
            # Clear canvas and display image
            self.canvas.delete("all")
            
            # Calculate image position (center image with pan offset)
            x = (canvas_width - new_width) // 2 + self.image_offset_x
            y = (canvas_height - new_height) // 2 + self.image_offset_y
            
            # Display image on canvas
            self.canvas_image = self.canvas.create_image(x, y, anchor=tk.NW, image=self.photo_image)
            
            # Store display parameters for coordinate conversion
            self.display_scale = scale
            self.display_x = x
            self.display_y = y
            
            # Redraw any marked points
            self.redraw_marked_points()
    
    def on_mouse_move(self, event):
        # Handle mouse movement over the image
        if self.dataset is None or self.display_image is None:
            return
        
        # Convert canvas coordinates to image pixel coordinates
        canvas_x = event.x
        canvas_y = event.y
        
        # Calculate image coordinates
        img_x = (canvas_x - self.display_x) / self.display_scale
        img_y = (canvas_y - self.display_y) / self.display_scale
        
        # Check if mouse is over the image
        if 0 <= img_x < self.original_image.size[0] and 0 <= img_y < self.original_image.size[1]:
            # Update pixel coordinate display
            self.pixel_x_var.set(f"{int(img_x)}")
            self.pixel_y_var.set(f"{int(img_y)}")
            
            # Convert pixel coordinates to geographic coordinates
            if self.geotransform:
                # Apply geotransformation to get geographic coordinates
                geo_x = self.geotransform[0] + img_x * self.geotransform[1] + img_y * self.geotransform[2]
                geo_y = self.geotransform[3] + img_x * self.geotransform[4] + img_y * self.geotransform[5]
                
                # Update geographic coordinate display
                self.lon_var.set(f"{geo_x:.6f}")
                self.lat_var.set(f"{geo_y:.6f}")
        else:
            # Clear coordinate display when mouse is outside image
            self.pixel_x_var.set("")
            self.pixel_y_var.set("")
            self.lon_var.set("")
            self.lat_var.set("")
    
    def mark_location(self):
        # Mark a specific location on the image using latitude and longitude
        if self.dataset is None:
            messagebox.showerror("Error", "Please load an image first")
            return
        
        try:
            # Get longitude and latitude from input fields
            lon = float(self.mark_lon_var.get())
            lat = float(self.mark_lat_var.get())
            
            # Convert geographic coordinates to pixel coordinates
            if self.geotransform:
                # Calculate the inverse transformation
                det = self.geotransform[1] * self.geotransform[5] - self.geotransform[2] * self.geotransform[4]
                
                if det == 0:
                    messagebox.showerror("Error", "Cannot convert coordinates")
                    return
                
                # Apply inverse geotransformation
                temp_x = lon - self.geotransform[0]
                temp_y = lat - self.geotransform[3]
                
                pixel_x = (self.geotransform[5] * temp_x - self.geotransform[2] * temp_y) / det
                pixel_y = (-self.geotransform[4] * temp_x + self.geotransform[1] * temp_y) / det
                
                # Check if coordinates are within image bounds
                if 0 <= pixel_x < self.original_image.size[0] and 0 <= pixel_y < self.original_image.size[1]:
                    # Convert to display coordinates
                    display_x = self.display_x + pixel_x * self.display_scale
                    display_y = self.display_y + pixel_y * self.display_scale
                    
                    # Store marked point
                    self.marked_points.append((display_x, display_y, lon, lat))
                    
                    # Draw cross at the location
                    self.draw_cross(display_x, display_y)
                    
                    messagebox.showinfo("Success", f"Location marked at: {lon:.6f}, {lat:.6f}")
                else:
                    messagebox.showerror("Error", "Coordinates are outside image bounds")
            
        except ValueError:
            # Handle invalid input
            messagebox.showerror("Error", "Please enter valid numeric coordinates")
    
    def draw_cross(self, x, y):
        # Draw a cross marker at specified coordinates
        size = 10  # Size of the cross
        
        # Draw cross lines on canvas
        self.canvas.create_line(x - size, y, x + size, y, fill='red', width=3, tags='marker')
        self.canvas.create_line(x, y - size, x, y + size, fill='red', width=3, tags='marker')
        
        # Draw circle around cross for better visibility
        self.canvas.create_oval(x - size - 2, y - size - 2, x + size + 2, y + size + 2, 
                               outline='red', width=2, tags='marker')
    
    def redraw_marked_points(self):
        # Redraw all marked points after zoom or pan
        self.canvas.delete('marker')  # Remove existing markers
        
        for display_x, display_y, lon, lat in self.marked_points:
            # Recalculate display position based on current zoom and pan
            # Convert back to pixel coordinates and then to current display coordinates
            if self.geotransform:
                det = self.geotransform[1] * self.geotransform[5] - self.geotransform[2] * self.geotransform[4]
                temp_x = lon - self.geotransform[0]
                temp_y = lat - self.geotransform[3]
                pixel_x = (self.geotransform[5] * temp_x - self.geotransform[2] * temp_y) / det
                pixel_y = (-self.geotransform[4] * temp_x + self.geotransform[1] * temp_y) / det
                
                # Convert to current display coordinates
                new_display_x = self.display_x + pixel_x * self.display_scale
                new_display_y = self.display_y + pixel_y * self.display_scale
                
                self.draw_cross(new_display_x, new_display_y)
    
    def on_mouse_wheel(self, event):
        # Handle zoom functionality with mouse wheel
        if self.dataset is None:
            return
        
        # Get zoom direction (positive for zoom in, negative for zoom out)
        if event.delta > 0:
            zoom_change = 1.1  # Zoom in
        else:
            zoom_change = 0.9  # Zoom out
        
        # Apply zoom factor limits
        new_zoom = self.zoom_factor * zoom_change
        if 0.1 <= new_zoom <= 10.0:  # Limit zoom range
            self.zoom_factor = new_zoom
            
            # Refresh display with new zoom level
            self.display_image_on_canvas(self.original_image)
    
    def on_mouse_press(self, event):
        # Handle mouse press for panning
        if self.dataset is None:
            return
        
        # Store starting position for panning
        self.pan_start_x = event.x
        self.pan_start_y = event.y
    
    def on_mouse_drag(self, event):
        # Handle mouse drag for panning
        if self.dataset is None:
            return
        
        # Calculate pan offset
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        
        # Update image offset
        self.image_offset_x += dx
        self.image_offset_y += dy
        
        # Update pan start position
        self.pan_start_x = event.x
        self.pan_start_y = event.y
        
        # Refresh display with new pan offset
        self.display_image_on_canvas(self.original_image)

# Main program execution
if __name__ == "__main__":
    # Create main window
    root = tk.Tk()
    
    # Create application instance
    app = GeoTIFFViewer(root)
    
    # Start GUI event loop
    root.mainloop()