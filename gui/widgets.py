from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QLineEdit, 
    QFileDialog, QGroupBox, QTextEdit, QProgressBar
)
from PyQt5.QtCore import pyqtSignal
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class FileSelectionWidget(QWidget):
    """A widget for selecting a file or directory."""
    
    file_selected = pyqtSignal(str)
    
    def __init__(self, label_text, placeholder_text, mode="file_open", 
                 file_filter="", title="Select File", parent=None):
        """Initialize file selection widget."""
        super().__init__()
        
        self.mode = mode
        self.file_filter = file_filter
        self.dialog_title = title
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Label
        if label_text:
            self.label = QLabel(label_text, self)
            layout.addWidget(self.label, 2)
        
        # Text field
        self.text_field = QLineEdit(self)
        self.text_field.setPlaceholderText(placeholder_text)
        layout.addWidget(self.text_field, 7)
        
        # Browse button
        self.browse_button = QPushButton("Browse...", self)
        self.browse_button.clicked.connect(self.browse)
        layout.addWidget(self.browse_button, 1)
    
    def browse(self):
        """Open a file dialog to select a file or directory."""
        path = ""
        if self.mode == "directory":
            path = QFileDialog.getExistingDirectory(self, self.dialog_title)
        elif self.mode == "file_open":
            path, _ = QFileDialog.getOpenFileName(self, self.dialog_title, "", self.file_filter)
        elif self.mode == "file_save":
            path, _ = QFileDialog.getSaveFileName(self, self.dialog_title, "", self.file_filter)
        
        if path:
            self.text_field.setText(path)
            self.file_selected.emit(path)
    
    def get_path(self):
        """Get the selected file or directory path."""
        return self.text_field.text()
    
    def set_path(self, path):
        """Set the file or directory path."""
        self.text_field.setText(path)


class LogDisplay(QGroupBox):
    """A widget for displaying log messages."""
    
    def __init__(self, title="Log", parent=None):
        """Initialize log display widget."""
        super().__init__(parent)  # ✅ Corrected: `parent` is correctly passed

        self.setTitle(title)  # ✅ Set the title separately

        layout = QVBoxLayout(self)
        
        # Text edit for displaying log messages
        self.text_edit = QTextEdit(self)
        self.text_edit.setReadOnly(True)
        layout.addWidget(self.text_edit)
    
    def append(self, message):
        """Append a message to the log display."""
        self.text_edit.append(message)
        
        # Auto-scroll to bottom
        scrollbar = self.text_edit.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def clear(self):
        """Clear the log display."""
        self.text_edit.clear()
        
class StatusBarWidget(QWidget):
    """A widget for displaying status messages and progress."""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        
        # Status label
        self.status_label = QLabel("Ready", self)
        layout.addWidget(self.status_label, 7)
        
        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar, 3)
    
    def set_status(self, message):
        """Set the status message."""
        self.status_label.setText(message)
    
    def set_progress(self, value):
        """Set the progress value."""
        if not self.progress_bar.isVisible():
            self.progress_bar.setVisible(True)
        
        if value >= 0 and value <= 100:
            self.progress_bar.setValue(value)
    
    def hide_progress(self):
        """Hide the progress bar."""
        self.progress_bar.setVisible(False)

class MatplotlibCanvas(FigureCanvasQTAgg):
    """Matplotlib canvas for embedding in Qt with improved performance."""
    def __init__(self, parent=None, width=5, height=4, dpi=100, projection='3d'):
        """Initialize matplotlib canvas."""
        # Create figure
        self.fig = plt.Figure(figsize=(width, height), dpi=dpi)
        
        # Create axes with the specified projection
        if projection == '3d':
            self.axes = self.fig.add_subplot(111, projection='3d')
        else:
            self.axes = self.fig.add_subplot(111)
        
        # Use tight layout for better space utilization
        self.fig.tight_layout()
        
        # Initialize canvas
        super(MatplotlibCanvas, self).__init__(self.fig)
        
        # Set parent if provided
        if parent:
            self.setParent(parent)
            
        # Configure for Qt
        from PyQt5.QtWidgets import QSizePolicy
        self.setMinimumSize(100, 100)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.updateGeometry()