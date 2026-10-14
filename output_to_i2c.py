"""
output_to_i2c.py

Helper for sending recognition results to an SSD1306 I2C OLED.
Designed to be imported and used alongside recognize_rknn.py.
"""

from luma.core.interface.serial import i2c
from luma.oled.device import ssd1306
from luma.core.render import canvas
from PIL import ImageFont

# --- Configuration ---
# These are the only two values you might need to change.
I2C_PORT = 1          # The I2C bus number (e.g., 1, 7). Check with `i2cdetect -y <port>`
I2C_ADDRESS = 0x3C    # The I2C address of your display. 0x3C is very common.
# ---------------------

# Try to load a default font. If not found, luma falls back to a basic one.
try:
    # This path is common on Debian/Ubuntu systems. If it fails, it's fine.
    FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
    FONT = ImageFont.truetype(FONT_PATH, 14)
except (OSError, IOError):
    print("Warning: Default font not found. Using luma's built-in fallback.")
    FONT = None

# Initialize the OLED device.
# This will raise an exception if the device isn't connected or the port is wrong.
try:
    serial = i2c(port=I2C_PORT, address=I2C_ADDRESS)
    device = ssd1306(serial, rotate=0) # rotate=0 is landscape, rotate=1 is portrait
    print(f"I2C OLED initialized on bus {I2C_PORT} at address {hex(I2C_ADDRESS)}")
except Exception as e:
    print(f"ERROR: Could not initialize I2C OLED. Check wiring, port, and address.")
    print(f"Details: {e}")
    device = None

def display_results(results: list[str]):
    """
    Display a list of strings on the OLED.

    Parameters
    ----------
    results : list[str]
        A list of strings, e.g., ["ID 1: TPOSE 0.87", "ID 2: BORED 0.62"]
    """
    if device is None:
        return # Silently do nothing if the display isn't available.

    # The canvas context manager handles clearing the buffer and
    # updating the physical display when the `with` block exits.
    with canvas(device) as draw:
        # If there are no results, show a default message.
        if not results:
            draw.text((0, 0), "No people", font=FONT, fill="white")
            return

        # Draw up to 4 lines of text (SSD1306 128x64 has room for ~4 lines of small text)
        y_pos = 0
        for line in results[:4]: # Limit to 4 lines to avoid overflow
            # Truncate the line to avoid it running off the right edge.
            # This is a simple truncation; for long text, you'd want scrolling.
            max_chars = 21 # Approx for a 128px wide display with default font
            if len(line) > max_chars:
                line = line[:max_chars-3] + "..."

            draw.text((0, y_pos), line, font=FONT, fill="white")
            y_pos += 16 # Move down for the next line

# A simple test if you run this script directly.
if __name__ == "__main__":
    print("Running OLED test...")
    test_results = [
        "ID 1: TPOSE 0.87",
        "ID 2: BORED 0.62",
        "Test line 3",
        "Test line 4"
    ]
    display_results(test_results)
    print("Test complete. Check the display.")