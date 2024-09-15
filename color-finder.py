import sys
from PIL import Image
from colorthief import ColorThief

def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])

def get_color_palette(image_path, color_count=6):
    try:
        color_thief = ColorThief(image_path)
        palette = color_thief.get_palette(color_count=color_count)
        return [rgb_to_hex(color) for color in palette]
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python script.py <image_path> [color_count]")
        sys.exit(1)

    image_path = sys.argv[1]
    color_count = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    palette = get_color_palette(image_path, color_count)
    
    if palette:
        print("Color Palette:")
        for color in palette:
            print(color)

if __name__ == "__main__":
    main()
