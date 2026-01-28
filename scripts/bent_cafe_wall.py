#!/usr/bin/env python3
"""Café Wall illusion but the lines are ACTUALLY slanted."""

from PIL import Image, ImageDraw

def create_slanted_cafe_wall(width=800, height=400, rows=8, cols=16, slant=6):
    img = Image.new('RGB', (width, height), 'white')
    draw = ImageDraw.Draw(img)

    cell_w = width // cols
    line_h = 3
    row_h = (height - line_h * (rows + 1)) // rows

    gray = '#808080'

    def get_line_y(row_idx, x):
        """Get y position of line at given x coordinate."""
        base_y = row_idx * (row_h + line_h) + line_h // 2
        direction = 1 if row_idx % 2 == 0 else -1
        return base_y + direction * slant * (x / width - 0.5)

    # Draw slanted lines first
    for row in range(rows + 1):
        y_left = get_line_y(row, 0)
        y_right = get_line_y(row, width)
        draw.line([(0, y_left), (width, y_right)], fill=gray, width=line_h)

    # Draw boxes between lines - they touch horizontally
    for row in range(rows):
        h_offset = (row % 2) * (cell_w // 2)

        for col in range(-1, cols + 1):
            x_left = col * cell_w + h_offset
            x_right = x_left + cell_w

            # Alternate colors
            if col % 2 == 0:
                color = 'black'
            else:
                color = 'white'

            # Get line positions at box edges (exact edge, no gap)
            top_y_left = get_line_y(row, x_left) + line_h / 2
            top_y_right = get_line_y(row, x_right) + line_h / 2
            bot_y_left = get_line_y(row + 1, x_left) - line_h / 2
            bot_y_right = get_line_y(row + 1, x_right) - line_h / 2

            points = [
                (x_left, top_y_left),
                (x_right, top_y_right),
                (x_right, bot_y_right),
                (x_left, bot_y_left),
            ]
            draw.polygon(points, fill=color)

    return img

if __name__ == '__main__':
    from pathlib import Path
    img = create_slanted_cafe_wall(slant=8)
    output_path = Path(__file__).parent.parent / "outputs" / "bent_cafe_wall.png"
    img.save(output_path)
    print(f"Saved to {output_path}")
