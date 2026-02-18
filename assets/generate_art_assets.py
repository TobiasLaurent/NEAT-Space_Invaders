"""Generate stylized game sprites and background for NEAT Space Invaders."""

from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFilter


ASSETS_DIR = Path(__file__).resolve().parent


def lerp_color(a, b, t):
    return tuple(int((1 - t) * x + t * y) for x, y in zip(a, b))


def make_background():
    width, height = 400, 400
    img = Image.new("RGB", (width, height), (5, 8, 18))
    draw = ImageDraw.Draw(img)

    top = (6, 10, 24)
    bottom = (22, 8, 40)
    for y in range(height):
        draw.line([(0, y), (width, y)], fill=lerp_color(top, bottom, y / (height - 1)))

    nebula = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    n_draw = ImageDraw.Draw(nebula, "RGBA")
    n_draw.ellipse((30, 30, 240, 220), fill=(74, 74, 180, 70))
    n_draw.ellipse((120, 150, 390, 370), fill=(180, 56, 120, 65))
    n_draw.ellipse((180, 10, 390, 180), fill=(54, 130, 180, 55))
    nebula = nebula.filter(ImageFilter.GaussianBlur(28))

    composed = Image.alpha_composite(img.convert("RGBA"), nebula)
    c_draw = ImageDraw.Draw(composed, "RGBA")

    rng = random.Random(17)
    for _ in range(120):
        x = rng.randint(0, width - 1)
        y = rng.randint(0, height - 1)
        r = rng.choice((1, 1, 1, 1, 2))
        b = rng.randint(175, 240)
        c_draw.ellipse((x - r, y - r, x + r, y + r), fill=(b, b, 255, 165))

    for _ in range(10):
        x = rng.randint(8, width - 9)
        y = rng.randint(8, height - 9)
        c_draw.line((x - 2, y, x + 2, y), fill=(255, 255, 255, 130), width=1)
        c_draw.line((x, y - 2, x, y + 2), fill=(255, 255, 255, 130), width=1)

    composed.convert("RGB").save(ASSETS_DIR / "background-black.png")


def make_player_ship():
    img = Image.new("RGBA", (100, 90), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")

    glow = Image.new("RGBA", (100, 90), (0, 0, 0, 0))
    g_draw = ImageDraw.Draw(glow, "RGBA")
    g_draw.polygon(
        [(50, 8), (78, 24), (92, 50), (88, 74), (72, 88), (28, 88), (12, 74), (8, 50), (22, 24)],
        fill=(255, 220, 90, 85),
    )
    glow = glow.filter(ImageFilter.GaussianBlur(5))
    img = Image.alpha_composite(img, glow)
    draw = ImageDraw.Draw(img, "RGBA")

    draw.polygon(
        [(50, 10), (76, 24), (90, 46), (90, 72), (72, 88), (28, 88), (10, 72), (10, 46), (24, 24)],
        fill=(228, 184, 52, 255),
        outline=(255, 240, 150, 255),
    )
    draw.polygon([(50, 18), (68, 30), (62, 46), (38, 46), (32, 30)], fill=(94, 224, 255, 230))
    draw.polygon([(18, 44), (36, 44), (24, 72), (10, 72)], fill=(180, 130, 40, 255))
    draw.polygon([(82, 44), (64, 44), (76, 72), (90, 72)], fill=(180, 130, 40, 255))
    draw.rectangle((44, 48, 56, 82), fill=(250, 213, 87, 255))
    draw.rectangle((45, 56, 55, 82), fill=(255, 170, 70, 255))
    draw.polygon([(42, 82), (58, 82), (54, 89), (46, 89)], fill=(255, 110, 70, 220))

    img.save(ASSETS_DIR / "pixel_ship_yellow.png")


def make_enemy_ship_red():
    img = Image.new("RGBA", (70, 50), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.ellipse((10, 20, 60, 40), fill=(175, 40, 60, 255), outline=(235, 110, 120, 255))
    draw.rectangle((24, 10, 46, 24), fill=(130, 26, 44, 255), outline=(220, 95, 110, 255))
    draw.rectangle((30, 14, 40, 20), fill=(255, 170, 170, 235))
    draw.rectangle((14, 29, 20, 37), fill=(218, 78, 93, 255))
    draw.rectangle((50, 29, 56, 37), fill=(218, 78, 93, 255))
    draw.rectangle((27, 33, 33, 39), fill=(255, 145, 120, 255))
    draw.rectangle((37, 33, 43, 39), fill=(255, 145, 120, 255))
    img.save(ASSETS_DIR / "pixel_ship_red_small.png")


def make_enemy_ship_green():
    img = Image.new("RGBA", (70, 50), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.ellipse((10, 20, 60, 40), fill=(45, 146, 82, 255), outline=(130, 230, 168, 255))
    draw.rectangle((24, 10, 46, 24), fill=(33, 102, 58, 255), outline=(130, 230, 168, 255))
    draw.rectangle((30, 14, 40, 20), fill=(196, 255, 218, 230))
    draw.rectangle((14, 29, 20, 37), fill=(82, 186, 122, 255))
    draw.rectangle((50, 29, 56, 37), fill=(82, 186, 122, 255))
    draw.polygon([(30, 40), (35, 30), (40, 40)], fill=(156, 255, 192, 255))
    img.save(ASSETS_DIR / "pixel_ship_green_small.png")


def make_enemy_ship_blue():
    img = Image.new("RGBA", (50, 50), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")
    draw.polygon([(25, 10), (40, 24), (25, 40), (10, 24)], fill=(52, 95, 174, 255), outline=(146, 198, 255, 255))
    draw.polygon([(25, 15), (35, 24), (25, 34), (15, 24)], fill=(84, 151, 234, 230))
    draw.rectangle((21, 22, 29, 27), fill=(220, 246, 255, 235))
    draw.rectangle((17, 30, 21, 36), fill=(125, 185, 255, 255))
    draw.rectangle((29, 30, 33, 36), fill=(125, 185, 255, 255))
    img.save(ASSETS_DIR / "pixel_ship_blue_small.png")


def make_laser(filename, core, glow):
    img = Image.new("RGBA", (100, 90), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")

    x0, y0, x1, y1 = 45, 28, 54, 57
    draw.rectangle((x0 - 2, y0 - 2, x1 + 2, y1 + 2), fill=(*glow, 75))
    draw.rectangle((x0 - 1, y0 - 1, x1 + 1, y1 + 1), fill=(*glow, 110))
    draw.ellipse((44, 24, 55, 33), fill=(*glow, 110))
    draw.ellipse((45, 53, 54, 62), fill=(*glow, 110))
    draw.rectangle((x0, y0, x1, y1), fill=(*core, 255))
    draw.rectangle((48, y0, 51, y1), fill=(255, 255, 255, 185))

    img.save(ASSETS_DIR / filename)


def main():
    make_background()
    make_player_ship()
    make_enemy_ship_red()
    make_enemy_ship_green()
    make_enemy_ship_blue()
    make_laser("pixel_laser_yellow.png", core=(255, 223, 94), glow=(255, 170, 60))
    make_laser("pixel_laser_red.png", core=(255, 104, 114), glow=(255, 52, 82))
    make_laser("pixel_laser_green.png", core=(110, 255, 162), glow=(48, 212, 120))
    make_laser("pixel_laser_blue.png", core=(130, 198, 255), glow=(70, 144, 250))
    print("Generated assets in", ASSETS_DIR)


if __name__ == "__main__":
    main()
