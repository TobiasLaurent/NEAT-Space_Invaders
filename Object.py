from Constants import *


class Ship:
    COOLDOWN = 30

    def __init__(self, x, y, health=100):
        self.x = x
        self.y = y
        self.health = health
        self.ship_img = None
        self.laser_img = None
        self.lasers = []
        self.cool_down_counter = 0

    def draw(self, window):
        window.blit(self.ship_img, (self.x, self.y))
        for laser in self.lasers:
            laser.draw(window)

    def move_lasers(self, obj):
        self.cooldown()
        laser_direction = 1    # direction in which the ship shoots: +1 for 'down' and -1 for 'up'
        for laser in self.lasers:
            laser.move(laser_direction)
            if laser.off_screen(HEIGHT):
                self.lasers.remove(laser)
            elif laser.collision(obj):
                obj.health -= 100           # kills player instantly
                # self.lasers.remove(laser)  # has to be done after iteration over all the players!
                # (optional since laser will be removed shortly after for leaving the screen)

    def cooldown(self):
        if self.cool_down_counter >= self.COOLDOWN:
            self.cool_down_counter = 0
        elif self.cool_down_counter > 0:
            self.cool_down_counter += 1

    def shoot(self):
        if self.cool_down_counter == 0:
            laser = Laser(self.x, self.y, self.laser_img)
            self.lasers.append(laser)
            self.cool_down_counter = 1

    def get_width(self):
        return self.ship_img.get_width()

    def get_height(self):
        return self.ship_img.get_height()


class Player(Ship):

    PLAYER_VEL = 5

    # ship of player
    YELLOW_SPACE_SHIP = pygame.image.load(os.path.join("assets", "pixel_ship_yellow.png"))

    # laser of player
    YELLOW_LASER = pygame.image.load(os.path.join("assets", "pixel_laser_yellow.png"))

    def __init__(self, x, y, health=100):
        super().__init__(x, y, health)
        self.ship_img = self.YELLOW_SPACE_SHIP
        self.laser_img = self.YELLOW_LASER
        self.mask = pygame.mask.from_surface(self.ship_img)
        self.max_health = health

    def move_lasers(self, objs):
        self.cooldown()
        laser_direction = -1  # direction in which the ship shoots: +1 for 'down' and -1 for 'up'
        for laser in self.lasers:
            laser.move(laser_direction)
            if laser.off_screen(HEIGHT):
                self.lasers.remove(laser)
            else:
                for obj in objs:
                    if laser.collision(obj):
                        objs.remove(obj)
                        if laser in self.lasers:
                            self.lasers.remove(laser)

    def draw(self, window):
        super().draw(window)
        # self.healthbar(window)

    def move_left(self):
        self.x -= self.PLAYER_VEL

    def move_right(self):
        self.x += self.PLAYER_VEL

    # a healthbar is not required when a single laser shot kills the player instantly
    # def healthbar(self, window):
    #    pygame.draw.rect(window, (255, 0, 0), (self.x, self.y +
    #                                           self.ship_img.get_height() + 10, self.ship_img.get_width(), 10))
    #    pygame.draw.rect(window, (0, 255, 0), (self.x, self.y + self.ship_img.get_height() +
    #                                           10, self.ship_img.get_width() * (self.health/self.max_health), 10))


class Enemy(Ship):

    # variations of ships
    RED_SPACE_SHIP = pygame.image.load(os.path.join("assets", "pixel_ship_red_small.png"))
    GREEN_SPACE_SHIP = pygame.image.load(os.path.join("assets", "pixel_ship_green_small.png"))
    BLUE_SPACE_SHIP = pygame.image.load(os.path.join("assets", "pixel_ship_blue_small.png"))

    # Laser variations
    RED_LASER = pygame.image.load(os.path.join("assets", "pixel_laser_red.png"))
    GREEN_LASER = pygame.image.load(os.path.join("assets", "pixel_laser_green.png"))
    BLUE_LASER = pygame.image.load(os.path.join("assets", "pixel_laser_blue.png"))

    COLOR_MAP = {
        "red": (RED_SPACE_SHIP, RED_LASER),
        "green": (GREEN_SPACE_SHIP, GREEN_LASER),
        "blue": (BLUE_SPACE_SHIP, BLUE_LASER)
    }

    ENEMY_VEL = 5

    def __init__(self, x, y, color, health=100):
        super().__init__(x, y, health)
        self.ship_img, self.laser_img = self.COLOR_MAP[color]
        self.mask = pygame.mask.from_surface(self.ship_img)

    def move(self):
        self.y += self.ENEMY_VEL

    def shoot(self):
        if self.cool_down_counter == 0:
            laser = Laser(self.x-20, self.y, self.laser_img)
            self.lasers.append(laser)
            self.cool_down_counter = 1


class Laser:

    LASER_VEL = 6

    def __init__(self, x, y, img):
        self.x = x
        self.y = y
        self.img = img
        self.mask = pygame.mask.from_surface(self.img)

    def draw(self, window):
        window.blit(self.img, (self.x, self.y))

    def move(self, direction):
        self.y += self.LASER_VEL * direction

    def off_screen(self, height):
        return not(self.y <= height and self.y >= 0)

    def collision(self, obj):
        return collide(self, obj)


# returns wether two objects collide or not
def collide(obj1, obj2):
    offset_x = obj2.x - obj1.x
    offset_y = obj2.y - obj1.y
    return obj1.mask.overlap(obj2.mask, (offset_x, offset_y)) != None
