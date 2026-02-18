from Object import Boss


def clamp_signed(value, scale):
    if scale == 0:
        return 0.0
    return max(-1.0, min(1.0, value / scale))


def build_observation(player, enemies, width, height):
    """Build observation with top-3 nearest enemies and top-3 nearest enemy lasers.

    Returns a 19-element tuple:
      [player_x_norm, player_y_norm,
       enemy1_dx, enemy1_dy, enemy2_dx, enemy2_dy, enemy3_dx, enemy3_dy,
       laser1_dx, laser1_dy, laser2_dx, laser2_dy, laser3_dx, laser3_dy,
       enemy_count_norm, boss_present,
       own_laser_count_norm, cooldown_norm, nearest_own_laser_dy]
    """
    player_center_x = player.x + (player.get_width() / 2)
    player_center_y = player.y + (player.get_height() / 2)

    enemy_slots = []   # (dist_sq, dx, dy)
    laser_slots = []   # (dist_sq, dx, dy)
    boss_present = 0.0

    for enemy in enemies:
        if isinstance(enemy, Boss):
            boss_present = 1.0

        enemy_center_x = enemy.x + (enemy.get_width() / 2)
        enemy_center_y = enemy.y + (enemy.get_height() / 2)
        dx = enemy_center_x - player_center_x
        dy = enemy_center_y - player_center_y
        dist_sq = dx * dx + dy * dy
        enemy_slots.append((dist_sq, dx, dy))

        for laser in enemy.lasers:
            laser_center_x = laser.x + (laser.img.get_width() / 2)
            laser_center_y = laser.y + (laser.img.get_height() / 2)
            laser_dx = laser_center_x - player_center_x
            laser_dy = laser_center_y - player_center_y
            laser_dist_sq = laser_dx * laser_dx + laser_dy * laser_dy
            laser_slots.append((laser_dist_sq, laser_dx, laser_dy))

    enemy_slots.sort(key=lambda e: e[0])
    laser_slots.sort(key=lambda l: l[0])

    def get_slot(slots, idx):
        if idx < len(slots):
            return clamp_signed(slots[idx][1], width), clamp_signed(slots[idx][2], height)
        return 0.0, 0.0

    e1dx, e1dy = get_slot(enemy_slots, 0)
    e2dx, e2dy = get_slot(enemy_slots, 1)
    e3dx, e3dy = get_slot(enemy_slots, 2)

    l1dx, l1dy = get_slot(laser_slots, 0)
    l2dx, l2dy = get_slot(laser_slots, 1)
    l3dx, l3dy = get_slot(laser_slots, 2)

    nearest_own_laser_dy_abs = float("inf")
    nearest_own_laser_dy = 0.0
    for laser in player.lasers:
        laser_center_y = laser.y + (laser.img.get_height() / 2)
        own_laser_dy = laser_center_y - player_center_y
        if abs(own_laser_dy) < nearest_own_laser_dy_abs:
            nearest_own_laser_dy_abs = abs(own_laser_dy)
            nearest_own_laser_dy = own_laser_dy

    return (
        max(0.0, min(1.0, player_center_x / width)),           # 1
        max(0.0, min(1.0, player_center_y / height)),          # 2
        e1dx, e1dy,                                             # 3, 4
        e2dx, e2dy,                                             # 5, 6
        e3dx, e3dy,                                             # 7, 8
        l1dx, l1dy,                                             # 9, 10
        l2dx, l2dy,                                             # 11, 12
        l3dx, l3dy,                                             # 13, 14
        min(1.0, len(enemies) / 20.0),                         # 15
        boss_present,                                           # 16
        min(1.0, len(player.lasers) / 6.0),                    # 17
        min(1.0, player.cool_down_counter / player.COOLDOWN),  # 18
        clamp_signed(nearest_own_laser_dy, height),            # 19
    )
