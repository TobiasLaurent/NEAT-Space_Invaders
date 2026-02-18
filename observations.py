from Object import Boss


def clamp_signed(value, scale):
    if scale == 0:
        return 0.0
    return max(-1.0, min(1.0, value / scale))


def build_observation(player, enemies, width, height):
    """Build richer, signed threat and self-state features for one player."""
    player_center_x = player.x + (player.get_width() / 2)
    player_center_y = player.y + (player.get_height() / 2)

    nearest_enemy_dist_sq = float("inf")
    nearest_enemy_dx = 0.0
    nearest_enemy_dy = 0.0

    nearest_laser_dist_sq = float("inf")
    nearest_laser_dx = 0.0
    nearest_laser_dy = height

    nearest_own_laser_dy_abs = float("inf")
    nearest_own_laser_dy = 0.0
    for laser in player.lasers:
        laser_center_y = laser.y + (laser.img.get_height() / 2)
        own_laser_dy = laser_center_y - player_center_y
        if abs(own_laser_dy) < nearest_own_laser_dy_abs:
            nearest_own_laser_dy_abs = abs(own_laser_dy)
            nearest_own_laser_dy = own_laser_dy

    boss_present = 0.0
    for enemy in enemies:
        if isinstance(enemy, Boss):
            boss_present = 1.0

        enemy_center_x = enemy.x + (enemy.get_width() / 2)
        enemy_center_y = enemy.y + (enemy.get_height() / 2)
        enemy_dx = enemy_center_x - player_center_x
        enemy_dy = enemy_center_y - player_center_y
        enemy_dist_sq = (enemy_dx * enemy_dx) + (enemy_dy * enemy_dy)
        if enemy_dist_sq < nearest_enemy_dist_sq:
            nearest_enemy_dist_sq = enemy_dist_sq
            nearest_enemy_dx = enemy_dx
            nearest_enemy_dy = enemy_dy

        for laser in enemy.lasers:
            laser_center_x = laser.x + (laser.img.get_width() / 2)
            laser_center_y = laser.y + (laser.img.get_height() / 2)
            laser_dx = laser_center_x - player_center_x
            laser_dy = laser_center_y - player_center_y
            laser_dist_sq = (laser_dx * laser_dx) + (laser_dy * laser_dy)
            if laser_dist_sq < nearest_laser_dist_sq:
                nearest_laser_dist_sq = laser_dist_sq
                nearest_laser_dx = laser_dx
                nearest_laser_dy = laser_dy

    return (
        max(0.0, min(1.0, player_center_x / width)),
        max(0.0, min(1.0, player_center_y / height)),
        clamp_signed(nearest_enemy_dx, width),
        clamp_signed(nearest_enemy_dy, height),
        clamp_signed(nearest_laser_dx, width),
        clamp_signed(nearest_laser_dy, height),
        min(1.0, len(enemies) / 20.0),
        boss_present,
        min(1.0, len(player.lasers) / 6.0),
        min(1.0, player.cool_down_counter / player.COOLDOWN),
        clamp_signed(nearest_own_laser_dy, height),
    )
