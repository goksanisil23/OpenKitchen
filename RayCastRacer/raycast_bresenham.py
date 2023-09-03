import sys

import numpy as np
import pygame

# Initialize Pygame
pygame.init()

# Set up display variables
WIDTH, HEIGHT = 800, 600
win = pygame.display.set_mode((WIDTH, HEIGHT))

# Set up robot variables
robot_pos = np.array([WIDTH // 2, HEIGHT // 2])
robot_size = 20
robot_speed = 5
robot_angle = 0.0
ROBOT_COLOR = (0, 255, 0)

# Set up obstacle variables
# obstacle_pos = [np.array([200, 200]), np.array([600, 400])]
obstacle_size = 50

# Set up raycast variables
num_rays = 100
max_ray_length = 200  # maximum range of the raycast sensor

# Obstacle
OBSTACLE_X, OBSTACLE_Y = WIDTH // 2, HEIGHT // 2
OBSTACLE_WIDTH, OBSTACLE_HEIGHT = 50, 200
obstacle = set(
    (x, y)
    for x in range(OBSTACLE_X, OBSTACLE_X + OBSTACLE_WIDTH)
    for y in range(OBSTACLE_Y, OBSTACLE_Y + OBSTACLE_HEIGHT)
)
OBSTACLE_COLOR = pygame.Color("sienna")

SENSOR_COLOR = (255, 255, 0)
SENSOR_HIT_COLOR = (255, 0, 0)


# Bresenham's Line Algorithm
def bresenham(x0, y0, x1, y1):
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))
    return points


def check_collision(screen, ray_pixels):
    for px, py in ray_pixels:
        try:
            if screen.get_at((px, py))[:3] == OBSTACLE_COLOR[:3]:
                return px, py
        except:
            print((px, py))
    return None


def draw_environment():
    win.fill((0, 0, 0))

    # Draw obstacles
    for pos in obstacle:
        pygame.draw.rect(
            win,
            OBSTACLE_COLOR,
            (
                pos[0] - obstacle_size // 2,
                pos[1] - obstacle_size // 2,
                obstacle_size,
                obstacle_size,
            ),
        )

    # Draw robot
    pygame.draw.rect(
        win,
        ROBOT_COLOR,
        (
            robot_pos[0] - robot_size // 2,
            robot_pos[1] - robot_size // 2,
            robot_size,
            robot_size,
        ),
    )

    # Draw robot's heading
    end_pos = robot_pos + robot_size * np.array(
        [np.cos(robot_angle), np.sin(robot_angle)]
    )
    pygame.draw.line(win, (0, 0, 255), robot_pos, end_pos.astype(int), 3)

    # Check collision along rays
    angles = np.linspace(-np.pi / 2, np.pi / 2, num_rays) + robot_angle
    ray_origin = robot_pos + np.array(
        [robot_size / 2 * np.cos(robot_angle), robot_size / 2 * np.sin(robot_angle)]
    )
    for angle in angles:
        end_pos = ray_origin + max_ray_length * np.array([np.cos(angle), np.sin(angle)])
        # pygame.draw.line(win, (255, 255, 255), ray_origin, end_pos.astype(int))

        # Hit
        pixels_along_ray = bresenham(
            int(ray_origin[0]), int(ray_origin[1]), int(end_pos[0]), int(end_pos[1])
        )
        # Draw rays
        hit_point = check_collision(win, pixels_along_ray)
        if hit_point:
            pygame.draw.line(win, SENSOR_COLOR, ray_origin, hit_point)
            pygame.draw.circle(win, (0, 0, 255), hit_point, 3)
        else:
            pygame.draw.line(win, SENSOR_COLOR, ray_origin, end_pos)

    pygame.display.update()


def handle_input():
    keys = pygame.key.get_pressed()
    global robot_angle

    if keys[pygame.K_UP]:
        robot_pos[0] += robot_speed * np.cos(robot_angle)
        robot_pos[1] += robot_speed * np.sin(robot_angle)
    if keys[pygame.K_DOWN]:
        robot_pos[0] -= robot_speed * np.cos(robot_angle)
        robot_pos[1] -= robot_speed * np.sin(robot_angle)
    if keys[pygame.K_LEFT]:
        robot_angle -= 0.05
    if keys[pygame.K_RIGHT]:
        robot_angle += 0.05


def main():
    clock = pygame.time.Clock()

    # Game loop
    while True:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        handle_input()
        draw_environment()


main()
