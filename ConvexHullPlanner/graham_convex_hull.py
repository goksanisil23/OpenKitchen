import math
import random
import sys

import pygame

# Initialize pygame
pygame.init()

# Screen dimensions
WIDTH, HEIGHT = 600, 400

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Convex Hull using Graham's Scan")


def random_points(num_points=30):
    return [
        (random.randint(0, WIDTH), random.randint(0, HEIGHT)) for _ in range(num_points)
    ]


def polar_angle(p0, p1):
    return math.atan2(p1[1] - p0[1], p1[0] - p0[0])


def distance(p0, p1):
    return (p1[0] - p0[0]) ** 2 + (p1[1] - p0[1]) ** 2


def graham_scan(points):
    start = min(points, key=lambda p: (p[1], p[0]))  # Lowest and then leftmost point
    sorted_points = sorted(
        points, key=lambda p: (polar_angle(start, p), distance(start, p))
    )

    if len(points) < 3:
        return []

    hull = [sorted_points[0], sorted_points[1]]

    for pt in sorted_points[2:]:
        while len(hull) > 1 and turn(hull[-2], hull[-1], pt) <= 0:
            hull.pop()
        hull.append(pt)

    return hull


def turn(p, q, r):
    return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])


points = random_points()
hull = graham_scan(points)

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)

    # Draw the convex hull
    pygame.draw.polygon(screen, RED, hull, 1)

    # Draw the random points
    for point in points:
        pygame.draw.circle(screen, BLUE, point, 5)

    pygame.display.flip()

pygame.quit()
sys.exit()
