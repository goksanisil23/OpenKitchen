#pragma once

inline float normalizeAngleDeg(float angle)
{
    while (angle < 360.F)
    {
        angle += 360.F;
    }
    while (angle >= 360.F)
    {
        angle -= 360.F;
    }
    return angle;
}