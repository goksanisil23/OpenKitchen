#include "raylib.h"

int main(void)
{
    const int screenWidth  = 800;
    const int screenHeight = 600;

    InitWindow(screenWidth, screenHeight, "Raylib Example");

    RenderTexture2D target             = LoadRenderTexture(screenWidth, screenHeight);
    RenderTexture2D manipulationTarget = LoadRenderTexture(screenWidth, screenHeight);

    Vector2 playerPos   = {screenWidth / 2.0f, screenHeight / 2.0f};
    float   playerSpeed = 5.0f;

    Camera2D camera = {0};
    camera.target   = playerPos;
    camera.offset   = (Vector2){screenWidth / 2.0f, screenHeight / 2.0f};
    camera.rotation = 0.0f;
    camera.zoom     = 1.0f;

    SetTargetFPS(60);

    while (!WindowShouldClose())
    {
        if (IsKeyDown(KEY_RIGHT))
            playerPos.x += playerSpeed;
        if (IsKeyDown(KEY_LEFT))
            playerPos.x -= playerSpeed;
        if (IsKeyDown(KEY_DOWN))
            playerPos.y += playerSpeed;
        if (IsKeyDown(KEY_UP))
            playerPos.y -= playerSpeed;

        camera.target = playerPos;

        // Draw to manipulation target for pixel manipulation
        BeginTextureMode(manipulationTarget);
        ClearBackground(RAYWHITE);
        DrawRectangle(10, 10, 100, 100, RED);
        DrawCircleV(playerPos, 20, BLUE);
        EndTextureMode();

        // Pixel manipulation
        Image image = LoadImageFromTexture(manipulationTarget.texture);
        for (int y = 0; y < image.height; y++)
        {
            for (int x = 0; x < image.width; x++)
            {
                Color *pixel = &((Color *)image.data)[y * image.width + x];
                if (pixel->r > 0)
                {
                    pixel->g = 255;
                }
            }
        }
        UpdateTexture(manipulationTarget.texture, image.data);
        // UnloadImage(image);

        // // Draw manipulated texture to final target
        // BeginTextureMode(manipulationTarget);

        // EndMode2D();
        // EndTextureMode();

        // Draw to backbuffer
        BeginDrawing();
        ClearBackground(RAYWHITE);
        BeginMode2D(camera);
        DrawTexture(manipulationTarget.texture, 0, 0, WHITE);
        EndMode2D();
        // DrawTexture(target.texture, 0, 0, WHITE);
        EndDrawing();
    }

    UnloadRenderTexture(target);
    UnloadRenderTexture(manipulationTarget);
    CloseWindow();

    return 0;
}
