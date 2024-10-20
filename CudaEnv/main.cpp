#include "raylib.h"

#include "DummyBrain.h"

#include <iostream>

#include "Environment.h"

int main(void)
{
    Environment env;

    dummy_brain::DummyBrain dummy_brains_1(0, NUM_AGENTS_TEAM_A);
    dummy_brain::DummyBrain dummy_brains_2(NUM_AGENTS_TEAM_A, NUM_AGENTS_TEAM_B);

    while (!WindowShouldClose())
    {
        // Decision making from last sensing + Move
        dummy_brains_1.step(env.stream_, env.d_agents_, env.d_rays_);
        dummy_brains_2.step(env.stream_, env.d_agents_, env.d_rays_);

        env.step();
    }

    return 0;
}
