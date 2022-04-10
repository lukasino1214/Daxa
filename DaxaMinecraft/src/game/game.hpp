#pragma once

#include <utils/window.hpp>
#include <game/player.hpp>
#include <game/graphics.hpp>

#include <chrono>
#include <fmt/format.h>

static void HelpMarker(const char *desc) {
    if (ImGui::IsItemHovered()) {
        ImGui::BeginTooltip();
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

struct Game {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point prev_frame_time;

    Window window;

    VkSurfaceKHR vulkan_surface = window.get_vksurface(daxa::instance->getVkInstance());
    RenderContext render_context{vulkan_surface, window.frame_dim};
    std::optional<daxa::ImGuiRenderer> imgui_renderer = std::nullopt;

    World world{render_context};
    Player3D player;

    bool paused = true;
    bool perf_menu = true;

    std::array<float, 40> frametimes = {};
    u64 frametime_rotation_index = 0;
    std::string fmt_str;

    Game() {
        window.set_user_pointer<Game>(this);
        reset_player();
        reset_keybinds();

        ImGui::CreateContext();
        ImGui_ImplGlfw_InitForVulkan(window.window_ptr, true);
        imgui_renderer.emplace(render_context.device, render_context.queue, render_context.pipeline_compiler);
    }

    void reset_keybinds() {
        player.keybinds = input::DEFAULT_KEYBINDS;
    }

    void reset_player() {
        player = Player3D{};
        player.pos = glm::vec3(World::DIM * Chunk::DIM) / 2.0f;
        player.pos.y = -10.0f;
        player.rot = {0.001f, -0.6f, 0.0f};
    }

    void update() {
        auto now = Clock::now();
        float dt = std::chrono::duration<float>(now - prev_frame_time).count();
        prev_frame_time = now;
        frametimes[frametime_rotation_index] = dt;
        frametime_rotation_index = (frametime_rotation_index + 1) % frametimes.size();

        ImGuiIO &io = ImGui::GetIO();

        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        if (perf_menu) {
            ImGui::Begin("Debug", &perf_menu, ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoDecoration);

            {
                float average = 0.0f;
                for (auto frametime : frametimes)
                    average += frametime;
                average /= static_cast<float>(frametimes.size());

                fmt_str.clear();
                fmt::format_to(std::back_inserter(fmt_str), "avg {:.2f} ms ({:.2f} fps)", average * 1000, 1.0f / average);
                ImGui::PlotLines("", frametimes.data(), static_cast<int>(frametimes.size()), static_cast<int>(frametime_rotation_index), fmt_str.c_str(), 0, 0.05f, ImVec2(0, 120.0f));

                fmt_str.clear();
                fmt::format_to(std::back_inserter(fmt_str), "{:.2f} {:.2f} {:.2f}", player.pos.x, player.pos.y, player.pos.z);
                ImGui::Text("%s", fmt_str.c_str());
            }

            ImGui::End();
        }

        auto fire_ray = [&]() {
            world.single_ray_pos = player.pos;
            world.single_ray_nrm = player.camera.vrot_mat * glm::vec4(0, 0, 1, 0);
        };

        if (io.KeysDown[GLFW_KEY_E])
            fire_ray();

        if (paused) {
            ImGui::Begin("Settings");
            if (ImGui::Button("Fire ray"))
                fire_ray();
            ImGui::SliderInt("Ray steps", &world.single_ray_steps, 0, 100);
            if (ImGui::Button("Reset player"))
                reset_player();
            ImGui::SliderFloat("Speed", &player.speed, 0.1f, 40.0f);
            HelpMarker("Speed to move (Blocks/s)");
            ImGui::SliderFloat("Sprint Speed", &player.sprint_speed, 1.1f, 50.0f);
            HelpMarker("Sprint Multiplier");
            ImGui::SliderFloat("FOV", &player.camera.fov, 0.1f, 170.0f);
            HelpMarker("Vertical field of view (Degrees)");
            ImGui::SliderFloat("Sensitivity", &player.mouse_sens, 0.01f, 10.0f);
            HelpMarker("Mouse rotation speed (Radians/Pixels_moved/200)");
            if (ImGui::TreeNode("Keybinds")) {
                if (ImGui::Button("Reset"))
                    reset_keybinds();
                static i32 *selected_keyitem = nullptr;
                if (ImGui::BeginTable("table1", 2, ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders | ImGuiTableFlags_Resizable | ImGuiTableFlags_Reorderable | ImGuiTableFlags_Hideable)) {
                    ImGui::TableSetupColumn("Control", ImGuiTableColumnFlags_WidthFixed);
                    ImGui::TableSetupColumn("Keybind", ImGuiTableColumnFlags_WidthStretch);
                    ImGui::TableHeadersRow();

                    auto keycode_to_str = [](i32 key) -> const char * {
                        static char s[32] = {};
                        for (auto &c : s)
                            c = '\0';
                        if ((key >= 'a' && key <= 'z') || (key >= 'A' && key <= 'Z') || (key >= 'A' && key <= 'Z')) {
                            s[0] = static_cast<char>(key);
                        } else if (key == GLFW_KEY_LEFT_SHIFT || key == GLFW_KEY_RIGHT_SHIFT) {
                            s[0] = 'S', s[1] = 'h', s[2] = 'i', s[3] = 'f', s[4] = 't';
                        } else if (key == GLFW_KEY_LEFT_CONTROL || key == GLFW_KEY_RIGHT_CONTROL) {
                            s[0] = 'C', s[1] = 't', s[2] = 'r', s[3] = 'l';
                        } else if (key == GLFW_KEY_SPACE) {
                            s[0] = 'S', s[1] = 'p', s[2] = 'a', s[3] = 'c', s[4] = 'e';
                        } else {
                            s[0] = '?';
                        }
                        return s;
                    };

                    auto keybind_row = [&](const char *label, i32 &key) {
                        ImGui::TableNextRow();
                        ImGui::TableSetColumnIndex(0);
                        if (ImGui::Selectable(label, selected_keyitem == &key))
                            selected_keyitem = &key;
                        ImGui::TableSetColumnIndex(1);
                        ImGui::Text("%s", keycode_to_str(key));
                    };

                    keybind_row("Move forwards", player.keybinds.move_pz);
                    keybind_row("Move backwards", player.keybinds.move_nz);
                    keybind_row("Move left", player.keybinds.move_px);
                    keybind_row("Move right", player.keybinds.move_nx);
                    keybind_row("Ascend", player.keybinds.move_py);
                    keybind_row("Descend", player.keybinds.move_ny);
                    keybind_row("Sprint", player.keybinds.toggle_sprint);

                    ImGui::EndTable();
                }

                ImGui::TreePop();
                if (selected_keyitem != nullptr) {
                    for (int i = 0; i < IM_ARRAYSIZE(io.KeysDown); i++) {
                        if (ImGui::IsKeyPressed(i)) {
                            *selected_keyitem = i;
                            selected_keyitem = nullptr;
                            break;
                        }
                    }
                }
            }
            ImGui::End();
        }

        ImGui::Render();

        window.update();
        player.update(dt);
        world.update(dt);
        redraw();
    }

    void redraw() {
        auto cmd_list = render_context.begin_frame(window.frame_dim);
        player.camera.resize(window.frame_dim.x, window.frame_dim.y);
        player.camera.set_pos(player.pos);
        player.camera.set_rot(player.rot.x, player.rot.y);
        auto vp_mat = player.camera.vrot_mat;
        world.draw(vp_mat, player, cmd_list, render_context.render_color_image);
        render_context.blit_to_swapchain(cmd_list);
        imgui_renderer->recordCommands(ImGui::GetDrawData(), cmd_list, render_context.swapchain_image.getImageViewHandle());
        render_context.end_frame(cmd_list);
    }

    Window &get_window() {
        return window;
    }

    void on_mouse_move(const glm::dvec2 m) {
        if (!paused) {
            double center_x = static_cast<double>(window.frame_dim.x / 2);
            double center_y = static_cast<double>(window.frame_dim.y / 2);
            auto offset = glm::dvec2{m.x - center_x, center_y - m.y};
            player.on_mouse_move(offset.x, offset.y);
            window.set_mouse_pos(glm::vec2(center_x, center_y));
        }
    }
    void on_mouse_scroll(const glm::dvec2 offset) {
        if (!paused) {
            player.camera.fov -= static_cast<float>(offset.y);
            if (player.camera.fov < 10.0f)
                player.camera.fov = 10.0f;
            if (player.camera.fov > 170.0f)
                player.camera.fov = 170.0f;
        }
    }
    void on_mouse_button(int button, int action) {
        if (!paused && action == GLFW_PRESS) {
            switch (button) {
            case GLFW_MOUSE_BUTTON_LEFT:
                // world.should_break = true;
                break;
            case GLFW_MOUSE_BUTTON_RIGHT:
                // world.should_place = true;
                break;
            default: break;
            }
        }
    }
    void on_key(int key, int action) {
        if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
            toggle_pause();
        if (key == GLFW_KEY_F3 && action == GLFW_PRESS)
            perf_menu = !perf_menu;
        if (!paused) {
            player.on_key(key, action);
        }
    }
    void on_resize() {
        update();
    }

    void toggle_pause() {
        window.set_mouse_capture(paused);
        paused = !paused;
    }
};