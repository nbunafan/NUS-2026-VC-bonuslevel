import unittest

from metro_motion import Obstacle, RunnerGame


class MetroObjectLifecycleTest(unittest.TestCase):
    def game_with(self, obstacle):
        game = RunnerGame()
        game.started = True
        game.spawn_timer = 999.0
        game.obstacles = [obstacle]
        return game

    def test_collected_coin_disappears_in_under_quarter_second(self):
        game = self.game_with(Obstacle(lane=1, kind="coin", progress=0.899))
        game.update(0.01)
        self.assertEqual(game.coins, 1)
        self.assertTrue(game.obstacles[0].resolved)
        for _ in range(4):
            game.update(0.05)
        self.assertEqual(game.obstacles, [])

    def test_successfully_jumped_barrier_uses_fast_exit(self):
        game = self.game_with(Obstacle(lane=1, kind="barrier", progress=0.899))
        game.jump_remaining = 0.60
        game.update(0.01)
        self.assertEqual(game.lives, 3)
        self.assertEqual(game.successful_dodges, 1)
        self.assertEqual(game.combo, 1)
        self.assertTrue(game.obstacles[0].resolved)
        for _ in range(5):
            game.update(0.05)
        self.assertEqual(game.obstacles, [])

    def test_other_lane_object_keeps_normal_perspective_exit(self):
        game = self.game_with(Obstacle(lane=0, kind="coin", progress=0.899))
        game.update(0.01)
        self.assertEqual(game.coins, 0)
        self.assertFalse(game.obstacles[0].resolved)

    def test_air_coin_cannot_be_collected_on_ground(self):
        game = self.game_with(Obstacle(lane=1, kind="air_coin", progress=0.899))
        game.update(0.01)
        self.assertEqual(game.coins, 0)
        self.assertFalse(game.obstacles[0].resolved)

    def test_air_coin_awards_triple_value_near_jump_apex(self):
        game = self.game_with(Obstacle(lane=1, kind="air_coin", progress=0.839))
        game.jump_remaining = 0.43
        game.update(0.01)
        self.assertEqual(game.coins, 3)
        self.assertGreaterEqual(game.score, 75)
        self.assertTrue(game.obstacles[0].resolved)

    def test_three_of_each_coin_unlocks_nus_milestone_once(self):
        game = RunnerGame()
        game.started = True
        for _ in range(3):
            game._collect_coin(Obstacle(lane=1, kind="coin"), 1)
            game._collect_coin(Obstacle(lane=1, kind="air_coin"), 3)

        self.assertEqual(game.orange_coins_collected, 3)
        self.assertEqual(game.blue_coins_collected, 3)
        self.assertEqual(game.coins, 12)
        self.assertTrue(game.nus_milestone_unlocked)
        self.assertEqual(game.nus_animation_remaining, game.nus_animation_duration)

        # Additional coins must not restart the one-off cutscene.
        game.nus_animation_remaining = 0.5
        game._collect_coin(Obstacle(lane=1, kind="coin"), 1)
        self.assertEqual(game.nus_animation_remaining, 0.5)

    def test_nus_cutscene_temporarily_freezes_gameplay(self):
        game = self.game_with(Obstacle(lane=1, kind="barrier", progress=0.2))
        game.nus_animation_remaining = 1.0
        game.update(0.2)
        self.assertEqual(game.obstacles[0].progress, 0.2)
        self.assertAlmostEqual(game.nus_animation_remaining, 0.8)

    def test_start_gate_freezes_distance_score_and_spawning(self):
        game = RunnerGame()
        game.update(2.0)
        self.assertEqual(game.distance, 0.0)
        self.assertEqual(game.score, 0.0)
        self.assertEqual(game.obstacles, [])

    def test_speed_progression_depends_on_distance_not_bonus_score(self):
        game = RunnerGame()
        game.started = True
        game.spawn_timer = 999.0
        game.score = 10000
        game.update(1.0)
        self.assertLess(game.speed, 0.28)

    def test_five_successes_unlock_two_times_combo_multiplier(self):
        game = RunnerGame()
        game.started = True
        for _ in range(5):
            game._collect_coin(Obstacle(lane=1, kind="coin"), 1)
        self.assertEqual(game.combo, 5)
        self.assertEqual(game.combo_multiplier, 2)
        self.assertEqual(game.pickups_collected, 5)
        self.assertEqual(game.mission_index, 1)

    def test_combo_expires_after_four_seconds_without_success(self):
        game = RunnerGame()
        game.started = True
        game.spawn_timer = 999.0
        game._collect_coin(Obstacle(lane=1, kind="coin"), 1)
        game.update(4.01)
        self.assertEqual(game.combo, 0)

    def test_hit_invincibility_prevents_rapid_life_loss(self):
        game = RunnerGame()
        game.started = True
        game.spawn_timer = 999.0
        game._hit()
        game._hit()
        self.assertEqual(game.lives, 2)
        game.update(1.16)
        game._hit()
        self.assertEqual(game.lives, 1)

    def test_pause_freezes_world_state(self):
        game = self.game_with(Obstacle(lane=1, kind="barrier", progress=0.2))
        game.toggle_pause()
        game.update(0.5)
        self.assertEqual(game.obstacles[0].progress, 0.2)
        self.assertEqual(game.distance, 0.0)

    def test_repeated_jump_request_is_accepted_immediately_after_landing(self):
        game = RunnerGame()
        game.start()
        game.spawn_timer = 999.0
        game.command("JUMP")
        self.assertAlmostEqual(game.jump_remaining, 0.86)
        game.update(0.43)
        game.command("JUMP")
        self.assertAlmostEqual(game.jump_remaining, 0.43)
        game.update(0.43)
        game.command("JUMP")
        self.assertAlmostEqual(game.jump_remaining, 0.86)

    def test_spawn_wave_always_keeps_a_safe_lane(self):
        game = RunnerGame()
        game.distance = 1000.0
        for _ in range(100):
            game.obstacles = []
            game._spawn_wave()
            barrier_lanes = {obstacle.lane for obstacle in game.obstacles if obstacle.kind == "barrier"}
            self.assertLessEqual(len(barrier_lanes), 2)


if __name__ == "__main__":
    unittest.main()
