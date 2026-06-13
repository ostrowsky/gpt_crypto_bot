import unittest

from menu_text import is_hide_menu_text, is_open_menu_text


class MenuTextTest(unittest.TestCase):
    def test_open_menu_by_emoji(self) -> None:
        self.assertTrue(is_open_menu_text("📋 Открыть меню"))

    def test_open_menu_by_words(self) -> None:
        self.assertTrue(is_open_menu_text("открыть меню"))
        self.assertTrue(is_open_menu_text("open menu"))

    def test_hide_menu_by_emoji(self) -> None:
        self.assertTrue(is_hide_menu_text("🙈 Скрыть меню"))

    def test_other_text_is_not_menu(self) -> None:
        self.assertFalse(is_open_menu_text("AXLUSDT"))
        self.assertFalse(is_hide_menu_text("AXLUSDT"))


if __name__ == "__main__":
    unittest.main()
