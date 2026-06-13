import unittest

from telegram_delivery_audit import chat_id_hash, classify_message


class TelegramDeliveryAuditTest(unittest.TestCase):
    def test_classifies_buy_signal_metadata(self) -> None:
        meta = classify_message(
            "🟢 *СИГНАЛ ПОКУПКИ* — 📈 Тренд\n\n"
            "*AXLUSDT*  `[15m]`\n"
            "💰 Цена: `0.0487`"
        )
        self.assertEqual(meta["message_kind"], "buy")
        self.assertEqual(meta["sym"], "AXLUSDT")
        self.assertEqual(meta["tf"], "15m")
        self.assertIn("AXLUSDT", meta["text_preview"])

    def test_classifies_sell_signal_metadata_without_markdown_symbol(self) -> None:
        meta = classify_message(
            "🔴 СИГНАЛ ПРОДАЖИ\n\n"
            "AXLUSDT  [15m]\n"
            "Выход: 0.0514"
        )
        self.assertEqual(meta["message_kind"], "sell")
        self.assertEqual(meta["sym"], "AXLUSDT")
        self.assertEqual(meta["tf"], "15m")

    def test_chat_id_hash_does_not_expose_raw_chat_id(self) -> None:
        raw = "179184487"
        hashed = chat_id_hash(raw)
        self.assertEqual(len(hashed), 16)
        self.assertNotIn(raw, hashed)


if __name__ == "__main__":
    unittest.main()
