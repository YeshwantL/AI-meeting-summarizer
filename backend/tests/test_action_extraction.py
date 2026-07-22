import unittest

from app.ai.action_items import (
    calculate_confidence,
    detect_deadline,
    detect_owner,
    detect_priority,
    extract_action_items,
    is_blacklisted,
    merge_duplicates,
    rewrite_vague_task,
)
from app.ai.engine import extract_actions


class FilteringTests(unittest.TestCase):
    def test_fuzzy_blacklist_rejects_short_noise(self):
        for phrase in (
            "read that",
            "please ping me",
            "follow-up",
            "sounds good!",
            "thanks everyone",
            "take one minute",
            "check this",
            "look into it",
        ):
            with self.subTest(phrase=phrase):
                self.assertTrue(is_blacklisted(phrase))

    def test_bad_conversation_never_becomes_tasks(self):
        transcript = (
            "Please read that. Please ping me. Sounds good. Thanks everyone. "
            "Please take a minute to fill it out. Okay. Cool."
        )
        self.assertEqual(extract_action_items(transcript), [])

    def test_confidence_threshold_is_configurable(self):
        transcript = "Maya will prepare the final customer release report by Friday."
        self.assertEqual(len(extract_action_items(transcript, confidence_threshold=70)), 1)
        self.assertEqual(extract_action_items(transcript, confidence_threshold=100), [])


class MetadataTests(unittest.TestCase):
    def test_priority_detection(self):
        self.assertEqual(detect_priority("Fix the critical API issue tomorrow"), "High")
        self.assertEqual(detect_priority("Prepare the release report"), "Medium")
        self.assertEqual(detect_priority("Read the supporting material"), "Low")

    def test_owner_detection(self):
        self.assertEqual(detect_owner("Jessica send the report"), "Jessica")
        self.assertIsNone(detect_owner("Send the report"))

    def test_deadline_detection(self):
        for text, expected in (
            ("Send it today", "today"),
            ("Complete it tomorrow", "tomorrow"),
            ("Deliver before Monday", "before Monday"),
            ("Finish by EOD", "EOD"),
            ("Review by end of month", "end of month"),
        ):
            with self.subTest(text=text):
                self.assertEqual(detect_deadline(text), expected)

    def test_contextual_rewrite_requires_named_object(self):
        self.assertIsNone(rewrite_vague_task("go check it", "The team discussed several topics."))
        self.assertEqual(
            rewrite_vague_task("go check it", "The onboarding document may be inaccurate."),
            "Review the onboarding document and verify its accuracy",
        )


class PipelineTests(unittest.TestCase):
    def test_expected_business_action_json(self):
        transcript = "Jessica will update the onboarding document and ask all new hires to read this as well before Friday."
        tasks = extract_action_items(transcript, confidence_threshold=70)

        self.assertEqual(len(tasks), 1)
        task = tasks[0]
        self.assertEqual(task["task"], "Update the onboarding document and distribute it to all new hires.")
        self.assertEqual(task["owner"], "Jessica")
        self.assertEqual(task["priority"], "Medium")
        self.assertEqual(task["deadline"], "before Friday")
        self.assertGreaterEqual(task["confidence"], 90)
        self.assertIn("Concrete business action", task["reason"])

    def test_named_command(self):
        tasks = extract_action_items("Jessica send the final release report before Monday.", confidence_threshold=70)
        self.assertEqual(tasks[0]["owner"], "Jessica")
        self.assertEqual(tasks[0]["task"], "Send the final release report.")

    def test_duplicate_tasks_merge(self):
        tasks = [
            {"task": "Update the project documentation.", "owner": None, "deadline": None, "confidence": 78},
            {"task": "Update project documentation.", "owner": "Rahul", "deadline": "Friday", "confidence": 91},
        ]
        merged = merge_duplicates(tasks)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["owner"], "Rahul")

    def test_compatibility_alias_uses_new_pipeline(self):
        self.assertEqual(
            extract_actions("Please ping me."),
            extract_action_items("Please ping me."),
        )

    def test_confidence_helper_is_bounded(self):
        confidence, reason = calculate_confidence(
            "Prepare the critical customer release report.", "Maya", "tomorrow", "named"
        )
        self.assertLessEqual(confidence, 99)
        self.assertTrue(reason.endswith("."))


if __name__ == "__main__":
    unittest.main()
