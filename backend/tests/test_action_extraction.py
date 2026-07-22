import unittest

from app.ai.engine import extract_actions


class ActionExtractionTests(unittest.TestCase):
    def test_keeps_explicit_commitments_and_assignments(self):
        text = (
            "Rahul will complete the backend by tomorrow and this is urgent. "
            "Rahul: I will fix the API before Friday. "
            "Maya will prepare the release notes. "
            "Action item: Priya to update the roadmap."
        )

        self.assertEqual(
            extract_actions(text),
            [
                {"task": "complete the backend", "owner": "Rahul", "deadline": "tomorrow", "priority": "High", "completed": False},
                {"task": "fix the API", "owner": "Rahul", "deadline": "Friday", "priority": "Medium", "completed": False},
                {"task": "prepare the release notes", "owner": "Maya", "deadline": None, "priority": "Medium", "completed": False},
                {"task": "update the roadmap", "owner": "Priya", "deadline": None, "priority": "Medium", "completed": False},
            ],
        )

    def test_rejects_discussion_hypotheticals_and_vague_statements(self):
        text = (
            "We should think about scalability. "
            "Why should they care? "
            "If needed, Maya will prepare slides tomorrow. "
            "I will give a shout out. "
            "The team discussed customer feedback."
        )

        self.assertEqual(extract_actions(text), [])

    def test_unnamed_commitment_requires_deadline_or_priority(self):
        text = "I will send the report by next week. I will review the design."

        self.assertEqual(len(extract_actions(text)), 1)
        self.assertEqual(extract_actions(text)[0]["task"], "send the report")

    def test_accepts_requests_and_informal_intentions(self):
        text = (
            "Please follow up with the customer. "
            "If you did not receive the survey, please ping me. "
            "I'm going to update the onboarding document. "
            "I'll ask Jessica to resend the invitation."
        )

        self.assertEqual(
            [item["task"] for item in extract_actions(text)],
            [
                "follow up with the customer",
                "ping me",
                "update the onboarding document",
                "ask Jessica to resend the invitation",
            ],
        )


if __name__ == "__main__":
    unittest.main()
