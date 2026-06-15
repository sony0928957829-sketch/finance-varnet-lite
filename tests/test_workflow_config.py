from pathlib import Path
import unittest


class DailyMarketReportWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        workflow_path = (
            Path(__file__).resolve().parents[1]
            / ".github"
            / "workflows"
            / "daily-market-report.yml"
        )
        cls.workflow = workflow_path.read_text(encoding="utf-8")

    def test_schedule_and_manual_trigger_are_configured(self):
        self.assertIn("workflow_dispatch:", self.workflow)
        self.assertIn("push:", self.workflow)
        self.assertIn('"src/**"', self.workflow)
        self.assertIn('cron: "30 23 * * *"', self.workflow)
        self.assertIn('cron: "30 7 * * 1-5"', self.workflow)

    def test_failure_notification_has_issue_permission(self):
        self.assertIn("issues: write", self.workflow)
        self.assertIn("if: failure()", self.workflow)
        self.assertIn("gh issue create", self.workflow)
        self.assertIn("gh issue comment", self.workflow)

    def test_manual_failure_notification_can_be_exercised(self):
        self.assertIn("test_failure_notification:", self.workflow)
        self.assertIn("inputs.test_failure_notification", self.workflow)
        self.assertIn('exit 1', self.workflow)

    def test_alternative_data_and_supplemental_health_are_artifacts(self):
        self.assertIn("data/alternative/*.parquet", self.workflow)
        self.assertIn("data/reports/*_supplemental_health.json", self.workflow)
        self.assertIn('"2330.TW"', self.workflow)
        self.assertIn('"TAIEX"', self.workflow)
        self.assertIn('"derivatives.taiwan_options"', self.workflow)

    def test_tests_are_required_and_use_pytest(self):
        self.assertIn("python -m pytest tests -q", self.workflow)
        self.assertNotIn("continue-on-error: true", self.workflow)


if __name__ == "__main__":
    unittest.main()
