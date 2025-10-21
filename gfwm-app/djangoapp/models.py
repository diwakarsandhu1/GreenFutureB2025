from django.db import models
from django.utils import timezone

class SurveyResponse(models.Model):
    environmental = models.IntegerField()
    social = models.IntegerField()
    governance = models.IntegerField()
    created_at = models.DateTimeField(default=timezone.now)

    def __str__(self):
        return f"SurveyResponse {self.id} - Environmental: {self.environmental}, Social: {self.social}, Governance: {self.governance}"