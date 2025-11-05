from django.apps import AppConfig


class MyappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'djangoapp'

    def ready(self):
        from data_science.quant.preprocessing import ensure_preprocessing_artifacts
        from data_science.quant.covariance_calculations import ensure_covariance_artifacts
        try:
            ensure_preprocessing_artifacts()
            ensure_covariance_artifacts()
        except Exception as e:
            print(f"[startup] Failed to ensure artifacts: {e}")