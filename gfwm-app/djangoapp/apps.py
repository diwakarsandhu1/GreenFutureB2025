from django.apps import AppConfig
import threading

class MyappConfig(AppConfig):
    default_auto_field = 'django.db.models.BigAutoField'
    name = 'djangoapp'

    def ready(self):
        from data_science.preprocess_and_filter.universal import ensure_universal_data
        from data_science.quant.optimized_markowitz.covariance_calculations import ensure_optimized_covariance_artifacts
        from data_science.quant.baseline_markowitz.covariance_calculations import ensure_baseline_covariance_artifacts
        
        def initialize_data():
            try:
                ensure_universal_data()
                ensure_optimized_covariance_artifacts()
                ensure_baseline_covariance_artifacts()
            except Exception as e:
                print(f"[startup] Failed to ensure artifacts: {e}")

        threading.Thread(target=initialize_data, daemon=True).start()
