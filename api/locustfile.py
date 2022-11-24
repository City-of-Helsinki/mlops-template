from locust import HttpUser, between, task


class WebsiteUser(HttpUser):
    wait_time = between(0, 1)

    predict_post_data = [
        {"sepal_length": 0, "sepal_width": 0, "petal_length": 0, "petal_width": 0}
    ]

    @task
    def predict(self):
        self.client.post("/predict", json=self.predict_post_data)
