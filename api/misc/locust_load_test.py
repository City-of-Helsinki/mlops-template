from locust import HttpUser, between, task


class WebsiteUser(HttpUser):
    wait_time = between(0, 1)

    predict_post_data = [
        {"sepal_length": 0, "sepal_width": 0, "petal_length": 0, "petal_width": 0}
    ]

    @task
    def predict(self):
        self.client.post("/predict", json=self.predict_post_data)


if __name__ == "__main__":
    print(
        "Use this file with locust, eg: locust -f locust_load_test.py -H http://127.0.0.1:8000"
    )
    print("How to perform load testing:")
    print("1. Run main.py or start api app as server")
    print("2. Run load testing and open browser at http://127.0.0.1:8000")
