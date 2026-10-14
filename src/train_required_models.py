
def train_classifier(X, y, name, model):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y
    )

    model = make_pipeline(SimpleImputer(), model)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    with mlflow.start_run(run_name=name):
        mlflow.log_metric("accuracy", accuracy_score(y_test, predictions))
        mlflow.sklearn.log_model(
            model, name="model", serialization_format="pickle"
        )

    print(name, "completed")
