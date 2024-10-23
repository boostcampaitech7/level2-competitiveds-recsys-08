import pickle

def save_model_to_pkl(models, name):
    # 피클로 저장
    lgb_save_path = "saved/models/"
    # Save each model using pickle
    for i, model in enumerate(models):
        model_filename = f"{lgb_save_path}{name}_model_fold_{i+1}.pkl"
        with open(model_filename, "wb") as file:
            pickle.dump(model, file)
        print(f"Model for fold {i+1} saved to {model_filename}")