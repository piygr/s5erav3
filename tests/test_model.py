from models.model import MNISTModel

def test_model():
    model = MNISTModel()
    params = sum(p.numel() for p in model.parameters())
    assert params < 25000, f"Model has {params} parameters, exceeding the limit."

if __name__ == "__main__":
    test_model()
