import torch.nn as nn


class AttributePredictor(nn.Module):
    def __init__(self, attribute_sizes, image_encoder_output_dim, image_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.attribute_sizes = attribute_sizes
        self.attribute_predictors = nn.ModuleList(
            [nn.Linear(image_encoder_output_dim, size) for size in attribute_sizes])
        # Apply Kaiming initialization to the attribute predictors
        for predictor in self.attribute_predictors:
            nn.init.kaiming_normal_(predictor.weight, nonlinearity='relu')
            nn.init.zeros_(predictor.bias)

    def predict_from_features(self, x):
        # Predict each attribute
        outputs = [predictor(x) for predictor in self.attribute_predictors]
        return outputs

    def forward(self, x):
        x = self.image_encoder(x)
        x = x.view(x.size(0), -1)  # Flatten the image features
        outputs = self.predict_from_features(x)
        return outputs
