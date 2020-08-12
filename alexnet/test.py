import torchvision.models as models
alexnet_model = models.alexnet(pretrained=True)

new_model = models.alexnet(pretrained=True)
new_classifier = nn.Sequential(*list(new_model.classifier.children())[:-1])
new_model.classifier = new_classifier