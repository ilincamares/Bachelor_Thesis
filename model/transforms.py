import torchvision.transforms as T

positive_transform = T.Compose([
            T.Grayscale(),
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), shear=10),
            T.Resize((96, 96)),
            T.ToTensor(),
            T.Normalize((0.5,), (0.5,))
        ])

base_transform = T.Compose([
    T.Grayscale(), T.Resize((96, 96)), T.ToTensor(), T.Normalize((0.5,), (0.5,))
])