import torch
import torchvision.transforms as transforms
from PIL import Image

# Define the neural network architecture
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        self.pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1)
        self.fc1 = torch.nn.Linear(32 * 16 * 16, 256)
        self.fc2 = torch.nn.Linear(256, 32)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.nn.functional.relu(x)
        x = self.pool(x)
        x = x.view(-1, 32 * 16 * 16)
        x = self.fc1(x)
        x = torch.nn.functional.relu(x)
        x = self.fc2(x)
        return x

def recognize(image):
    items = ['A', 'B', 'C', 'D', "del", 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', "nothing", 'O', 'P', 'Q', 'R', 'S', "space", 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    # Load the PyTorch model
    model = Net()
    model.load_state_dict(torch.load('translater.pth'))
    model.eval()


    # Load and preprocess the image
    #image_path = 'export1.JPG'
    #image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    image = transform(image)

    # Get the prediction
    model.eval()
    with torch.no_grad():
        output = model(image)
    predicted_class = torch.argmax(output).item()

    print(f"The predicted class is {items[predicted_class]}")