import argparse
import torch
from PIL import Image
from torchvision import transforms
from model import RPSModel

classes = ['rock', 'paper', 'scissors']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RPSModel().to(device)
model.load_state_dict(torch.load('saved_model/rps_model2.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

parser = argparse.ArgumentParser()
parser.add_argument('--image', type=str, required=True)
args = parser.parse_args()

img = Image.open(args.image).convert('RGB')
img_tensor = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    output = model(img_tensor)
    _, predicted = torch.max(output, 1)

print(f"Prediction: {classes[predicted.item()]}")
```

### predict_webcam.py

```python
import torch
import cv2
from torchvision import transforms
from model import RPSModel

classes = ['rock', 'paper', 'scissors']
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = RPSModel().to(device)
model.load_state_dict(torch.load('saved_model/rps_model2.pth', map_location=device))
model.eval()

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((300, 300)),
    transforms.ToTensor()
])

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    x, y, w, h = 100, 100, 300, 300
    roi = frame[y:y+h, x:x+w]

    with torch.no_grad():
        input_tensor = transform(roi).unsqueeze(0).to(device)
        output = model(input_tensor)
        _, predicted = torch.max(output, 1)
        label = classes[predicted.item()]

    cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 255, 255), 2)
    cv2.putText(frame, f'Prediction: {label}', (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Rock Paper Scissors Classifier", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
