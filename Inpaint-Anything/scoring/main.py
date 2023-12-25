import typing
from pathlib import Path

import torch

from mmedit.apis import init_model, restoration_inference


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model = init_model(
    "Inpaint-Anything/scoring/configs/clipiqa/clipiqa_coop_koniq.py", 
    "Inpaint-Anything/scoring/iter_80000.pth", 
    device=device,
)


def get_score(image_path: typing.Union[str, Path]):
    assert isinstance(image_path, str) or isinstance(image_path, Path), "The parameter 'image_path' should be string or pathlib.Path"
    
    output, attributes = restoration_inference(model, image_path, return_attributes=True)
    output = output.float().detach().cpu().numpy()
    score = attributes[0][0].detach().item()
    return score


if __name__=="__main__":
    image_folder = Path("test_images")
    for image_path in image_folder.iterdir():
        print(image_path)
        print(get_score(image_path))
