import cv2
import torch
import torchvision
import torchvision.transforms as T
from PIL import Image

from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.structures import ImageList

ROI_HEADS_SCORE_THRESH_TEST = 0.5
MODEL_ARCH = "COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"
DEVICE = torch.device("cuda:0")


class RPNPredictor(DefaultPredictor):
    def __init__(self, cfg):
        super(RPNPredictor, self).__init__(cfg)

        self.pixel_mean = torch.tensor(cfg.MODEL.PIXEL_MEAN).view(-1, 1, 1)
        self.pixel_std = torch.tensor(cfg.MODEL.PIXEL_STD).view(-1, 1, 1)

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape
            (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():
            # Apply pre-processing to image.
            if self.input_format == "RGB":
                # whether the model expects BGR inputs or RGB
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(
                original_image).apply_image(original_image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))

            image = (image - self.pixel_mean) / self.pixel_std
            image = image.to(self.model.device)
            image = ImageList.from_tensors(
                [image], self.model.backbone.size_divisibility)

            feature = self.model.backbone(image.tensor)
            proposal, t = self.model.proposal_generator(image, feature)
            instance, t = self.model.roi_heads(image, feature, proposal)

            return instance[0]


class RoIFeatureExtractor:
    def __init__(self):
        self.model = torchvision.models.resnet50(pretrained=True)
        self.model.fc = torch.nn.Identity()
        self.model.eval()
        self.model.to(DEVICE)
        self.transform = T.Compose([
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _crop(self, image, instance):
        height, width = instance.image_size

        image = cv2.resize(image, (width, height))

        croped_images = []
        pred_boxes = instance.pred_boxes
        for box in iter(pred_boxes):
            x1 = max(0, min(width-1, int(box[0].item())))
            y1 = max(0, min(height-1, int(box[1].item())))
            x2 = max(x1, min(width-1, int(box[2].item())))
            y2 = max(y1, min(height-1, int(box[3].item())))

            croped_image = image[y1:y2, x1:x2]
            croped_images.append(self._cv2_to_pil(croped_image))
        return croped_images

    def _cv2_to_pil(self, image):
        img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

        return img

    def _transform_images(self, images):
        images = [self.transform(image) for image in images]
        images = torch.stack(images)
        return images

    def extract(self, image, instance):
        croped_images = self._crop(image, instance)
        inputs = self._transform_images(croped_images).to(DEVICE)
        with torch.no_grad():
            features = self.model(inputs)
        return features


# Configure experiment
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file(MODEL_ARCH))
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(MODEL_ARCH)

# Create Predictor
predictor = RPNPredictor(cfg)
feature_extractor = RoIFeatureExtractor()


def predict(image_path):
    im = cv2.imread(image_path)
    output = predictor(im)
    features = feature_extractor.extract(im, output)

    return features


if __name__ == "__main__":
    res = predict("input.jpg")
    print(res.shape)
