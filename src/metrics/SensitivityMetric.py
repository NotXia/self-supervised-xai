# import torch
# import torch.nn.functional as F
# import torchvision

# from .BaseMetric import BaseMetric
# from xai import OnlyImageClassificationModel, OnlyTextClassificationModel


# class SensitivityMetric(BaseMetric):
#     def __init__(self, device):
#         super().__init__()
#         self.device = device


#     def accumulate(self, attr_orig, explainer, inputs, explainer_kwargs):
#         if isinstance(model, OnlyImageClassificationModel):
#             self.value += self.__sensitivity_image_classifier(attr_orig, explainer, inputs, explainer_kwargs)
#         elif isinstance(model, OnlyTextClassificationModel):
#             self.value += self.__sensitivity_text_classifier(attr_orig, explainer, inputs, explainer_kwargs)
#         else:
#             raise NotImplementedError()
#         self.count += 1


#     def __noise(self, size):
#         return 0.01 * torch.normal(0.0, 1.0, size)

#     @torch.no_grad()
#     def __sensitivity_image_classifier(self, attr_orig, explainer, inputs, explainer_kwargs):
#         attr_orig = attr_orig[0]
#         attr_orig = (attr_orig - attr_orig.min()) / (attr_orig.max() - attr_orig.min())
#         inputs_perturb = inputs["pixel_values"] + self.noise(inputs["pixel_values"].shape)

#         attr_perturb = explainer(inputs_perturb, **explainer_kwargs)[0]

#         return torch.norm(attr_orig - attr_perturb)


#     @torch.no_grad()
#     def __sensitivity_text_classifier(self, attr_orig, explainer, inputs, explainer_kwargs):
#         attr_orig = attr_orig[0]
#         attr_orig = (attr_orig - attr_orig.min()) / (attr_orig.max() - attr_orig.min())
#         inputs_perturb = inputs["input_ids"] + self.noise(inputs["input_ids"].shape)

#         attr_perturb = explainer(inputs_perturb, **explainer_kwargs)[0]

#         return torch.norm(attr_orig - attr_perturb)

#         attribution = (attribution - attribution.min()) / (attribution.max() - attribution.min())

#         logits = model(inputs["input_ids"], inputs["attention_mask"], attribution=None)
#         logits_masked = model(inputs["input_ids"], inputs["attention_mask"], attribution=attribution)
#         pred = torch.argmax(logits, dim=1)[0].item()

#         return max((logits[0, pred] - logits_masked[0, pred]).item(), 0)
