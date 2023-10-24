import torch

def mask_accuracy(predicted_mask, real_mask):

  # Intersection and union for IoU
  intersection = torch.logical_and(predicted_mask, real_mask)
  union = torch.logical_or(predicted_mask, real_mask)
  iou = torch.sum(intersection).item() / torch.sum(union).item()

  # Intersection and Dice coefficient
  intersection = torch.sum(torch.logical_and(predicted_mask, real_mask)).item()
  dice = (2.0 * intersection) / (torch.sum(predicted_mask).item() + torch.sum(real_mask).item())

  return iou, dice