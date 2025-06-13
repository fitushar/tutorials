import torch
from typing import Optional, Tuple

try:
    from transformers import CLIPModel, CLIPTokenizer

    _clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    _clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
    _clip_model.eval()
except Exception:  # pragma: no cover - fallback when transformers/torch not available
    _clip_tokenizer = None
    _clip_model = None


def _clip_region_weights(prompt: str) -> Optional[torch.Tensor]:
    """Compute region weights using CLIP text embeddings.

    Returns ``None`` if the CLIP model is unavailable.
    """

    if _clip_model is None:
        return None

    texts = [prompt, "head", "chest", "abdomen", "pelvis"]
    tokens = _clip_tokenizer(texts, return_tensors="pt", padding=True)
    with torch.no_grad():
        feats = _clip_model.get_text_features(**tokens)

    prompt_feat = feats[0]
    region_feats = feats[1:]
    sim = torch.nn.functional.cosine_similarity(region_feats, prompt_feat.unsqueeze(0), dim=1)
    weights = torch.nn.functional.softmax(sim, dim=0)
    return weights


def prompt_to_mask(prompt: str, output_size: Tuple[int, int, int]) -> torch.Tensor:
    """Create a simple mask tensor from a text prompt.

    This routine provides a very coarse mapping from anatomical keywords in
    ``prompt`` to regions of interest in the output mask.  The implementation is
    intentionally lightweight and does **not** attempt to produce an accurate
    medical segmentation.  It merely highlights approximate axial sections of
    the volume so that a prompt such as ``"abdomen"`` guides the ControlNet to
    the middle of the volume.

    Args:
        prompt: Input text prompt.
        output_size: Desired spatial size of the mask ``(H, W, D)``.

    Returns:
        torch.Tensor: Mask tensor of shape ``(1, 1, *output_size)``.
    """

    mask = torch.zeros((1, 1, *output_size), dtype=torch.uint8)
    zdim = output_size[2]

    weights = _clip_region_weights(prompt)
    if weights is not None:
        splits = torch.cumsum((weights * zdim).round().long(), dim=0)
        prev = 0
        for idx, end in enumerate(splits):
            mask[..., prev:end] = 1
            prev = end
    else:
        lower = prompt.lower()

        if any(k in lower for k in ["head", "neck"]):
            mask[..., : zdim // 4] = 1
        if any(k in lower for k in ["chest", "thorax", "lung"]):
            mask[..., zdim // 4 : zdim // 2] = 1
        if any(k in lower for k in ["abdomen", "liver", "kidney"]):
            mask[..., zdim // 2 : 3 * zdim // 4] = 1
        if any(k in lower for k in ["pelvis", "bladder", "leg"]):
            mask[..., 3 * zdim // 4 :] = 1

    return mask


def prompt_to_region_indices(prompt: str) -> tuple[list[int], list[int]]:
    """Infer body region indices from a prompt.

    The MAISI pipeline divides the body into four regions ordered from
    superior to inferior: head/neck, chest, abdomen and pelvis.  This helper
    searches for keywords in the prompt and returns one-hot encoded indices
    representing the top and bottom region indicated by the prompt.  If no
    keywords are found the full body is assumed.

    Args:
        prompt: Input text prompt.

    Returns:
        Tuple of lists representing ``top_region_index`` and
        ``bottom_region_index``.
    """

    regions = ["head", "chest", "abdomen", "pelvis"]

    weights = _clip_region_weights(prompt)
    if weights is not None:
        idx = torch.argmax(weights).item()
        active = [idx]
    else:
        lower = prompt.lower()
        keywords = [
            ["head", "neck"],
            ["chest", "thorax", "lung"],
            ["abdomen", "liver", "kidney"],
            ["pelvis", "bladder", "leg"],
        ]
        active = [i for i, keys in enumerate(keywords) if any(k in lower for k in keys)]

    if not active:
        active = [0, 3]

    top = [1 if i == active[0] else 0 for i in range(4)]
    bottom = [1 if i == active[-1] else 0 for i in range(4)]

    return top, bottom
