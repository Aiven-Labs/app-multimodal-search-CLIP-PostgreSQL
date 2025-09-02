# Notes

The developer center “ancestor” to this code, using OpenSearch and
~25,000 Unsplash images
https://aiven.io/developer/opensearch-multimodal-search (this URL will 
change, but should redirect to the new location, which may be in the
OpenSearch workshop mentioned next)

This workshop is based on the OpenSearch original, at
https://github.com/Aiven-Labs/workshop-multimodal-search-CLIP-OpenSearch


----

Introduction to CLIP
https://openai.com/index/clip/ 

Also reference to Llava https://llava-vl.github.io/ which relies on CLIP

----

openAI CLIP dates from 2021, so well established
Texts are encoded
Images are encoded
Those encodings are then aligned in a shared embedding space, enabling the model to perform tasks that require understanding the relationship between images and text.
This does mean it’s only going to do really well with text that falls within the text training sets.
And of course all the normal caveats about bias, inclusivity, data range and so on always apply.

**ViT-B/32**:  uses a Vision Transformer (ViT) as the image encoder, with an 
image patch size of 32 (larger patch size means fewer patches per image at a given size, and thus faster/smaller models) and the ‘Base’ size model. Larger models with smaller patch sizes will tend to need more memory and compute in exchange for better performance

The openAI CLIP GitHub repository: https://github.com/openai/CLIP gives example Python code similar to what we’re doing (but without the database storage). It also shows how to do the reverse - take an image and return its prediction/description of what the image contains.

`clip.load` returns
* `model`
* `preprocess` - the TorchVision transform needed by the model

`photos_preprocessed = torch.stack([preprocess(photo) for photo in photos]).to(device)`
* `preprocess` was returned by `clip.load`
* `torch.stack(...)` concatenates the array of results
* `.to(device)` …

`torch.nograd` 
* `model.encode_image` takes the concatenated data
* `a /= b` is (essentially) short for `a = a / b`
* `photos_features /= photos_features.norm(dim=-1, keepdim=True)` normalises 
  all the vectors so they’re the same “length”, which means we can compare distance between vectors

The openAI CLIP GitHub README also suggests looking at
* [OpenCLIP](https://github.com/mlfoundations/open_clip): includes larger 
  and independently trained CLIP models up to 
  ViT-G/14
* [Hugging Face implementation of CLIP](https://huggingface.co/docs/transformers/model_doc/clip): for easier integration with the HF 
  ecosystem

Example prompts
* cat; dog on chair;
* man jumping; 
* snowy streets; woolen hat; santa claus _– because it finds christmas 
  scenes?_



